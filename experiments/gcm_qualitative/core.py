"""
Qualitative-harness core: observe + intervene.

Loaded once, called repeatedly. Each function returns a dict that
render.py turns into one or more dashboard panels.

Mode A (observe) — full src + tgt, no intervention. Per requested
component (SAE feature or attention head) we emit a per-token activation
strip and a logit-lens projection at the configured anchor positions.

Mode B (intervene) — prompt only (2-shot ending with `<tgt_lang>: `),
generate baseline, register a PyTorch forward hook implementing the op,
generate again, compare. Also score log p(gold target | prompt) under
both for a quick quantitative readout.

Attention aggregation modes (configurable per component or globally):
    "i"   — mean attention paid by this head, averaged over all queries:
            attn[h, q, k].mean(q) → per-token weight on each key position
    "ii"  — attention to last_src_idx from each query:
            attn[h, q, last_src] → per-token weight on each query position
    "iii" — attention from last_src_idx as query (DEFAULT):
            attn[h, last_src, k] → per-token weight on each key position
            This matches the GCM patch site, so it's the most directly
            comparable to the universal-head IE rankings.
"""
from __future__ import annotations

import contextlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

# Match the existing experiment-import convention: sys.path-based, since
# the repo's __init__.py files are .gitignore'd (TODO P0 #1).
_GCM_DIR = Path(__file__).resolve().parent.parent / "gcm_translation"
if str(_GCM_DIR) not in sys.path:
    sys.path.insert(0, str(_GCM_DIR))

from gcm_core import tokenize_pair                         # noqa: E402
from flores_pairs import make_prompt, get_shots            # noqa: E402

from lang_probing_src.config import LAYER_NUM              # noqa: E402


# ---------------------------------------------------------------------------
# Sample dataclass
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    name: str
    src_lang: str
    tgt_lang: str
    src: str
    tgt: str = ""              # optional; observe uses it, intervene ignores
    shot_indices: tuple = (0, 1)


def build_sample(d: dict) -> Sample:
    return Sample(
        name=d["name"],
        src_lang=d["src_lang"],
        tgt_lang=d["tgt_lang"],
        src=d["src"],
        tgt=d.get("tgt", ""),
        shot_indices=tuple(d.get("shots", [0, 1])),
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def get_hf_model(model):
    """nnsight LanguageModel exposes the wrapped HF model as `_model`."""
    return getattr(model, "_model", model)


def get_hf_submodule(hf_model, layer_idx: int):
    """The raw torch nn.Module for a transformer block — needed for hooks."""
    return hf_model.model.layers[layer_idx]


def decoded_tokens(tokenizer, input_ids: torch.Tensor) -> List[str]:
    """Decode each token id individually so we can render a per-token strip."""
    ids = input_ids[0].tolist() if input_ids.ndim == 2 else input_ids.tolist()
    return [tokenizer.decode([i]) for i in ids]


@contextlib.contextmanager
def capture_o_proj_inputs(hf_model, n_layers: int):
    """Forward-pre hooks on every layer's o_proj that capture its input."""
    captures: List[Optional[torch.Tensor]] = [None] * n_layers
    handles = []

    def make_hook(layer_idx):
        def pre(_module, args):
            captures[layer_idx] = args[0].detach()
            return None
        return pre

    for L in range(n_layers):
        h = hf_model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(make_hook(L))
        handles.append(h)
    try:
        yield captures
    finally:
        for h in handles:
            h.remove()


def head_residual_contribution(
    o_proj_input: torch.Tensor,   # [B, S, d_model]
    W_o: torch.Tensor,            # [d_model, d_model]
    head_idx: int,
    head_dim: int,
) -> torch.Tensor:
    """The h-th head's contribution to the residual stream.

    o_proj is `out = W_o @ concat(head_0..head_{H-1})`. Slice column-wise.
    Returns [B, S, d_model].
    """
    sl = slice(head_idx * head_dim, (head_idx + 1) * head_dim)
    head_in = o_proj_input[:, :, sl]              # [B, S, head_dim]
    W_o_slice = W_o[:, sl]                         # [d_model, head_dim]
    return head_in @ W_o_slice.T                   # [B, S, d_model]


def logit_lens_topk(
    residual_contrib: torch.Tensor,   # [d_model]
    final_norm,
    lm_head,
    tokenizer,
    k: int = 8,
    apply_norm: bool = True,
) -> List[Dict]:
    """Project a residual contribution through (optional) final norm + lm_head."""
    x = residual_contrib.unsqueeze(0).unsqueeze(0)  # [1, 1, d]
    with torch.no_grad():
        if apply_norm:
            x = final_norm(x)
        logits = lm_head(x).squeeze(0).squeeze(0)   # [V]
    probs = torch.softmax(logits.float(), dim=-1)
    top = torch.topk(probs, k=k)
    return [
        {"token": tokenizer.decode([int(idx)]), "prob": float(p)}
        for p, idx in zip(top.values.tolist(), top.indices.tolist())
    ]


# ---------------------------------------------------------------------------
# Mode A — observe
# ---------------------------------------------------------------------------


def observe(
    model,
    submodule,           # passed in for compat; we use hf_model below
    autoencoder,
    tokenizer,
    device,
    sample: Sample,
    sae_features: List[int],
    attention_heads: List[Dict],     # [{"layer": int, "head": int, "modes": [...]}, ...]
    logit_lens_anchors: List[str],   # subset of {"last_src_token", "first_tgt_token"}
    default_attn_modes: List[str],
    max_response_tokens: int = 128,
) -> Dict:
    """Observe a single sample. See module docstring for the format."""
    hf_model = get_hf_model(model)
    cfg = hf_model.config
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    d_model = cfg.hidden_size
    head_dim = d_model // n_heads
    hf_submodule = get_hf_submodule(hf_model, LAYER_NUM)

    # 1. Build prompt + target
    shots = get_shots(sample.src_lang, sample.tgt_lang)
    if sample.shot_indices != (0, 1):
        # honor YAML override (rare path)
        from flores_pairs import _load_split
        from lang_probing_src.config import NAME_TO_LANG_CODE
        s = _load_split(NAME_TO_LANG_CODE[sample.src_lang])
        t = _load_split(NAME_TO_LANG_CODE[sample.tgt_lang])
        shots = [(s[i], t[i]) for i in sample.shot_indices]

    prompt = make_prompt(sample.src_lang, sample.tgt_lang, shots, sample.src)
    response = sample.tgt if sample.tgt else "(no target provided)"
    pr = tokenize_pair(tokenizer, prompt, response, device, max_response_tokens)
    tokens = decoded_tokens(tokenizer, pr.input_ids)
    last_src_idx = pr.last_src_idx
    first_tgt_idx = pr.response_start

    # 2. Single forward pass: capture L16 residual + every layer's o_proj input
    sae_features = list(sae_features) if sae_features else []
    layer16_resid = {"out": None}

    def cap_l16(_mod, _inputs, output):
        out_tensor = output[0] if isinstance(output, tuple) else output
        layer16_resid["out"] = out_tensor.detach()
        return output

    h_l16 = hf_submodule.register_forward_hook(cap_l16)
    try:
        with torch.no_grad(), capture_o_proj_inputs(hf_model, n_layers) as o_proj_caps:
            outputs = hf_model(
                pr.input_ids,
                output_attentions=True,
                use_cache=False,
            )
    finally:
        h_l16.remove()

    attentions = outputs.attentions   # tuple of [B, n_heads, S, S]

    # SAE acts at L16
    sae_acts = None
    if sae_features and layer16_resid["out"] is not None:
        l16 = layer16_resid["out"][0]                       # [S, d]
        with torch.no_grad():
            sae_acts = autoencoder.encode(l16).detach()     # [S, SAE_DIM]

    # 3. Build panels
    panels = []

    # --- SAE features ---
    for feat_idx in sae_features:
        if sae_acts is None:
            continue
        col = sae_acts[:, feat_idx].float().cpu().tolist()  # [S] activations
        panels.append({
            "kind": "sae",
            "id": f"f{feat_idx}",
            "title": f"SAE feature {feat_idx} (L16)",
            "subtitle": f"per-token activation across {len(tokens)} tokens",
            "strip": [{"token": tokens[i], "value": float(col[i])} for i in range(len(tokens))],
            "max_value": float(max(abs(v) for v in col)) if col else 0.0,
        })

    # --- Attention heads ---
    final_norm = hf_model.model.norm
    lm_head = hf_model.lm_head

    for h_spec in attention_heads:
        L = h_spec["layer"]
        H = h_spec["head"]
        modes = h_spec.get("modes") or default_attn_modes

        attn_LH = attentions[L][0, H]    # [S, S]   (B=1 → squeeze)

        strips = {}
        if "i" in modes:
            v = attn_LH.mean(dim=0).float().cpu().tolist()              # [S] over k
            strips["i"] = {
                "label": "(i) mean attention paid (averaged over queries)",
                "axis": "key positions",
                "data": [{"token": tokens[i], "value": float(v[i])} for i in range(len(tokens))],
            }
        if "ii" in modes:
            v = attn_LH[:, last_src_idx].float().cpu().tolist()         # [S] over q
            strips["ii"] = {
                "label": "(ii) attention to last_src_token from each query",
                "axis": "query positions",
                "data": [{"token": tokens[i], "value": float(v[i])} for i in range(len(tokens))],
            }
        if "iii" in modes:
            v = attn_LH[last_src_idx, :].float().cpu().tolist()         # [S] over k
            strips["iii"] = {
                "label": "(iii) attention from last_src_token (default — matches GCM patch)",
                "axis": "key positions",
                "data": [{"token": tokens[i], "value": float(v[i])} for i in range(len(tokens))],
            }

        # Logit lens
        ll = {}
        if o_proj_caps[L] is not None:
            o_in = o_proj_caps[L].float()                              # [1, S, d_model]
            W_o = hf_model.model.layers[L].self_attn.o_proj.weight.float()
            head_resid = head_residual_contribution(o_in, W_o, H, head_dim)[0]  # [S, d_model]
            for anchor in logit_lens_anchors:
                if anchor == "last_src_token":
                    pos = last_src_idx
                elif anchor == "first_tgt_token":
                    pos = min(first_tgt_idx, head_resid.shape[0] - 1)
                else:
                    continue
                ll[anchor] = {
                    "position_token": tokens[pos],
                    "topk_with_norm": logit_lens_topk(
                        head_resid[pos], final_norm, lm_head, tokenizer,
                        k=8, apply_norm=True,
                    ),
                    "topk_no_norm": logit_lens_topk(
                        head_resid[pos], final_norm, lm_head, tokenizer,
                        k=8, apply_norm=False,
                    ),
                }

        panels.append({
            "kind": "head",
            "id": f"L{L}.H{H}",
            "title": f"Layer {L} head {H}",
            "subtitle": f"kv-group {H // 4} (q-heads {(H//4)*4}-{(H//4)*4+3} share K/V)",
            "strips": strips,
            "logit_lens": ll,
        })

    return {
        "sample": {
            "name": sample.name,
            "src_lang": sample.src_lang,
            "tgt_lang": sample.tgt_lang,
            "src": sample.src,
            "tgt": sample.tgt,
        },
        "tokens": tokens,
        "prompt_len": pr.prompt_len,
        "last_src_idx": last_src_idx,
        "first_tgt_idx": first_tgt_idx,
        "panels": panels,
    }


# ---------------------------------------------------------------------------
# Mode B — intervene
# ---------------------------------------------------------------------------


@dataclass
class InterveneOp:
    kind: str            # "ablate_head" | "ablate_feature" | "steer_feature"
    layer: Optional[int] = None
    head: Optional[int] = None
    feature_idx: Optional[int] = None
    scale: float = 1.0
    positions: str = "all"   # "all" | "last_src_only"

    @classmethod
    def from_dict(cls, d):
        return cls(
            kind=d["kind"],
            layer=d.get("layer"),
            head=d.get("head"),
            feature_idx=d.get("feature_idx"),
            scale=float(d.get("scale", 1.0)),
            positions=d.get("positions", "all"),
        )

    @property
    def label(self) -> str:
        if self.kind == "ablate_head":
            return f"ablate L{self.layer}.H{self.head} ({self.positions})"
        if self.kind == "ablate_feature":
            return f"ablate SAE f{self.feature_idx} ({self.positions})"
        if self.kind == "steer_feature":
            return f"steer SAE f{self.feature_idx} ×{self.scale} ({self.positions})"
        return self.kind


def _hook_ablate_head(layer_module, head_idx: int, head_dim: int,
                      prompt_len: int, last_src_idx: int, positions: str):
    """Pre-hook on o_proj that zeros the head's slice of its input.

    With KV-cache active during decoding, this hook fires once with the full
    prompt then once per new token (seq=1). For positions='all' we ablate
    every call. For positions='last_src_only' we ablate only the call that
    has the matching prompt length AND only at the last_src position.
    """
    sl = slice(head_idx * head_dim, (head_idx + 1) * head_dim)

    def pre(_module, args):
        x = args[0].clone()
        if positions == "all":
            x[..., sl] = 0
        elif positions == "last_src_only":
            # Apply only on the single forward whose seq dim matches the prompt
            # (i.e. the prefill, before incremental decoding).
            if x.shape[1] == prompt_len:
                x[:, last_src_idx, sl] = 0
            # else: leave decode steps untouched
        else:
            raise ValueError(f"unknown positions: {positions}")
        return (x,) + args[1:]

    return layer_module.self_attn.o_proj.register_forward_pre_hook(pre)


def _hook_sae_intervene(submodule, autoencoder, feature_idx: int, scale: float,
                        kind: str, prompt_len: int, last_src_idx: int, positions: str):
    """Forward post-hook on L16 that zeroes (ablate) or scales/activates
    (steer) one SAE feature at the configured positions.

    Mechanic: encode → mutate f[feature_idx] → decode → add the diff
    (decoded_new - decoded_clean) to the residual. Same identity used by
    gcm_core's SAE patch.
    """
    def post(_module, _inputs, output):
        out_tensor = output[0] if isinstance(output, tuple) else output
        # Skip during decode if we only want to act on the prefill-prompt pass
        if positions == "last_src_only" and out_tensor.shape[1] != prompt_len:
            return output

        x = out_tensor                                # [B, S, d]
        f_clean = autoencoder.encode(x[0])            # [S, SAE_DIM]
        f_new = f_clean.clone()
        if positions == "last_src_only":
            target_pos = [last_src_idx]
        else:  # "all"
            target_pos = slice(None)

        if kind == "ablate_feature":
            f_new[target_pos, feature_idx] = 0
        elif kind == "steer_feature":
            base = f_clean[target_pos, feature_idx]
            inactive = (base.abs() < 1e-6).to(base.dtype)
            f_new[target_pos, feature_idx] = base * scale + scale * inactive
        else:
            raise ValueError(f"unknown SAE op kind: {kind}")

        d_clean = autoencoder.decode(f_clean)
        d_new = autoencoder.decode(f_new)
        delta = (d_new - d_clean).to(out_tensor.dtype)        # [S, d]
        out_tensor = out_tensor + delta.unsqueeze(0)
        if isinstance(output, tuple):
            return (out_tensor,) + output[1:]
        return out_tensor

    return submodule.register_forward_hook(post)


def intervene(
    model,
    submodule,           # nnsight wrapper (we use hf_submodule for hooks)
    autoencoder,
    tokenizer,
    device,
    sample: Sample,
    op: InterveneOp,
    max_new_tokens: int = 64,
) -> Dict:
    """Apply one op, generate, return baseline vs intervened text + Δ logp(gold)."""
    hf_model = get_hf_model(model)
    cfg = hf_model.config
    d_model = cfg.hidden_size
    n_heads = cfg.num_attention_heads
    head_dim = d_model // n_heads
    hf_submodule = get_hf_submodule(hf_model, LAYER_NUM)

    shots = get_shots(sample.src_lang, sample.tgt_lang)
    prompt = make_prompt(sample.src_lang, sample.tgt_lang, shots, sample.src)

    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    prompt_len = prompt_ids.shape[1]
    last_src_idx = prompt_len - 1

    def gen(ids):
        with torch.no_grad():
            out = hf_model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        return out

    # Baseline (no intervention)
    base_out = gen(prompt_ids)
    base_text = tokenizer.decode(base_out[0, prompt_len:], skip_special_tokens=True)

    def install_op():
        if op.kind == "ablate_head":
            return [_hook_ablate_head(
                hf_model.model.layers[op.layer], op.head, head_dim,
                prompt_len, last_src_idx, op.positions,
            )]
        elif op.kind in ("ablate_feature", "steer_feature"):
            return [_hook_sae_intervene(
                hf_submodule, autoencoder, op.feature_idx, op.scale, op.kind,
                prompt_len, last_src_idx, op.positions,
            )]
        else:
            raise ValueError(f"unknown op.kind: {op.kind}")

    # Intervened generation
    handles = install_op()
    try:
        intv_out = gen(prompt_ids)
        intv_text = tokenizer.decode(intv_out[0, prompt_len:], skip_special_tokens=True)
    finally:
        for h in handles:
            h.remove()

    # Score gold target log-prob under both, if provided
    gold_logp_baseline: Optional[float] = None
    gold_logp_intervened: Optional[float] = None
    if sample.tgt:
        gold_ids = tokenizer(sample.tgt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        full_ids = torch.cat([prompt_ids, gold_ids], dim=1)

        def score(ids):
            with torch.no_grad():
                logits = hf_model(ids).logits
            pred = logits[0, prompt_len - 1: ids.shape[1] - 1, :]
            tgt = ids[0, prompt_len: ids.shape[1]]
            return float(F.log_softmax(pred.float(), dim=-1)
                         .gather(-1, tgt.unsqueeze(-1)).sum())

        gold_logp_baseline = score(full_ids)
        handles = install_op()
        try:
            gold_logp_intervened = score(full_ids)
        finally:
            for h in handles:
                h.remove()

    return {
        "sample": {
            "name": sample.name,
            "src_lang": sample.src_lang,
            "tgt_lang": sample.tgt_lang,
            "src": sample.src,
            "tgt": sample.tgt,
        },
        "op": {
            "kind": op.kind,
            "label": op.label,
            "layer": op.layer,
            "head": op.head,
            "feature_idx": op.feature_idx,
            "scale": op.scale,
            "positions": op.positions,
        },
        "prompt_text": prompt,
        "baseline_text": base_text,
        "intervened_text": intv_text,
        "gold_logp_baseline": gold_logp_baseline,
        "gold_logp_intervened": gold_logp_intervened,
        "gold_logp_delta": (
            None if gold_logp_baseline is None
            else gold_logp_intervened - gold_logp_baseline
        ),
    }
