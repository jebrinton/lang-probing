"""
Stage 1: Per-token SAE ablation analysis with caching.

Runs GPU inference for each experiment defined in a YAML config,
collecting per-token SAE activations and logprob deltas, and caches
the results as JSON files for Stage 2 visualization.

Usage:
    python scripts/analyze_tokens.py --config configs/token_analysis_example.yaml --output_dir outputs/token_analysis/
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from einops import rearrange

from lang_probing_src.ablate import logits_to_probs, get_probe_ablation_mask
from lang_probing_src.config import (
    MODEL_ID,
    SAE_ID,
    NAME_TO_LANG_CODE,
    TRACER_KWARGS,
)
from lang_probing_src.probe import load_probe
from lang_probing_src.utils import setup_model, get_device_info
from lang_probing_src.utils_input_output import (
    get_input_features_vector,
    get_output_features_vector,
    load_effects_files,
    get_language_pairs_and_concepts,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Copied from scripts/ablate.py (lives in a script, not the library)
# ---------------------------------------------------------------------------

def get_batch_positions_masks(tokenizer, contexts, sources, targets, device="cpu"):
    """
    Constructs the batch input_ids and boolean masks for source and target positions.
    """
    prompts = []
    for ctx, src, tgt in zip(contexts, sources, targets):
        if tgt:
            prompt = f"{ctx}{src} >> {tgt}"
        else:
            prompt = src
        prompts.append(prompt)

    encoding = tokenizer(prompts, padding=True, return_tensors="pt", return_offsets_mapping=True)
    input_ids = encoding["input_ids"].to(device)
    offset_mapping = encoding["offset_mapping"]

    batch_size, seq_len = input_ids.shape
    source_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
    target_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)

    for i in range(batch_size):
        ctx_len = len(contexts[i])
        src_len = len(sources[i])
        tgt_len = len(targets[i]) if targets[i] else 0

        src_start_char = ctx_len
        src_end_char = src_start_char + src_len

        separator_len = 4 if tgt_len > 0 else 0
        tgt_start_char = src_end_char + separator_len
        tgt_end_char = tgt_start_char + tgt_len

        offsets = offset_mapping[i]

        for j, (start, end) in enumerate(offsets):
            if start == end == 0:
                continue
            if start >= src_start_char and end <= src_end_char:
                source_mask[i, j] = True
            if tgt_len > 0 and start >= tgt_start_char and end <= tgt_end_char:
                target_mask[i, j] = True

    return input_ids, source_mask, target_mask


# ---------------------------------------------------------------------------
# Per-token ablation (single sample)
# ---------------------------------------------------------------------------

def ablate_and_collect_per_token(
    model, submodule, autoencoder, tokenizer,
    input_ids, feature_indices, ablate_mask, prob_mask,
):
    """
    Run ablation for a single sample and return full per-token arrays.

    Modeled on ablate_batch() but returns per-token logprob arrays instead
    of aggregated statistics.  Expects batch dimension = 1.

    Returns dict with:
        sae_activations: np.ndarray [S, K]  (pre-ablation SAE coefficients)
        logprob_orig:    list[float|None]    length S  (None at position 0)
        logprob_interv:  list[float|None]    length S
        logprob_delta:   list[float|None]    length S
    """
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)

    prob_mask[:, 0] = False
    feature_indices_np = np.asarray(feature_indices)

    with torch.no_grad():
        # --- intervention + clean in one trace ---
        with model.trace() as tracer:
            # Forward pass WITH ablation
            with tracer.invoke(input_ids):
                acts = submodule.output
                encoded_acts = autoencoder.encode(acts)
                encoded_acts_clean = encoded_acts.clone()

                selected_features = encoded_acts[:, :, feature_indices_np]
                pre_abl_slice = selected_features.clone().cpu().save()

                mask_reshaped = rearrange(ablate_mask, 'b s -> b s 1')
                ablated_slice = selected_features.masked_fill(mask_reshaped, 0.0)
                encoded_acts[:, :, feature_indices_np] = ablated_slice

                decoded_acts = autoencoder.decode(encoded_acts)
                decoded_acts_clean = autoencoder.decode(encoded_acts_clean)
                submodule.output = submodule.output + (decoded_acts - decoded_acts_clean)

                logits_interv = model.lm_head.output
                log_probs_interv = logits_to_probs(logits_interv, input_ids).cpu().save()

            # Forward pass WITHOUT ablation (clean baseline)
            with tracer.invoke(input_ids):
                logits_orig = model.lm_head.output
                log_probs_orig = logits_to_probs(logits_orig, input_ids).cpu().save()

    # Unwrap nnsight proxies
    log_probs_interv = getattr(log_probs_interv, "value", log_probs_interv)
    log_probs_orig = getattr(log_probs_orig, "value", log_probs_orig)
    pre_abl = getattr(pre_abl_slice, "value", pre_abl_slice)

    # sae_activations: [1, S, K] -> [S, K]
    sae_activations = pre_abl[0].detach().numpy()

    # logprobs are [1, S-1] — pad position 0 as None
    S = input_ids.shape[1]
    lp_orig = [None] + log_probs_orig[0].float().tolist()
    lp_interv = [None] + log_probs_interv[0].float().tolist()
    lp_delta = [None] + (log_probs_interv[0] - log_probs_orig[0]).float().tolist()

    return {
        "sae_activations": sae_activations,
        "logprob_orig": lp_orig,
        "logprob_interv": lp_interv,
        "logprob_delta": lp_delta,
    }


# ---------------------------------------------------------------------------
# Prompt construction helpers
# ---------------------------------------------------------------------------

def build_prompt_free_form(exp):
    """Return (context, source, target) for a free-form experiment entry."""
    if "source_text" in exp:
        # Multilingual free-form
        context = exp.get("context", "")
        source = exp["source_text"]
        target = exp["target_text"]
    else:
        # Monolingual free-form
        context = ""
        source = exp["text"]
        target = None
    return context, source, target


def build_prompt_flores(exp, num_shots=2):
    """Return (context, source, target) for a FLORES-based experiment entry."""
    source_lang = exp["source_lang"]
    i = exp["flores_index"]
    num_samples = 1012  # FLORES devtest size

    if i >= num_samples:
        raise ValueError(f"flores_index {i} out of range (max {num_samples - 1})")

    source_code = NAME_TO_LANG_CODE[source_lang]
    source_dataset = load_dataset("gsarti/flores_101", source_code, split="devtest")

    target_lang = exp.get("target_lang")
    if target_lang:
        target_code = NAME_TO_LANG_CODE[target_lang]
        target_dataset = load_dataset("gsarti/flores_101", target_code, split="devtest")

        # Build 2-shot context
        idx_1 = (i + 1) % num_samples
        idx_2 = (i + 2) % num_samples
        ctx = (
            f"{source_dataset[idx_2]['sentence']} >> {target_dataset[idx_2]['sentence']}\n"
            f"{source_dataset[idx_1]['sentence']} >> {target_dataset[idx_1]['sentence']}\n"
        )
        source = source_dataset[i]["sentence"]
        target = target_dataset[i]["sentence"]
    else:
        # Monolingual FLORES
        ctx = ""
        source = source_dataset[i]["sentence"]
        target = None

    return ctx, source, target


# ---------------------------------------------------------------------------
# Feature vector loading
# ---------------------------------------------------------------------------

INPUT_FEATURES_DIR = Path("/projectnb/mcnet/jbrin/lang-probing/outputs/sentence_input_features/")


def load_feature_vector(exp, effects_files=None, language_pairs_available=None):
    """
    Load or compute the feature importance vector for an experiment.

    Returns:
        np.ndarray of shape [SAE_DIM]
    """
    feats = exp.get("feats", "input")
    concept = exp["concept"]
    value = exp["value"]
    source_lang = exp.get("source_lang")
    target_lang = exp.get("target_lang")

    if feats == "input":
        return get_input_features_vector(INPUT_FEATURES_DIR, source_lang, concept, value)

    elif feats == "output":
        if effects_files is None:
            raise ValueError("effects_files required for output features")
        if target_lang:
            return get_output_features_vector(effects_files, (source_lang, target_lang), concept, value)
        else:
            # Monolingual output: average over all language pairs with matching source_lang
            if language_pairs_available is None:
                raise ValueError("language_pairs_available required for monolingual output features")
            src_langs = [src for src, tgt in language_pairs_available if tgt == source_lang]
            if not src_langs:
                # Try using source_lang as target
                src_langs = [src for src, tgt in language_pairs_available if tgt == source_lang]
            if not src_langs:
                raise ValueError(f"No language pairs found for {source_lang}")
            feat_vecs = []
            for sl in src_langs:
                try:
                    fv = get_output_features_vector(effects_files, (sl, source_lang), concept, value)
                    feat_vecs.append(fv)
                except (KeyError, TypeError) as e:
                    logger.warning("Skipping pair (%s, %s): %s", sl, source_lang, e)
            if not feat_vecs:
                raise ValueError(f"No valid output features for {source_lang}, {concept}={value}")
            return np.mean(feat_vecs, axis=0)

    elif feats == "random":
        # Random baseline — uniform random vector
        rng = np.random.RandomState(42)
        return rng.randn(32768).astype(np.float32)

    else:
        raise ValueError(f"Unknown feats type: {feats}")


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def _serialize_experiment(exp, tokenizer, context, source, target,
                          input_ids, source_mask, target_mask,
                          ablate_mask, prob_mask, feature_indices,
                          per_token_result):
    """Build the JSON-serialisable cache dict for one experiment."""
    S = input_ids.shape[1]
    ids_list = input_ids[0].tolist()

    tokens = []
    for pos in range(S):
        tokens.append({
            "id": ids_list[pos],
            "text": tokenizer.decode([ids_list[pos]]),
            "position": pos,
        })

    sae_vals = per_token_result["sae_activations"]  # [S, K]

    return {
        "metadata": {
            "name": exp["name"],
            "source_lang": exp.get("source_lang"),
            "target_lang": exp.get("target_lang"),
            "concept": exp["concept"],
            "value": exp["value"],
            "k": exp.get("k"),
            "feats": exp.get("feats", "input"),
            "ablate_loc": exp.get("ablate_loc", "source"),
            "prob_loc": exp.get("prob_loc", "source"),
            "use_probe": exp.get("use_probe", False),
            "feature_indices": [int(x) for x in feature_indices],
        },
        "tokens": tokens,
        "masks": {
            "source_mask": source_mask[0].tolist(),
            "target_mask": target_mask[0].tolist(),
            "ablate_mask": ablate_mask[0].tolist(),
            "prob_mask": prob_mask[0].tolist(),
        },
        "sae_activations": {
            "feature_ids": [int(x) for x in feature_indices],
            "values": sae_vals.tolist(),
        },
        "logprobs": {
            "original": per_token_result["logprob_orig"],
            "intervention": per_token_result["logprob_interv"],
            "delta": per_token_result["logprob_delta"],
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(exp, model, submodule, sae, tokenizer, device,
                   effects_files=None, language_pairs_available=None):
    """Run a single experiment entry and return the cache dict."""
    name = exp["name"]
    logger.info("Running experiment: %s", name)

    # --- Prompt construction ---
    prompt_type = exp["prompt_type"]
    if prompt_type == "free_form":
        context, source, target = build_prompt_free_form(exp)
    elif prompt_type == "flores":
        context, source, target = build_prompt_flores(exp)
    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type}")

    # --- Tokenize + masks ---
    input_ids, source_mask, target_mask = get_batch_positions_masks(
        tokenizer, [context], [source], [target], device=device,
    )
    S = input_ids.shape[1]

    # --- Ablation / probability masks ---
    ablate_loc = exp.get("ablate_loc", "source")
    prob_loc = exp.get("prob_loc", "source")
    ablate_mask = source_mask.clone() if ablate_loc == "source" else target_mask.clone()
    prob_mask = source_mask.clone() if prob_loc == "source" else target_mask.clone()

    # --- Probe-based mask refinement ---
    if exp.get("use_probe", False):
        probe_layer = exp.get("probe_layer", 16)
        probe_n = exp.get("probe_n", 1024)
        concept = exp["concept"]
        value = exp["value"]
        source_lang = exp.get("source_lang", "")
        probe_path = (
            Path("/projectnb/mcnet/jbrin/lang-probing/outputs/word_probes")
            / source_lang / concept / value
            / f"probe_layer{probe_layer}_n{probe_n}.joblib"
        )
        if probe_path.exists():
            probe = load_probe(probe_path)
            ablate_mask = get_probe_ablation_mask(
                model, input_ids, probe, probe_layer, ablate_mask, device=str(device),
            )
        else:
            logger.warning("Probe not found at %s — skipping probe masking", probe_path)

    # --- Handle empty ablation mask ---
    if not ablate_mask.any():
        logger.warning("Empty ablation mask for %s — SAE activations still collected but deltas zeroed", name)

    # --- Feature vector + top-K indices ---
    k = exp.get("k", 5)
    exp["k"] = k  # ensure it's stored
    try:
        feat_vec = load_feature_vector(exp, effects_files, language_pairs_available)
    except (FileNotFoundError, KeyError, ValueError) as e:
        logger.warning("Skipping %s: %s", name, e)
        return None
    feature_indices = np.argsort(feat_vec)[-k:]

    # --- Per-token ablation ---
    result = ablate_and_collect_per_token(
        model, submodule, sae, tokenizer,
        input_ids, feature_indices, ablate_mask, prob_mask,
    )

    # --- Serialize ---
    return _serialize_experiment(
        exp, tokenizer, context, source, target,
        input_ids, source_mask, target_mask,
        ablate_mask, prob_mask, feature_indices,
        result,
    )


def main(args):
    # --- Load config ---
    with open(args.config) as f:
        config = yaml.safe_load(f)

    defaults = config.get("defaults", {})
    experiments = config["experiments"]

    # Merge defaults into each experiment
    for exp in experiments:
        for k, v in defaults.items():
            exp.setdefault(k, v)

    # --- Model setup ---
    logger.info("Loading model and SAE...")
    device, _ = get_device_info()
    model, submodule, sae, tokenizer = setup_model(MODEL_ID, SAE_ID)

    # --- Load effects files if any experiment needs output features ---
    effects_files = None
    language_pairs_available = None
    if any(exp.get("feats") == "output" for exp in experiments):
        logger.info("Loading effects files for output features...")
        effects_files = load_effects_files()
        language_pairs_available, _ = get_language_pairs_and_concepts(effects_files)

    # --- Output directory ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Run each experiment ---
    for exp in experiments:
        cache = run_experiment(
            exp, model, submodule, sae, tokenizer, device,
            effects_files=effects_files,
            language_pairs_available=language_pairs_available,
        )
        if cache is None:
            continue

        out_path = output_dir / f"{exp['name']}.json"
        with open(out_path, "w") as f:
            json.dump(cache, f, indent=2)
        logger.info("Saved: %s", out_path)

    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-token SAE ablation analysis (Stage 1)")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--output_dir", required=True, help="Directory for JSON output files")
    args = parser.parse_args()
    main(args)
