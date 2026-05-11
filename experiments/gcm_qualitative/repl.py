"""
Qualitative-harness REPL.

Usage:
    python experiments/gcm_qualitative/repl.py \\
        --config experiments/gcm_qualitative/configs/samples.yaml \\
        --port 8765

Loads the model + SAE once, starts an HTTP server on the given port, and
drops into a REPL. Each command (`observe`, `intervene`, ...) renders new
panels into `out/` and updates `manifest.json`; the dashboard polls and
shows them without a page reload.

Open the dashboard at  http://127.0.0.1:<port>/dashboard.html
On SCC: forward the port through VSCode's Ports panel.
"""
from __future__ import annotations

import argparse
import shlex
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# Add project root to sys.path so `lang_probing_src` and `experiments` import.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lang_probing_src.config import MODEL_ID, SAE_ID
from lang_probing_src.utils import setup_model

from experiments.gcm_qualitative.core import (
    Sample, build_sample, observe, intervene, InterveneOp,
)
from experiments.gcm_qualitative.render import (
    render_observe_result, render_intervene_result,
    write_manifest, reset_manifest,
)
from experiments.gcm_qualitative.server import start_server


HARNESS_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = HARNESS_DIR / "templates"
DEFAULT_OUT_DIR = HARNESS_DIR / "out"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


@dataclass
class HarnessConfig:
    samples: List[Sample]
    sae_features: List[int]
    attention_heads: List[Dict]   # [{"layer", "head", "modes": optional}]
    logit_lens_anchors: List[str]
    default_attn_modes: List[str]
    intervene_ops: List[Dict]
    intervene_max_new_tokens: int
    observe_max_response_tokens: int


def load_config(path: Path) -> HarnessConfig:
    with open(path) as f:
        d = yaml.safe_load(f)
    samples = [build_sample(s) for s in d.get("samples", [])]
    components = d.get("components", {})
    obs = d.get("observe", {})
    intv = d.get("intervene", {})
    return HarnessConfig(
        samples=samples,
        sae_features=list(components.get("sae_features", []) or []),
        attention_heads=list(components.get("attention_heads", []) or []),
        logit_lens_anchors=list(obs.get("logit_lens_anchors", ["last_src_token", "first_tgt_token"])),
        default_attn_modes=list(obs.get("default_attn_modes", ["iii"])),
        intervene_ops=list(intv.get("ops", []) or []),
        intervene_max_new_tokens=int(intv.get("max_new_tokens", 64)),
        observe_max_response_tokens=int(obs.get("max_response_tokens", 128)),
    )


# ---------------------------------------------------------------------------
# REPL state
# ---------------------------------------------------------------------------


class HarnessSession:
    def __init__(self, config_path: Path, out_dir: Path):
        self.config_path = config_path
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # Copy dashboard template into the served directory
        shutil.copy(TEMPLATE_DIR / "dashboard.html", out_dir / "dashboard.html")
        reset_manifest(out_dir)
        self.config: HarnessConfig = load_config(config_path)
        self.model = None
        self.submodule = None
        self.autoencoder = None
        self.tokenizer = None
        self.device = None

    def load_model(self):
        print("[repl] loading model + SAE — first time only, ~30–60s …", flush=True)

        import torch
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        # NOTE: We bypass `setup_model()` for the LanguageModel construction
        # to pass `attn_implementation="eager"` at load time. In current
        # transformers, eager attention is what makes `self_attn.output[1]`
        # carry a real tensor (sdpa returns None there); calling
        # `set_attn_implementation('eager')` post-hoc on the nnsight-wrapped
        # model didn't reliably propagate. We still use setup_autoencoder
        # via the original utility (in the external lang-similarity repo).
        sys.stdout.write("[repl] constructing LanguageModel with attn_implementation=eager …\n")
        sys.stdout.flush()
        from nnsight import LanguageModel
        from transformers import AutoTokenizer
        model = LanguageModel(
            MODEL_ID,
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            device_map=device,
            attn_implementation="eager",
        )
        submodule = model.model.layers[16]
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token

        # Load the SAE through the same path setup_model uses, but call the
        # external setup_autoencoder helper directly.
        sys.stdout.write("[repl] loading SAE checkpoint …\n"); sys.stdout.flush()
        from huggingface_hub import hf_hub_download
        sys.path.insert(0, "/projectnb/mcnet/jbrin/lang-similarity/src")
        from sae.utils import setup_autoencoder  # noqa: E402
        sae_filename = ("aya-23-8b-layer16.pt" if "aya" in MODEL_ID.lower()
                        else "llama-3-8b-layer16.pt")
        sae_path = hf_hub_download(repo_id=SAE_ID, filename=sae_filename)
        autoencoder = setup_autoencoder(sae_path)

        # Freeze grads (harness does no backward)
        try:
            model.requires_grad_(False)
            if autoencoder is not None:
                autoencoder.requires_grad_(False)
        except Exception:
            pass

        # Confirm the underlying HF model is now eager.
        hf_model = getattr(model, "_model", model)
        try:
            current = getattr(hf_model.config, "_attn_implementation", "?")
            sys.stdout.write(f"[repl]   model config._attn_implementation = {current!r}\n")
        except Exception:
            pass
        sys.stdout.flush()

        # Materialize the SAE on GPU. The existing gcm_translation pipeline
        # always uses the autoencoder *inside* `model.trace()` blocks where
        # nnsight implicitly materializes weights; we call it directly, so
        # we have to do it ourselves. Detect meta params and reload from the
        # local checkpoint if needed.
        if autoencoder is not None:
            sys.stdout.write("[repl] materializing autoencoder on GPU …\n"); sys.stdout.flush()
            self._materialize_autoencoder(autoencoder, device)
            sys.stdout.write("[repl]   autoencoder ready\n"); sys.stdout.flush()

        # Warmup trace — nnsight's LanguageModel keeps the HF model on `meta`
        # until the first `model.trace()` call. Run a tiny trace so the model
        # is dispatched on GPU and subsequent direct calls (e.g. the
        # intervene path's `hf_model.generate(...)`) don't see meta tensors.
        sys.stdout.write("[repl] warmup trace (dispatches model to GPU) …\n"); sys.stdout.flush()
        try:
            from lang_probing_src.config import TRACER_KWARGS
            import torch as _torch
            warm_ids = tokenizer("hi", return_tensors="pt").input_ids.to(device)
            with model.trace(warm_ids, **TRACER_KWARGS), _torch.no_grad():
                pass
            sys.stdout.write("[repl]   warmup OK\n"); sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(f"[repl]   WARNING: warmup trace failed: {e!r}\n"); sys.stdout.flush()

        self.model = model
        self.submodule = submodule
        self.autoencoder = autoencoder
        self.tokenizer = tokenizer
        print("[repl] model ready.", flush=True)

    @staticmethod
    def _materialize_autoencoder(autoencoder, device):
        """Ensure the autoencoder's weights are real (not meta) and on `device`."""
        import torch
        meta_params = [n for n, p in autoencoder.named_parameters() if p.device.type == "meta"]
        if meta_params:
            print(f"[repl] autoencoder has {len(meta_params)} meta-device params; "
                  f"reloading from checkpoint …", flush=True)
            from huggingface_hub import hf_hub_download
            sae_filename = ("aya-23-8b-layer16.pt" if "aya" in MODEL_ID.lower()
                            else "llama-3-8b-layer16.pt")
            ckpt_path = hf_hub_download(repo_id=SAE_ID, filename=sae_filename)
            state = torch.load(ckpt_path, weights_only=True, map_location="cpu")
            # to_empty allocates real storage on `device` for any meta params,
            # then load_state_dict copies the real weights in.
            autoencoder.to_empty(device=device)
            autoencoder.load_state_dict(state)
        autoencoder.to(device)
        # Make sure encode(x) doesn't trip over self.device == None either.
        if hasattr(autoencoder, "device"):
            autoencoder.device = device

    # -----------------------------------------------------------------
    # Commands
    # -----------------------------------------------------------------

    def cmd_help(self, *_):
        print(_HELP)

    def cmd_samples(self, *_):
        for i, s in enumerate(self.config.samples):
            print(f"  [{i}] {s.name}: {s.src_lang} → {s.tgt_lang}")
            print(f"      src: {s.src!r}")
            print(f"      tgt: {s.tgt!r}")

    def cmd_components(self, *_):
        print(f"  SAE features: {self.config.sae_features}")
        print(f"  Heads: {[(h['layer'], h['head']) for h in self.config.attention_heads]}")
        print(f"  Default attn modes: {self.config.default_attn_modes}")
        print(f"  Logit-lens anchors: {self.config.logit_lens_anchors}")

    def cmd_attn_modes(self, *args):
        if not args:
            print("  current default modes:", self.config.default_attn_modes)
            print("  usage: attn-modes i,ii,iii   (subset of {i,ii,iii})")
            return
        modes = [m.strip() for m in args[0].split(",")]
        bad = [m for m in modes if m not in {"i", "ii", "iii"}]
        if bad:
            print(f"  unknown modes: {bad} (must be subset of i, ii, iii)")
            return
        self.config.default_attn_modes = modes
        print(f"  default attention modes set to: {modes}")

    def cmd_observe(self, *args):
        targets = self._resolve_samples(args)
        if not targets:
            return
        run_dir = self.out_dir
        all_entries = []
        for s in targets:
            print(f"  [observe] {s.name} …", flush=True)
            try:
                result = observe(
                    self.model, self.submodule, self.autoencoder, self.tokenizer, self.device,
                    s,
                    sae_features=self.config.sae_features,
                    attention_heads=self.config.attention_heads,
                    logit_lens_anchors=self.config.logit_lens_anchors,
                    default_attn_modes=self.config.default_attn_modes,
                    max_response_tokens=self.config.observe_max_response_tokens,
                )
                entries = render_observe_result(result, run_dir)
                all_entries.extend(entries)
                print(f"            -> {len(entries)} panels.", flush=True)
            except Exception as e:
                print(f"  [observe] ERROR for {s.name}: {e!r}")
                import traceback; traceback.print_exc()
        if all_entries:
            write_manifest(run_dir, all_entries, run_label="observe")

    def cmd_intervene(self, *args):
        targets = self._resolve_samples(args)
        if not targets:
            return
        ops = [InterveneOp.from_dict(d) for d in self.config.intervene_ops]
        if not ops:
            print("  no intervene ops in config (intervene.ops). edit YAML and `reload`.")
            return
        all_entries = []
        for s in targets:
            for op in ops:
                print(f"  [intervene] {s.name} :: {op.label} …", flush=True)
                try:
                    result = intervene(
                        self.model, self.submodule, self.autoencoder, self.tokenizer, self.device,
                        s, op,
                        max_new_tokens=self.config.intervene_max_new_tokens,
                    )
                    entries = render_intervene_result(result, self.out_dir)
                    all_entries.extend(entries)
                except Exception as e:
                    print(f"  [intervene] ERROR for {s.name} :: {op.label}: {e!r}")
                    import traceback; traceback.print_exc()
        if all_entries:
            write_manifest(self.out_dir, all_entries, run_label="intervene")

    def cmd_clear(self, *_):
        reset_manifest(self.out_dir)
        # Also delete fragment files so the directory doesn't accumulate.
        for f in self.out_dir.glob("observe_*.html"):
            f.unlink()
        for f in self.out_dir.glob("intervene_*.html"):
            f.unlink()
        print("  manifest cleared.")

    def cmd_reload(self, *_):
        try:
            self.config = load_config(self.config_path)
            print(f"  reloaded {self.config_path.name}: "
                  f"{len(self.config.samples)} samples, "
                  f"{len(self.config.sae_features)} sae feats, "
                  f"{len(self.config.attention_heads)} heads, "
                  f"{len(self.config.intervene_ops)} ops")
        except Exception as e:
            print(f"  reload failed: {e!r}")

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _resolve_samples(self, args) -> List[Sample]:
        if not args or args[0] == "all":
            if not self.config.samples:
                print("  no samples in config — edit YAML and `reload`.")
                return []
            return list(self.config.samples)
        # Allow either index or name
        out: List[Sample] = []
        for a in args:
            try:
                idx = int(a)
                out.append(self.config.samples[idx])
                continue
            except (ValueError, IndexError):
                pass
            for s in self.config.samples:
                if s.name == a:
                    out.append(s); break
            else:
                print(f"  no sample matching {a!r}")
        return out


_HELP = """
commands:
  observe [<sample_name|index> ...]    forward pass + token-strip + logit lens
  intervene [<sample_name|index> ...]  baseline vs intervened generation
  samples                              list configured samples
  components                           list configured components
  attn-modes <subset of i,ii,iii>      change default attention aggregation modes
  reload                               re-read the YAML
  clear                                wipe the dashboard
  help                                 this message
  quit | exit                          leave the REPL

Open dashboard:  http://127.0.0.1:<port>/dashboard.html
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str,
                   default=str(HARNESS_DIR / "configs" / "samples.yaml"))
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT_DIR))
    p.add_argument("--no-server", action="store_true",
                   help="Skip the HTTP server (e.g. for non-interactive scripted use).")
    args = p.parse_args()

    out_dir = Path(args.out)
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[repl] config not found: {config_path}", file=sys.stderr); sys.exit(2)

    session = HarnessSession(config_path, out_dir)

    server = None
    if not args.no_server:
        server = start_server(out_dir, port=args.port)
        print(f"[repl] dashboard at  http://127.0.0.1:{args.port}/dashboard.html")
        print(f"[repl] (forward port {args.port} via VSCode if remote)")
    else:
        print(f"[repl] no-server mode; manifest in {out_dir}/manifest.json")

    session.load_model()
    print(_HELP)

    cmds = {
        "help": session.cmd_help, "?": session.cmd_help,
        "samples": session.cmd_samples,
        "components": session.cmd_components,
        "attn-modes": session.cmd_attn_modes, "modes": session.cmd_attn_modes,
        "observe": session.cmd_observe, "o": session.cmd_observe,
        "intervene": session.cmd_intervene, "i": session.cmd_intervene,
        "clear": session.cmd_clear,
        "reload": session.cmd_reload, "r": session.cmd_reload,
    }

    while True:
        try:
            line = input("(qualitative) > ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not line:
            continue
        if line in {"quit", "exit", "q"}:
            break
        try:
            tokens = shlex.split(line)
        except ValueError as e:
            print(f"  parse error: {e}"); continue
        cmd, *rest = tokens
        fn = cmds.get(cmd)
        if not fn:
            print(f"  unknown command: {cmd!r}; try `help`"); continue
        try:
            fn(*rest)
        except Exception as e:
            print(f"  command error: {e!r}")
            import traceback; traceback.print_exc()

    if server is not None:
        print("[repl] shutting down server.")
        server.shutdown()


if __name__ == "__main__":
    main()
