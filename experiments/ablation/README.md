# `ablation`

**Hypotheses:** H2, H4 — causal check for shared multilingual features.

**Question:** Does zeroing out the top-K SAE features for a grammatical concept reduce the probability of the correct reference output? In both monolingual and translation settings?

**Method:** Zero-ablate top-K SAE features at source or target token positions (optionally restricted to probe-positive positions via `--use_probe`). Measure Δp(reference) = (exp(Δlog p) − 1). See `run.py` for the seven experiment configs.

## Run

```bash
# v3 canonical, probe-targeted, monolingual (March 2026 run)
python experiments/ablation/run.py \
    --experiment mono_input \
    --concept Number --value Sing --k 10 \
    --max_samples 256 --batch_size 8 \
    --use_probe --probe_layer 32 --probe_n 1024 \
    --output_dir outputs/ablation
```

`AAA_RUN.md` records the exact invocations used for the v3 production run.

Batch-submission template: `run/run_ablate.sh` (edit before qsub-ing).

## Configs

Seven experiment configs are defined in `run.py`'s `EXP_CONFIGS` dict:
- `mono_input`, `mono_output`, `mono_random` — monolingual
- `multi_input`, `multi_output`, `multi_input_random`, `multi_output_random` — translation

YAMLs under `configs/` are a TODO (today the config is CLI-driven). See [TODO.md](../../TODO.md) "Unify ablation config under YAML."

## Inputs

- **Input feature vectors:** `outputs/input_features/{Lang}/{Concept}/{Value}/diff_vector.pt` (from `experiments/input_features`).
- **Output feature effects:** `outputs/output_features/.../effects_{Src}_{Tgt}.pt` (from `experiments/output_features`).
- **Probes:** `outputs/probes/word_probes/{Lang}_{Concept}_{Value}_l{layer}_n{n}.joblib`.
- FLORES pairs (for translation prompts), Llama-3.1-8B + layer-16 SAE.

## Outputs

- `outputs/ablation/results_{experiment}.jsonl` — per-(src, tgt, concept, value, k) rows with `mean_delta`, `frac_active_at_ablated`, etc.

## Figures

- `img/ablation/barplot_source_{C}_{V}_{k}.png`, `barplot_target_{C}_{V}_{k}.png`

## Status

Active. v3 covers `mono_*` configs only; `multi_*` pending re-run with probe filter. See [LEDGER.md](../../LEDGER.md#ablation).
