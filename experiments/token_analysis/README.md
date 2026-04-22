# `token_analysis`

**Hypotheses:** H2 — qualitative, per-token view of feature activations and logprob effects under ablation.

**Question:** At which token positions do specific SAE features activate, and what happens to per-token logprobs when we ablate them?

**Method:** YAML-configured per-token analysis. For each example (sentence or translation pair): extract SAE activations per token, ablate a chosen feature set, collect per-token Δ log p, render three-panel HTML (activation heatmap, logprob-delta heatmap, per-token table).

## Run

```bash
python experiments/token_analysis/run.py \
    --config experiments/token_analysis/configs/mood.yaml \
    --output_dir outputs/token_analysis
python experiments/token_analysis/visualize.py  # renders HTML
```

Batch-submission templates: `run/run_token_analysis*.sh` (one per config).

## Configs

- `configs/example.yaml` — baseline example.
- `configs/mood.yaml` — grammatical mood analyses.
- `configs/multilang.yaml` — multilingual translation examples.
- `configs/other_feats.yaml` — miscellaneous concepts.

## Inputs

- Probes at `outputs/probes/word_probes/*.joblib` — **note:** probe-filename-format mismatch silently disables probe filtering. See [TODO.md](../../TODO.md) ("analyze_tokens.py probe reader").
- Pre-computed input/output feature vectors.
- FLORES pairs / UD sentences as configured.

## Outputs

- `outputs/token_analysis/*.json` — one JSON per config, with per-token activations + logprobs.
- `outputs/token_analysis_html/*.html` — three-panel renderings.

## Status

Active. Probe-filtering disabled until the filename mismatch is fixed. See [LEDGER.md](../../LEDGER.md#token_analysis).
