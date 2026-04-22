# `probes`

**Hypotheses:** INFRA — linear classifiers for grammatical concepts on SAE activations. Consumed by `output_features`, `ablation`, and `token_analysis`.

**Question:** Can we train reliable linear probes for `(language, concept, value, layer)` grammatical detection? Which layers are informative?

**Method:** cuML GPU logistic regression. Word-level tokens aligned via `word_ids()` (MWT-aware). 4-fold CV grid search over C ∈ logspace(−4, 3, 16), L2 penalty, balanced class weights, QN solver.

## Run

```bash
python experiments/probes/run.py \
    --languages English Spanish French German Turkish Arabic Hindi Chinese Indonesian \
    --output_dir outputs/probes
python experiments/probes/visualize.py
```

Batch-submission template: `run/run_probes.sh`.

## Inputs

- UD treebanks (training/test splits + concept tags).
- Pre-cached activations from `outputs/activations/` (layer 16 or 32 depending on target).

## Outputs

- `outputs/probes/word_probes/{Lang}_{Concept}_{Value}_l{layer}_n{max_samples}.joblib` — canonical filename format.
- `outputs/probes/all_probe_results_{YYYY-MM-DD_HH:MM}.csv` — grid-search summary CSV.
- `outputs/probes/processed_sentences/` — auto-generated pyconll parse cache (safe to delete to force re-parse).

## Figures

- `img/probe_performance/{C}_{V}_accuracy_vs_layer.png`
- `img/probe_performance/{C}_{V}_test_accuracy_distribution.png`
- `img/probe_performance/all_concepts_c_value_vs_layer.png`

## Known caveats

- **Probe filename format:** the canonical format is `l{layer}_n{n}.joblib`. `experiments/token_analysis/run.py` currently expects the older format `probe_layer{layer}_n{n}`; fix in [TODO.md](../../TODO.md).
- `docs/reference_paper.tex` Related Work says "mass-mean probes"; our code trains logistic regression. Terminology fix pending on the paper side.
- High probe accuracies (mean 96.68%, max 99.98%) — plausibly reflects lexical morphological predictability rather than deep grammatical encoding. Consider shuffled-label baselines.

## Status

Active (canonical training run 2025-11-22). See [LEDGER.md](../../LEDGER.md#probes-infra).
