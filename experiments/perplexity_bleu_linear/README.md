# `perplexity_bleu_linear`

**Hypotheses:** H1 — BLEU predictable from monolingual source and target competence.

**Question:** How much of BLEU variance across language pairs is explained by `BLEU ≈ a + β₁P_src + β₂P_tgt [+ β₃P_src·P_tgt]`? Low R² → cross-lingual interaction matters. High R² → monolingual competence is enough.

**Method:** Compute corpus perplexity per language on FLORES devtest. Compute Perplexity Error Rate (PER) per language on Multi-BLiMP minimal pairs. Join with external BLEU scores. Fit OLS with optional raw/log transform + optional interaction term. Visualize competence scatter and joint contours.

## Pipeline (run in order)

```bash
# 1. corpus perplexity per language per model
python experiments/perplexity_bleu_linear/run_perplexity.py --model_id llama --batch_size 8
python experiments/perplexity_bleu_linear/run_perplexity.py --model_id aya   --batch_size 8

# 2. perplexity error rate on Multi-BLiMP minimal pairs
python experiments/perplexity_bleu_linear/run_per.py --multilang --languages English Spanish French German Turkish Hebrew Hindi Chinese Indonesian

# 3. join BLEU + PPL
python experiments/perplexity_bleu_linear/combine_csvs.py

# 4. fit the linear model
python experiments/perplexity_bleu_linear/run_linear_fit.py --feature-transform raw --include-interaction yes
python experiments/perplexity_bleu_linear/run_linear_fit.py --feature-transform log --include-interaction no

# 5. visualize
python experiments/perplexity_bleu_linear/visualize_correlation.py
python experiments/perplexity_bleu_linear/visualize_error_bar.py
```

## Inputs

- FLORES devtest (HuggingFace `gsarti/flores_101`).
- Multi-BLiMP (HuggingFace `jumelet/multiblimp`).
- External BLEU scores (joined per-language-pair).
- Models: Llama-3.1-8B, Aya-23-8B.

## Outputs

- `outputs/perplexity_bleu/perplexity_results_{model}.csv` — raw PPL per language (tiny pilot files).
- `outputs/perplexity_bleu/combined_results_{model}.csv` — joined BLEU + PPL (current source of truth, ~28 KB).
- `outputs/perplexity_comparison/error_rates_by_language_{model}.json` — PER per language.
- `outputs/perplexity_comparison/perplexity_matrices_{model}.npz`.
- `outputs/perplexity_bleu/linear_models/linear_coeffs_{model}_{raw|log}_{joint|nojoint}.csv`.
- `outputs/perplexity_bleu/linear_models/linear_predictions_*.csv`.

## Figures

- `img/perplexity_bleu/{aya,llama}_{source,target,joint}_competence*.png` — current H1 figures.
- `img/perplexity_bleu/perplexity_plot_{model}.png` (+ `_sorted`).
- `img/perplexity_bleu/perplexity_vs_bleu_{model}_sorted.png`.

## Known caveats

- R² low (Llama=0.02, Aya=0.10). Rank-1 SVD approximation referenced in `docs/reference_paper.tex` has no code yet — **TODO to add here**. See [TODO.md](../../TODO.md).

## Status

Active. See [LEDGER.md](../../LEDGER.md#perplexity_bleu_linear).
