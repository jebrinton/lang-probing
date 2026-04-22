# `counterfactual_attribution`

**Hypotheses:** H2 — which SAE features mediate grammatical preferences?

**Question:** For each grammatical minimal pair (e.g., "The cat sat on the mat" vs. "… mats"), which SAE features explain the model's preference for one over the other?

**Method:** For each pair in `data/grammatical_pairs.json`, forward-pass with SAE encode/decode at layer 16, compute `logP(original) − logP(counterfactual)`, backpropagate to SAE feature activations. Rank by `grad` and `grad × activation` (indirect effect).

## Run

```bash
python experiments/counterfactual_attribution/run.py \
    --data_file data/grammatical_pairs.json \
    --output_dir outputs/counterfactual_attribution \
    --top_k 50
# add --save_raw_tensors to persist full per-token gradient tensors
```

Batch-submission template: `run/run_counterfactual_attribution.sh`.

After the main run, aggregate and visualize:

```bash
python experiments/counterfactual_attribution/analyze.py
python experiments/counterfactual_attribution/visualize.py
```

## Inputs

- `data/grammatical_pairs.json` — 30 English minimal pairs (prefix + original token + counterfactual token + concept label).
- Llama-3.1-8B + SAE layer 16.

## Outputs

- `outputs/counterfactual_attribution/aggregated_by_concept.json` — top-K features per concept.
- `outputs/counterfactual_attribution/per_pair_results.json` — raw per-pair rankings.
- `outputs/counterfactual_attribution/skipped_pairs.json` — pairs skipped (multi-token counterfactuals).
- `outputs/counterfactual_attribution/raw_gradients/{id}_{act,grad}.pt` (optional).

## Figures

- `outputs/counterfactual_attribution/plots/` — bar_top20, jaccard_top50, metric_distribution_by_concept, feature_scatter.

## Status

Prototype. English-only; per-concept samples are tiny (Polarity=2, Aspect/Mood=3). Multilingual extension pending — expand `grammatical_pairs.json`. See [LEDGER.md](../../LEDGER.md#counterfactual_attribution).
