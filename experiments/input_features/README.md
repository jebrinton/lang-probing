# `input_features`

**Hypotheses:** H2 — features that read grammatical concepts into the model.

**Question:** Which SAE features fire differentially for positive vs. negative examples of a grammatical concept in a given language? How shared are those feature sets across languages?

**Method:** For each (language, concept, value), load UD sentences tagged positive/negative, extract SAE activations, mean-pool over tokens per sentence, compute `mean(positive) − mean(negative)` over sentences. Produces a 32768-dim diff vector per (language, concept, value).

## Run

```bash
# Current canonical — sentence level
python experiments/input_features/run.py
python experiments/input_features/visualize.py
```

## Configs

- `configs/sentence.yaml` — sentence-level (current canonical).
- `configs/word.yaml` — **stub for the word-level variant specified in `docs/reference_paper.tex` §"Word-level procedure"** (not yet implemented). TODO in `TODO.md`.

## Inputs

- UD treebanks via `ConlluDatasetPooled` (treebank="PUD").
- Cached residual-stream activations from `outputs/activations/` (from `experiments/activations_collection`).
- SAE at layer 16.

## Outputs

- `outputs/input_features/{Language}/{Concept}/{Value}/diff_vector.pt` — torch tensor, shape `[32768]`.

## Figures

- `img/input_features/feature_language_distribution_{C}_{V}_top{K}.png`
- `img/input_features/jaccard_similarity_{C}_{V}_top{K}.png`
- `img/input_features/{Lang}_{C}_{V}_magnitudes.png`

## Status

Active. See [LEDGER.md](../../LEDGER.md#input_features).
