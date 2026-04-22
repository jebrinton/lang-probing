# `input_output_overlap`

**Hypotheses:** H2 (the central claim).

**Question:** Do the top-K SAE features identified monolingually (input features) overlap with the top-K features driving a probe during translation (output features)? Across how many languages and concepts?

**Method:** Pure consumer. Load input features (`outputs/input_features/…/diff_vector.pt`) and output features (`outputs/output_features/…/effects_{Src}_{Tgt}.pt`); compute Jaccard at top-K, signal plots (top-k magnitude vs complementary-set rank), per-language distributions.

## Run

```bash
python experiments/input_output_overlap/visualize.py
```

## Inputs

- `outputs/input_features/{Lang}/{Concept}/{Value}/diff_vector.pt`
- `outputs/output_features/.../effects_{Src}_{Tgt}.pt`

## Outputs

- `img/input_output/{Lang}_{Concept}_{Value}/jaccard_topk.png`
- `img/input_output/{Lang}_{Concept}_{Value}/signal_input.png`
- `img/input_output/{Lang}_{Concept}_{Value}/signal_output.png`

Some signal-plot code paths are commented out in `visualize.py`; TODO to re-enable and verify.

## Status

Active. Needs a headline summary statistic (aggregate Jaccard@k) for the paper's central H2 claim. See [LEDGER.md](../../LEDGER.md#input_output_overlap).
