# `activations_collection`

**Hypotheses:** INFRA — residual-stream activation cache for downstream use.

**Question:** n/a.

**Method:** Load UD treebanks, trace Llama-3.1-8B via nnsight, extract mean-pooled residual-stream activations per layer (from `model.model.layers[N]`), partition and save as parquet.

## Run

```bash
python experiments/activations_collection/run.py \
    --languages English Spanish French German Turkish Arabic Hindi Chinese Indonesian
```

(Currently needs to be submitted as a GPU job on SCC — see [LAB_NOTEBOOK.md](../../LAB_NOTEBOOK.md) 2026-04-22 entry for context.)

## Outputs

- `outputs/activations/language={Language}/layer={N}/data.parquet` — schema: `sentence_id, language, sentence_text, layer, tags, activation`.

23 languages × 9 layers (layer 31 is the final Llama layer in the `model.model.layers` indexing convention) produces 207 parquet files, ~70 MB each.

## Layer convention gotcha

`config.py` defines both:
- `LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 32]` — PyTorch `hidden_states` convention (33-element tuple; index 32 = final layer output).
- `COLLECTION_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]` — `model.model.layers` module-list convention (32 modules, index 31 = final layer).

Both refer to the same physical layer. Not a bug; see [LAB_NOTEBOOK.md](../../LAB_NOTEBOOK.md) 2026-04-22 for the investigation.

## Known caveats

- `run.py:108` — `args.languages` referenced inside `main()` as a module-global (works but ugly).
- `run.py:37-42` — silent exception swallows UD load failures and crashes later on NameError.
- `run.py:112` — **hardcoded `for layer in [32]:`** contradicts `COLLECTION_LAYERS`. Restore the loop.

All three in [TODO.md](../../TODO.md).

## Status

Stable (Oct 27 run). Safe to rerun after fixing the hardcode. See [LEDGER.md](../../LEDGER.md#activations_collection-infra).
