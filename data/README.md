# `data/`

Small committed datasets. Anything big (model weights, parquet caches, treebanks) lives elsewhere — see `src/lang_probing_src/config.py` paths.

## `grammatical_pairs.json`

Minimal pairs for counterfactual attribution (`experiments/counterfactual_attribution/`).

Currently 30 English pairs spanning 7 concepts: Aspect, Gender, Mood, Number, Person, Polarity, Tense.

### Schema

```jsonc
{
  "id": "number_01",                 // short unique id, used for filenames
  "prefix": "The cats",               // tokens the model conditions on
  "original_token": " sit",           // correct next-token (with leading space)
  "counterfactual_token": " sits",    // alternative next-token with different concept value
  "concept": "Number",                // one of Aspect, Gender, Mood, Number, Person, Polarity, Tense
  "concept_value_orig": "Plur",
  "concept_value_cf": "Sing",
  "note": "Subject-verb number agreement: plural subject expects plural verb"
}
```

Constraints:
- `original_token` and `counterfactual_token` must each be a **single Llama-3.1 token** after tokenization. `counterfactual_attribution/run.py` skips multi-token pairs and lists them in `outputs/counterfactual_attribution/skipped_pairs.json`.
- Pairs must differ on exactly one grammatical dimension; lexical similarity (e.g. `sit` ↔ `sits`) keeps the probe gradient interpretable.

### Multilingual extension (planned)

To add a new language, create a sibling file `grammatical_pairs_{lang}.json` (e.g. `grammatical_pairs_spanish.json`) with the same schema. The counterfactual runner supports a `--data_file` argument and can be invoked per-language:

```bash
python experiments/counterfactual_attribution/run.py \
    --data_file data/grammatical_pairs_spanish.json \
    --output_dir outputs/counterfactual_attribution/spanish
```

Per-concept counts should be >= 5 to make cross-concept comparisons stable (the current English set has Polarity=2, Aspect/Mood=3 — see [TODO.md](../TODO.md)).

### Suggested language seed set

For parity with existing FLORES / Multi-BLiMP work, prioritize:

- English (done)
- Spanish
- French
- German
- Turkish (agglutinative; interesting case/morphology)
- Chinese (no morphological agreement; tests whether features are robust without explicit grammar)
