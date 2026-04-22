# TODOs and oddities

Sourced from `INVENTORY.md`, weird-things scan (2026-04-22), scientific-anomalies scan (2026-04-22), and the layer-31/32 investigation (2026-04-22). Checkbox items are actionable; the question list at the bottom is for the author.

Cross-reference `LEDGER.md` for where each item fits in the experiment tree.

---

## Blockers

Will produce wrong outputs or crash if the scenario hits.

- [ ] **`torchtyping` missing from `requirements.txt`** (pre-existing). `src/lang_probing_src/features/sparse_activations.py` imports `from torchtyping import TensorType`; `features/attribution.py` depends on it. The current `probes` conda env does not have it installed, so the output-features pipeline cannot import. Fix: `pip install torchtyping` + add to `requirements.txt`.
- [ ] **`SAE_FILENAME` undefined** (pre-existing). `experiments/output_features/run.py:152` references `SAE_FILENAME` but the symbol is not defined anywhere in `src/lang_probing_src/config.py`. The script will NameError at runtime. Either define it in `config.py` or replace the usage with the SAE repo's default filename.

- [ ] **`scripts/collect_activations.py:37-42`** — `except Exception` catches load failures but the following code still tries to use `sentences`, which is only defined in the successful branch. Will NameError on any UD load failure. Fix: `raise` after `logging.error`, or initialize `sentences = []` and `return` / `continue`.
- [ ] **`scripts/attribution_flores.py`** — uses `from src.config` (wrong module; the package is `lang_probing_src`). There's a `sys.path.insert` hack that may or may not work depending on CWD. Replace with `from lang_probing_src.config import …`. Drop the sys.path hack.
- [ ] **`scripts/analyze_tokens.py`** probe reader — expects filenames of the form `probe_layer{layer}_n{n}.joblib`, but `scripts/word_probes.py:201` writes `l{layer}_n{n}.joblib`. Probe filtering in `token_analysis` runs today without ever loading probes (silent no-op). Canonical is `l{layer}_n{n}`; fix the reader.

## Important — silent failures, wrong outputs, off-by-one

- [ ] **`src/lang_probing_src/config.py:71,74`** — `"zho_simpl"` key defined twice; second definition (`"Chinese"`) silently overwrites the first (`"Chinese (Simplified)"`). Line 76 builds `NAME_TO_LANG_CODE` from the post-override dict. Remove one; pick the right one.
- [ ] **`scripts/collect_activations.py:112`** — hardcoded `for layer in [32]: # TODO: remove when done`. Contradicts `COLLECTION_LAYERS`. The 207 parquets on disk indicate earlier runs used `COLLECTION_LAYERS`; this hardcode was presumably temporary. Restore `for layer in COLLECTION_LAYERS:`.
- [ ] **`src/lang_probing_src/ablate.py:12,16`** — duplicate `import torch`. Remove one.
- [ ] **`src/lang_probing_src/ablate.py:122-170`** — dead `old_get_batch_positions_masks()` function; the newer `get_batch_positions_masks()` on line 71 is what's used. Delete it.
- [ ] **`src/lang_probing_src/ablate.py:130-133`** — clamps `probe_layer` to `num_layers-1`. Add a one-line comment explaining *why* (PyTorch `hidden_states` has 33 entries; `model.model.layers` has 32; probe filenames use the former convention, module-list access uses the latter).
- [ ] **`LAYERS` vs `COLLECTION_LAYERS`** in `config.py` — `LAYERS = […, 32]`, `COLLECTION_LAYERS = […, 31]`. Keep both but add a docstring comment clarifying they refer to the same physical layer under two indexing conventions. Consider renaming `LAYERS` → `HIDDEN_STATE_LAYERS` and `COLLECTION_LAYERS` → `MODULE_LAYERS` so it's self-documenting.
- [ ] **16+ hardcoded absolute paths** across `scripts/` should use `config.OUTPUTS_DIR`, `config.IMG_DIR`, etc. Known offenders: `analyze_tokens.py:230,378`, `ablate.py:266` (scripts/), `sentence_input_features.py:27,60`, `sentence_input_features_visualize.py:218,225`, `input_output_features_visualize.py:122,126`, `perplexity_comparison.py:17`. Sweep and replace. Makes the repo portable.
- [ ] **`scripts/collect_activations.py:108`** — `args` referenced inside `main()` as a module-global (defined in `__name__ == "__main__"` block). Works, but is a smell; pass `args` to `main(args)` for legibility.
- [ ] **Ablation config naming.** `scripts/ablate.py` defines `multi_input_random` / `multi_output_random` in `EXP_CONFIGS`, but the on-disk output files are `results_multi_random_src.jsonl` / `results_multi_random_tgt.jsonl`. Canonical going forward: `multi_input_random` / `multi_output_random`. Rename files when reorganizing `outputs/`.
- [ ] **v3 ablation `multi_*` configs not yet rerun with `--use_probe`.** v3 monolingual done Mar 23; multilingual still uses v1 data. Decide whether to rerun (paper-critical if H2 claim rests on multilingual ablation).

## Scientific anomalies — investigate

Numbers that look off, based on spot-checking result files.

- [ ] **Ablation mono-random baseline: 66.7% of `mean_delta` values are exactly 0.0** (`outputs/ablation_results_3_23/results_mono_random.jsonl`). Expected: near-zero mean with nonzero variance. Exact zeros suggest either no samples entered the ablation mask (probe never fired), or a bug where the "random" branch short-circuits. Investigate before trusting v3 bar charts.
- [ ] **Counterfactual attribution per-concept sample sizes are tiny.** Polarity=2, Aspect/Mood=3, Number/Tense=6. The top-50 feature rankings built from 2–6 pairs are fragile. Expand `data/grammatical_pairs.json` substantially before drawing cross-concept conclusions.
- [ ] **Linear model R² is low.** Llama=0.02, Aya=0.10. Prediction MAE ≈ 5–6 BLEU points across a ~1.7–27 BLEU range; predictions cluster near intercept. Perplexity coefficients are sensibly negative. Either H1 (monolingual-competence-only) doesn't hold well for Llama, or there's outlier-language leverage (paper note flags Turkish/Hebrew). Plot residuals, refit without those two.
- [ ] **Probe accuracies cluster suspiciously high** (mean 96.68%, max 99.98%, 175 probes; train-test gap <0.15 for all). Probes are learning real signal, but word-level morphological grammar may be near-trivially predictable from surface form alone. Add a shuffled-label baseline to quantify task difficulty before drawing "probe detects grammatical concept" conclusions.

## Scientific — missing code

- [ ] **Rank-1 SVD approximation of the BLEU-vs-perplexity linear fit.** `.tex` claims "88% faithful for Llama" — no code in this repo produces this. Reproducible here in `experiments/perplexity_bleu_linear/`. Lightweight. See Wave 5.

## Nice-to-have

- [ ] **Unify word-level and sentence-level input features** under one `experiments/input_features/run.py` with `level: {sentence, word}` config flag. Word-level procedure (with priority-based negative sampling from `reference_paper.tex`) is not currently implemented.
- [ ] **Multilingual counterfactual attribution.** Currently English-only (30 pairs). Expand `data/grammatical_pairs.json` to at least 3–4 languages to test the cross-lingual claim.
- [ ] **Reconcile `perplexity_results_*.csv` (tiny pilot, ~600 B) vs `combined_results_*.csv` (28 KB).** Latest is canonical; older files are pilots. Document in LEDGER or archive.
- [ ] **Probe grid-search checkpoints.** 14 `all_probe_results_2025-11-22_*.csv` from the same day's grid search. Only the latest is authoritative. Archive the rest.
- [ ] **Probe terminology.** `reference_paper.tex` Related Work says "mass-mean probes"; our code trains logistic regression. Leave a note on the paper side; code stays LR.
- [ ] **Input/output overlap headline statistic.** 29 subdirs of Jaccard/signal plots exist. Produce a single aggregate (e.g., mean Jaccard@k across languages/concepts) for the paper's central H2 claim.
- [ ] **Delete stragglers:** older English `_l30_*` probes created Nov 22 13:45-13:55 (before the `_l32` convention). No cached layer-30 activations exist. Safe to remove after author confirms.
- [ ] **Old `outputs/features/`** (Oct 14, stale) — archived. Consider final disposition.
- [ ] **`outputs/probes/processed_sentences/`** is a growing cache of pyconll-parsed sentences. Not gitignored (but the whole `outputs/` tree is). Add a README note that it's auto-generated.

## Unresolved author questions

Answer these and the corresponding TODOs become actionable:

- [ ] Layer-30 probes for English — safe to delete, or do you want them kept?
- [ ] v3 multilingual ablation — run it?
- [ ] Counterfactual language expansion — target set? (English + Spanish + ? + ?)
- [ ] Output features: keep only latest of the 53 `attributions/` runs per (src, tgt)? If so, how to disambiguate (timestamp? the last run should be authoritative)?

---

## Archived concerns (resolved; left here for history)

- **~~Layer 31 vs 32 mismatch.~~** Investigated 2026-04-22. Not a bug — two PyTorch indexing conventions (`hidden_states[32]` == `model.model.layers[31]` == same tensor). `ablate.py` already handles it. See `LEDGER.md` cross-experiment notes.
- **~~`images/` vs `img/` path prefix in .tex.~~** `.tex` lives in Overleaf; no repo-side fix needed.
- **~~`jannik_output_features.py` purpose.~~** Was reference-only. Archiving in Wave 1.
