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

## Wave 4 follow-up — hardcoded path rewrites

After the `outputs/` and `img/` dir renames in Wave 4, many scripts still reference the OLD paths. Scripts will write to new paths? No — the paths are hardcoded in the scripts themselves, so the renames will break reads and cause old-path writes until these are fixed.

Old → new map:
- `outputs/sentence_input_features` → `outputs/input_features`
- `outputs/attributions` → `outputs/output_features`
- `outputs/ablation_results_3_23` → `outputs/ablation`
- `outputs/perplexity_bleu` → `outputs/perplexity_bleu_linear/bleu_and_ppl`
- `outputs/perplexity_comparison` → `outputs/perplexity_bleu_linear/per`
- `outputs/token_analysis_html` → `outputs/token_analysis/html`
- `img/ablate_v3_prob_bar` → `img/ablation`
- `img/input_output` → `img/input_output_overlap`
- `img/perplexity_bleu` → `img/perplexity_bleu_linear`
- `img/probe_performance` → `img/probes`
- `img/sentence_input_features` → `img/input_features`

Also: `config.py` constants (e.g., `PROBES_DIR`) should follow suit — paths built from `config.OUTPUTS_DIR` automatically pick up the tree, so only the leaf names in hardcoded strings need updating.

Action: grep + sed across `experiments/`, `src/lang_probing_src/config.py`, and any remaining module that hardcodes an output path.

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
- [ ] **Counterfactual attribution English per-concept sample sizes are tiny.** Polarity=2, Aspect/Mood=3, Number/Tense=6. The top-50 feature rankings built from 2–6 pairs are fragile. Multilingual cells (fra/spa/tur/ara from Multi-BLiMP, overnight 2026-04-22) are now well-populated; English still needs expansion, especially Polarity / Aspect / Mood.
- [ ] **Linear model R² is low.** Llama=0.02, Aya=0.10. Prediction MAE ≈ 5–6 BLEU points across a ~1.7–27 BLEU range; predictions cluster near intercept. Perplexity coefficients are sensibly negative. Either H1 (monolingual-competence-only) doesn't hold well for Llama, or there's outlier-language leverage (paper note flags Turkish/Hebrew). Plot residuals, refit without those two.
- [ ] **Probe accuracies cluster suspiciously high** (mean 96.68%, max 99.98%, 175 probes; train-test gap <0.15 for all). Probes are learning real signal, but word-level morphological grammar may be near-trivially predictable from surface form alone. Add a shuffled-label baseline to quantify task difficulty before drawing "probe detects grammatical concept" conclusions.
- [ ] **Universal SAE features (f9539, f14366, f12731).** Feature 9539 ranks top-50 in every cell of every language in the overnight multilingual counterfactual-attribution run (10/11 eng, 6/6 fra, 6/6 spa, 4/4 tur, 8/8 ara). Similar behavior from f14366, f12731. Hypothesis: "general grammatical-prediction" features firing on every morphological-choice site, not specific-concept carriers. Checks: (a) max-activating contexts across FLORES — do these fire on every verb-prediction position? (b) ablation specificity — does ablating f9539 hurt model logprob on any next-token site or only grammatical-decision sites? (c) decoder logit-lens — what does `W_dec[9539] @ W_unembed.T` promote?
- [ ] **Turkish Number=Sing ablation: +1.052 Δorig (opposite direction).** Top-20 attribution features, when ablated, INCREASE logP(orig). Hypotheses: (1) sign-convention bug — top features are actually cf-promoting, ablating them helps orig; (2) Turkish-specific — agglutinative morphology interacts with the SAE in unanticipated ways; (3) top-by-|grad×act| are features that SUPPRESS orig. Load top-5 features for this cell, inspect signed_mean_grad values.
- [ ] **ara/Person/3 ablation also positive (+0.556).** Same investigation as Turkish.
- [ ] **fra/Person/{1,2} ablation ratio vs random ≈ 10⁶.** Random baseline was exactly 0.000. Possibilities: (1) genuine — 20 random features happened to have zero effect at cf_pos (plausible given sparse SAE); (2) random-feature pool isn't sampled correctly. Inspect per-pair deltas in `ablation/fra/Person_1/holdout_ablation.json`.
- [ ] **Arabic-dual → English null result.** Top ara/Dual/Dual attribution features have Cohen's d = −0.075 on English "two/both/pair" sentences (vs. null 0.003). Possible causes: (a) Arabic-dual features are dual-specific, no English transfer; (b) English bin A (~55 sentences) too small; (c) top ara/Dual/Dual features are dominated by universal features (f9539 etc.) with uniformly high English activation. Retry with narrower target-feature filter (drop features also top-50 in other Arabic cells).
- [ ] **Null prefixes in Multi-BLiMP Arabic.** ~350 rows had `prefix=None`, silently filtered during overnight pair build. Why? Data-quality question; resolve before trusting ara cell counts as representative.

## Scientific — missing code

- [x] **Rank-1 SVD approximation of the BLEU-vs-perplexity linear fit.** Reproduced in `experiments/perplexity_bleu_linear/rank1_approximation.py`. Llama rank-1 faithfulness = **88.31%** (matches the paper's 88% claim exactly). Aya = 82.37%. Figures at `img/perplexity_bleu_linear/linear_effects_ranks_{model}.png`, `linear_effects_{model}.png`.

## Nice-to-have

- [ ] **Unify word-level and sentence-level input features** under one `experiments/input_features/run.py` with `level: {sentence, word}` config flag. Word-level procedure (with priority-based negative sampling from `reference_paper.tex`) is not currently implemented.
- [x] ~~**Multilingual counterfactual attribution.**~~ Ran overnight 2026-04-22 — eng/fra/spa/tur/ara via Multi-BLiMP. See LEDGER::counterfactual_attribution v2. Tense coverage still missing from Multi-BLiMP; template supplement deferred.
- [ ] **Tense supplement for counterfactual attribution.** Multi-BLiMP has no Tense cells. Build template-generated Tense pairs to close the gap left by the overnight session.
- [ ] **Relocate overnight counterfactual-attribution scripts.** `scripts/attribute_multilingual.py`, `scripts/build_multiblimp_pairs.py`, `scripts/write_report.py` (from the overnight worktree) still live under flat `scripts/` / `run/`. Move to `experiments/counterfactual_attribution/` to match the Wave 3–4 layout.
- [ ] **Decoder logit-lens for top SAE features.** `W_dec[f] @ W_unembed.T` per feature; cache as `[4096, vocab]` matmul; per-feature top-20 promoted tokens. Dashboard context; weak at mid-stream layers but useful.
- [ ] **Max-activating tokens across FLORES.** Per top feature, top-1% activating token contexts across all 5 languages' FLORES devtest. Needed to interpret f9539 / f14366 / f12731.
- [ ] **UD POS/Feats profile per feature.** For each top feature, tabulate UPOS + morphological-feature values of its top-1% activating tokens across UD-PADT (ar), UD-GSD (fr/es), UD-BOUN (tr). Distinguishes generic-verbal from past-tense-morpheme features.
- [ ] **Cross-lingual co-activation graph.** Union of co-activation clusters across all 5 langs (Pearson over 1000 FLORES sentences → NetworkX).
- [ ] **Cross-lingual cross-concept.** Features that are multi-concept in more than one language (pending analysis).
- [ ] **Multi-BLiMP coverage expansion.** Turkish Gender, French past-participle gender agreement with preceding direct object — not in Multi-BLiMP.
- [ ] **Syed-style two-pass attribution patching** as a comparison to current single-pass grad×act (zero-baseline indirect effect). arxiv.org/abs/2602.16080.
- [ ] **Aruna's attention-head identity ablation** on MT prompts (no contrastive pair required; rank heads by logprob drop on r₀ after setting h_i to identity). Useful when contrastives are unavailable.
- [ ] **Multi-token counterfactual with summed-logprob metric.** Overnight session used LAST BPE only. Extension: sum logprob over full multi-token counterfactual span, backprop the sum.
- [ ] **Gender → sexist English (example #1 from overnight plan).** Ablate top Romance-gender features during English generation on crafted prompts ("The doctor said ___"); measure distributional shift on gendered pronouns / stereotype tokens.
- [ ] **Formality → British spelling (example #2).** English minimal pairs on "color/colour", "realize/realise"; check if ablating top Arabic/German formality features shifts logprob on British variants.
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
- [x] ~~Counterfactual language expansion — target set?~~ Overnight 2026-04-22 settled on eng/fra/spa/tur/ara via Multi-BLiMP. Additional languages still open if desired.
- [ ] Output features: keep only latest of the 53 `attributions/` runs per (src, tgt)? If so, how to disambiguate (timestamp? the last run should be authoritative)?
- [ ] Arabic Multi-BLiMP null prefixes (~350 rows): investigate upstream or document as known filtering step?

---

## Archived concerns (resolved; left here for history)

- **~~Layer 31 vs 32 mismatch.~~** Investigated 2026-04-22. Not a bug — two PyTorch indexing conventions (`hidden_states[32]` == `model.model.layers[31]` == same tensor). `ablate.py` already handles it. See `LEDGER.md` cross-experiment notes.
- **~~`images/` vs `img/` path prefix in .tex.~~** `.tex` lives in Overleaf; no repo-side fix needed.
- **~~`jannik_output_features.py` purpose.~~** Was reference-only. Archiving in Wave 1.
- **~~Heaviside STE fix for SAE gate.~~** Investigated during overnight 2026-04-22 counterfactual-attribution rewrite. Moot: the existing code wraps `encode()` in `torch.no_grad()` and treats `z = f_saved.detach().clone().requires_grad_(True)` as a leaf variable, so gradients only flow through `decode()` onward. The non-differentiable gate never enters the backward pass. No autoencoder-side change needed.
- **~~Counterfactual attribution: fixed-last-token position.~~** Fixed overnight 2026-04-22 in `scripts/attribute_multilingual.py` via per-pair `cf_position_idx`.
- **~~Counterfactual attribution: grad aggregation contaminated by early-token signal.~~** Fixed overnight 2026-04-22: defaults to `grad[cf_pos]`; optional windowed sum behind a flag.
- **~~Counterfactual attribution: no per-value signed aggregation.~~** Added overnight 2026-04-22 (signed / abs / signed_gxa / abs_gxa tensors per (lang, concept, value) cell), enabling the sign-flip analysis.
