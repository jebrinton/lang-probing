# Project ledger

**Project:** `lang-probing` — mechanistic investigation of how multilingual LLMs translate. We probe for grammatical concepts in Llama-3.1-8B (and compare with Aya-23-8B), look for shared multilingual features in the layer-16 SAE (`jbrinkma/sae-llama-3-8b-layer16`, 32768 features), and causally test their role with ablations and counterfactual attribution.

**Thesis (one sentence):** LLMs translate largely by applying the same monolingual computations they already use for language modeling, reading and writing through a fuzzy "semantic hub" of multilingual grammatical features — rather than via language-pair-specific translation modules.

## Hypotheses

- **H1** — BLEU is predictable from monolingual source and target competence, with no cross-lingual interaction term needed. (`BLEU ≈ a + β₁P_src + β₂P_tgt`.)
- **H2** — The "noisy channel" is multilingual. Features that read in grammatical concepts (input features) overlap heavily with features that write them out (output features), across languages.
- **H3** — Adding a language to an LLM is well-approximated by improving its monolingual capability. (Monolingual fine-tuning ≈ parallel-corpus fine-tuning for translation.)
- **H4** — Translation uses the same monolingual circuits the model already uses for language modeling. Task-generality, not a separate translation module.

## How to read this ledger

- **This file is curated.** Each experiment has a stable section. Updated when something changes methodologically or when we learn something.
- **`LAB_NOTEBOOK.md`** is chronological, append-only. Day-by-day log. Cross-references experiments here by name (e.g., "see LEDGER::ablation").
- **`TODO.md`** collects open items — blockers, scientific anomalies, fix-its, ideas.
- **Figures** are linked with markdown relative paths. No fig IDs yet — those come after the paper outline.

---

## Experiments

Ordered by current activity (most active first). Each folder under `experiments/` has a README mirroring the structure below.

### `ablation`

- **Hypotheses:** H2, H4
- **Status:** active (v3 probe-targeted, mono only; multi pending)
- **Question:** Does zeroing out the top-K SAE features for a grammatical concept actually reduce the probability of the correct output? Is the reduction comparable in monolingual and translation contexts? (If yes → causal evidence that these features are *used* during translation, not just correlates.)
- **Method:** Zero-ablate top-K SAE features at source-token or target-token positions, optionally restricted to probe-positive positions. Measure Δp(reference) = (exp(Δ log p) − 1). Seven experimental configs: `mono_input`, `mono_output`, `mono_random`, `multi_input`, `multi_output`, `multi_input_random`, `multi_output_random`.
- **Version history:**
  - v1 (2025-12-22) — heatmap visualization, no probe filter.
  - v2 (2026-01-12) — bar-chart redesign on the same data.
  - v3 (2026-03-23) — `--use_probe --probe_layer 32 --probe_n 1024`; ablation restricted to probe-positive token positions. Only `mono_*` configs rerun.
- **Findings (so far):** Input and output ablations reduce Δp by ~100–300× the random baseline, but absolute magnitudes are small (~−3×10⁻⁴). Feature effects are weak but real. Random baseline has suspicious exact zeros in 66.7% of mono-random rows — **see TODO**.
- **Caveats / uncertainties:** `multi_*` configs still use v1 data. Effect sizes may just be small; or the Δp normalization may be compressing signal.
- **TODOs:**
  - Rerun `multi_*` with probe filter.
  - Investigate the 66.7% exact-zero random baseline.
  - Unify config naming: `multi_input_random` / `multi_output_random` are canonical (not `*_src` / `*_tgt`).
- **Figures:** (current, v3) [img/ablation/barplot_source_*.png](img/ablation/), [barplot_target_*.png](img/ablation/). (v1/v2 archived under `_archive/img/`.)
- **Run:** `python experiments/ablation/run.py --config experiments/ablation/configs/mono_input.yaml` (post-restructure)

### `counterfactual_attribution`

- **Hypotheses:** H2
- **Status:** prototype (English-only); multilingual extension in progress
- **Question:** For each grammatical minimal pair (e.g., "The cat sat on the mat" vs. "… mats"), which SAE features mediate the model's preference? Are they the same features we find with diff-in-means + probe attribution?
- **Method:** For each pair, forward through model + SAE at layer 16, compute `log p(preferred) − log p(alternative)`, backpropagate to SAE feature activations. Rank features by `grad` and `grad × activation` (indirect effect).
- **Version history:**
  - v1 (2026-04-13) — core pipeline. 30 English pairs; 29 processed; 1 skipped (multi-token counterfactual).
  - v1.1 (2026-04-17) — added visualization (bar, jaccard, scatter).
- **Findings (so far):** Feature rankings are reproducible per-pair. But per-concept aggregation is underpowered: Polarity has only 2 pairs, Aspect/Mood have 3 each. Meaningful cross-concept comparison needs more data.
- **Caveats / uncertainties:** English-only. Small per-concept sample sizes (top-50 rankings from 2–6 pairs are fragile). Multilingual extension means expanding `data/grammatical_pairs.json`.
- **TODOs:**
  - Expand `grammatical_pairs.json` to non-English (H2 point — the claim is cross-lingual).
  - Add more pairs per concept, especially Polarity / Aspect / Mood.
  - Cross-reference top features against `input_features` / `output_features` rankings.
- **Figures:** [outputs/counterfactual_attribution/plots/](outputs/counterfactual_attribution/plots/).
- **Run:** `python experiments/counterfactual_attribution/run.py` (post-restructure)

### `token_analysis`

- **Hypotheses:** H2
- **Status:** active (Apr 17)
- **Question:** At which token positions do specific SAE features activate, and what happens to per-token logprobs when we ablate them? Qualitative / debug view.
- **Method:** YAML-configured per-token analysis. For each example sentence or translation pair: extract SAE activations per token, ablate a chosen feature set, collect per-token Δ log p, render as HTML with three panels (activation heatmap, logprob-delta heatmap, table).
- **Version history:**
  - v1 (2026-03-30) — core pipeline.
  - v1.1 (2026-04-17) — added multilang + mood configs.
- **Findings (so far):** Qualitative tool. No cross-experiment synthesis yet.
- **Caveats / uncertainties:** **Probe filename mismatch silently disables probe filtering** (reader expects `probe_layer{layer}_n{n}.joblib`, writer produces `l{layer}_n{n}.joblib`). See TODO.
- **TODOs:**
  - Fix probe filename format consumer.
  - Decide whether this experiment stands alone or gets folded into `ablation` as a debug view.
- **Figures:** [outputs/token_analysis_html/](outputs/token_analysis_html/) (HTML, not PNG)
- **Run:** `python experiments/token_analysis/run.py --config experiments/token_analysis/configs/mood.yaml` (post-restructure)

### `perplexity_bleu_linear`

- **Hypotheses:** H1
- **Status:** active (Mar 2 fit)
- **Question:** How much BLEU variance across language pairs is explained by a linear combination of source and target perplexity? (Low R² = cross-lingual interaction matters. High R² = monolingual competence is enough.)
- **Method:** Compute corpus PPL per language on FLORES devtest; compute Perplexity Error Rate (PER) per language on Multi-BLiMP minimal pairs; join with external BLEU scores; fit OLS with combinations of raw/log transforms and optional interaction term. Visualize with source/target competence scatter and joint-competence contour.
- **Version history:**
  - v1 (2026-02-09) — scatter + mixed-lm.
  - v2 (2026-02-20) — Pearson/Spearman + PER-based figures.
  - v3 (2026-03-02) — OLS fits with interaction toggle.
- **Findings (so far):** R² = 0.02 (Llama) to 0.10 (Aya). Coefficients are consistently negative (higher perplexity → lower BLEU, correct direction). **But the fit is genuinely weak** — prediction MAE is ~5–6 BLEU points, predictions cluster near intercept. The monolingual-competence-only linear-in-PER model is a surprisingly bad BLEU predictor for Llama. **However:** a rank-1 SVD decomposition of the BLEU(src, tgt) matrix is 88.31% faithful for Llama (matches the paper's 88% claim), and 82.37% for Aya. This says the BLEU surface is well-approximated by a single src×tgt outer product — i.e. BLEU decomposes multiplicatively by src and tgt competence even when a simple linear-in-PER model fails. The tension between R² ≈ 0.02 and rank-1 ≈ 88% is evidence that PER is a noisy proxy for the "true" competence vectors embedded in the rank-1 factors.
- **Caveats / uncertainties:** Does H1 really hold if R² is this low? Or is this a data issue (MAE-vs-BLEU-range, noisy BLEU, outlier languages)?
- **TODOs:**
  - ~~Add rank-1 SVD approximation~~ ✓ Done (`rank1_approximation.py`, 2026-04-22).
  - Refit without Turkish / Hebrew to see if specific languages dominate residuals.
  - Plot residuals.
  - Reconcile `perplexity_results_*.csv` (tiny pilot) vs `combined_results_*.csv` (likely canonical) as source of truth.
  - Investigate why PER-based linear model is low-R² but rank-1 matrix decomposition is ~88% — is PER a bad proxy for the competence latents?
- **Figures:** [img/perplexity_bleu/](img/perplexity_bleu/) — source_competence, target_competence, joint_competence_{scatter,contour}, perplexity_plot, perplexity_vs_bleu_sorted.
- **Run:** `python experiments/perplexity_bleu_linear/run.py --config experiments/perplexity_bleu_linear/configs/llama.yaml` (post-restructure)

### `input_features`

- **Hypotheses:** H2
- **Status:** active (Dec 1 sentence-level; word-level variant pending)
- **Question:** Which SAE features fire differentially for positive vs. negative examples of a grammatical concept in a given language? Are those feature sets shared across languages?
- **Method:** **Sentence level (current):** for each (language, concept, value), load UD sentences tagged positive/negative, extract SAE activations, mean-pool over tokens per sentence, compute mean(pos) − mean(neg) over sentences. Produces a 32768-dim diff vector per (language, concept, value). **Word level (spec'd in paper, not yet implemented):** same but over token-aligned words with priority-based negative sampling.
- **Version history:**
  - v1 (exploratory, `zzz_input_space.py`) — archived.
  - v2 (2025-12-01) — `sentence_input_features.py`, current canonical sentence-level.
- **Findings (so far):** Diff vectors are sensible (non-zero, language-dependent). Jaccard overlap across languages is non-trivial at top-k; distribution of multilingual-ness per feature is visualized.
- **Caveats / uncertainties:** Word-level procedure is specified in `reference_paper.tex` §"Word-level procedure" with priority-based negative sampling. Not yet implemented. Until then, sentence-level is the only empirical claim.
- **TODOs:**
  - Unify sentence + word under one `run.py` with `level: {sentence, word}` config. Word-level → `experiments/input_features/configs/word.yaml`.
  - Implement priority-based negative sampling as specified.
- **Figures:** [img/sentence_input_features/](img/sentence_input_features/) — feature_language_distribution, jaccard_similarity, per-language magnitudes.
- **Run:** `python experiments/input_features/run.py --config experiments/input_features/configs/sentence.yaml` (post-restructure)

### `output_features`

- **Hypotheses:** H2
- **Status:** active (Dec 6–15 runs; one canonical per src-tgt pair retained)
- **Question:** Which SAE features does a late-layer grammatical-concept probe's prediction *depend on* during translation? (Gradient attribution from probe logit → SAE features.)
- **Method:** Train a concept probe at a late layer (32) on UD data. At layer 16 during translation (FLORES sentence pairs), run forward, decode the SAE, attribute gradient of probe logit w.r.t. SAE features via nnsight. Accumulate per-token, save as effect tensors per (src, tgt).
- **Version history:**
  - v1 (2025-12, multiple reruns in 53 timestamped dirs).
- **Findings (so far):** Effect tensors are sparse (~0.7% nonzero) with magnitudes ~10⁻⁴, as expected given the SAE's sparsity. Cross-language comparability is the purpose of `input_output_overlap`.
- **Caveats / uncertainties:** `attribution_flores.py` is canonical. `jannik_output_features.py` was a reference implementation — archived. Import bug (`from src.config` → should be `lang_probing_src.config`) is a **TODO**.
- **TODOs:**
  - Fix `from src.config` import.
  - Decide which of the 53 timestamped `outputs/attributions/` runs to retain per (src, tgt) pair (likely latest; archive the rest).
- **Figures:** Indirect — consumed by `input_output_overlap`.
- **Run:** `python experiments/output_features/run.py --config experiments/output_features/configs/default.yaml` (post-restructure)

### `input_output_overlap`

- **Hypotheses:** H2 (the central claim)
- **Status:** active (Dec 15)
- **Question:** Do the top SAE features for a concept, measured monolingually (input), overlap with the top features that drive a concept probe during translation (output)? Across how many languages?
- **Method:** Pure consumer of `input_features` and `output_features`. Computes Jaccard of top-k sets, "signal plots" (top-k magnitude ranked against complementary-set rank), per-language distributions.
- **Version history:**
  - v1 (2025-12-15).
- **Findings (so far):** 29 (language, concept, value) combos have overlap plots. Haven't been systematically reviewed — central H2 claim is plotted but not quantified into a single headline statistic yet.
- **Caveats / uncertainties:** "Signal plot" code is partially commented out in main. Needs a clean pass.
- **TODOs:**
  - Produce a single aggregate statistic (e.g., mean Jaccard@k across languages/concepts).
  - Uncomment and verify signal plots.
- **Figures:** [img/input_output/](img/input_output/) — 29 subdirs with `jaccard_topk.png`, `signal_input.png`, `signal_output.png`.
- **Run:** `python experiments/input_output_overlap/visualize.py --config experiments/input_output_overlap/configs/default.yaml` (post-restructure)

### `probes` (INFRA)

- **Hypotheses:** H2 (infrastructure)
- **Status:** active (Nov 22 canonical run; 300 probes on disk)
- **Question:** Can we train reliable linear probes for (language, concept, value, layer) grammatical detection?
- **Method:** cuML GPU logistic regression, word-level tokens (aligned via `word_ids()`, MWT-aware), 4-fold CV grid search over C ∈ logspace(−4, 3, 16), L2, balanced class weights, QN solver.
- **Version history:**
  - v1 (legacy sentence-level, `train_probes.py`) — archived.
  - v2 (2025-11-22) — `word_probes.py`, cuML, canonical.
- **Findings (so far):** Probe accuracies cluster high (mean 96.68%, max 99.98%, min 0.79). Small train-test gap (max 0.15). Probes are learning real signal, but the task is easy — word-level grammatical morphology is highly predictable lexically. Whether this reflects "deep grammatical" encoding is an open interpretive question.
- **Caveats / uncertainties:** `.tex` mentions "mass-mean probes"; our code trains logistic regression. Terminology mismatch, code is canonical. 14 CSV checkpoints from the Nov 22 grid search — the latest is authoritative.
- **TODOs:**
  - Unify probe filename format on `l{layer}_n{n}.joblib` (fix consumers that expect `probe_layer{layer}_n{n}`).
  - Archive intermediate grid-search CSVs; keep only the final `all_probe_results.csv`.
  - Clarify "mass-mean" wording in paper (LEDGER note: paper side).
- **Figures:** [img/probe_performance/](img/probe_performance/) — accuracy_vs_layer per concept, test_accuracy_distribution, all_concepts_c_value_vs_layer.
- **Run:** `python experiments/probes/run.py --config experiments/probes/configs/word_cuml.yaml` (post-restructure)

### `activations_collection` (INFRA)

- **Hypotheses:** INFRA
- **Status:** stable (Oct 27 run; 23 languages × 9 layers on disk)
- **Question:** n/a — cache residual-stream activations per language × layer for downstream consumption.
- **Method:** Load UD treebanks, trace Llama-3.1-8B via nnsight, extract mean-pooled residual-stream activations, partition and save as `outputs/activations/language={L}/layer={N}/data.parquet`.
- **Version history:**
  - v1 (2025-10-27) — 207 parquets produced.
- **Findings (so far):** n/a. Infrastructure.
- **Caveats / uncertainties:** Line 112 of `collect_activations.py` hardcodes `for layer in [32]:` despite `COLLECTION_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]`. The 207 parquets on disk suggest all 9 layers were collected at some point — the hardcode was presumably temporary. **See TODO.**
- **TODOs:**
  - Fix line 112 hardcode (should use `COLLECTION_LAYERS`).
  - Fix silent exception on line 37–42 (`sentences` undefined if load fails).
- **Run:** `python experiments/activations_collection/run.py --languages English Spanish …` (post-restructure)

### `monolingual_ft` (H3) — SCAFFOLD ONLY

- **Hypotheses:** H3
- **Status:** scaffold — no code in this repo. The finetuning lives in Jannik's separate codebase.
- **Question:** Does monolingual fine-tuning improve translation performance comparably to parallel-corpus fine-tuning?
- **Method:** (Planned.) Given a finetuned checkpoint (monolingual or parallel), run FLORES BLEU, Multi-BLiMP PER, and representation-similarity evaluation against a baseline. This repo hosts the *evaluation* side, not training.
- **TODOs:**
  - When a checkpoint arrives: fill out `experiments/monolingual_ft/evaluate.py` (currently stub).
- **Run:** (once scaffolded) `python experiments/monolingual_ft/evaluate.py --checkpoint <path> --baseline <path>`

---

## Archived experiments

See [_archive/experiments/README.md](_archive/experiments/README.md) for legacy work — steering vectors, PCA, cosine similarity heatmaps (fall 2025). Archived because the project has transitioned to SAE-latent features as the primary H2 evidence; SV work is in residual space and serves as background, not current claims.

## Cross-experiment notes

- **Layer 31 vs 32 is a naming artifact, NOT a bug** (investigated 2026-04-22). Llama-3.1-8B has 32 transformer layers. PyTorch indexes `outputs.hidden_states` as a 33-entry tuple (embedding + 32 layer outputs), so `hidden_states[32]` is the final layer's output. `model.model.layers` is a 32-entry list indexed 0..31, so `model.model.layers[31]` is the same layer. Probes use the hidden_states convention (`_l32`); cached activations use the module-list convention (`layer=31`). They refer to the same tensor. `src/lang_probing_src/ablate.py:130-133` explicitly clamps `probe_layer=32` to index 31. Rename or add a comment for future-you; no semantic change needed.
- **Probe filename format:** canonical is `l{layer}_n{n}.joblib`. Any consumer expecting `probe_layer{layer}_n{n}` must be fixed.
- **Ablation config naming:** canonical is `multi_input_random` / `multi_output_random` (not `*_src` / `*_tgt`).
- **Output dir naming:** each experiment writes to `outputs/<experiment_name>/`; only the latest version is in-tree. Historical versions under `_archive/outputs/`.
