# INVENTORY.md

Reconnaissance of `/projectnb/mcnet/jbrin/lang-probing` as of 2026-04-20.
Pure inventory — no restructuring proposals, no deletion recommendations.

Anchor claims (for tie-in labels below):
- **H1** — BLEU predictable from monolingual src/tgt competence, no cross-lingual interaction needed.
- **H2** — Grammatical concepts share a language/task-agnostic "semantic hub"; input and output feature spaces overlap.
- **H3** — Adding translation for a new language = improving monolingual competence (monolingual FT ≈ parallel-corpus FT).
- **LEGACY** — From earlier project phases (summer/fall 2025): typological-similarity analyses, SV/PCA cross-lingual representation, Aya-vs-Llama comparisons.
- **INFRA** — Infrastructure (collect activations, train probes) supporting one or more Hs.

---

## 1. Script inventory

### 1.1 `scripts/` (Python, user-invoked)

#### Ablation family

**`scripts/ablate.py`** (21 KB, Mar 23) — **production ablation pipeline (v3).** For each (source_lang, target_lang, concept, value, k): builds monolingual or translation prompts (2-shot FLORES examples for multi), tokenizes with offset_mapping to build per-token source/target masks, optionally applies a late-layer word probe to restrict ablation to concept-positive token positions, loads input / output / random feature vectors (top-K SAE features), calls `ablate_batch()` to zero those features at the ablation positions via nnsight tracer, and writes **Δp(reference) = (exp(Δlogp)−1)** per example. Supports 7 experiment configs: `mono_input`, `mono_output`, `mono_random`, `multi_input`, `multi_output`, `multi_input_random`, `multi_output_random`. Inputs: input-feature `diff_vector.pt` (hardcoded `outputs/sentence_input_features/{Lang}/{Concept}/{Value}/diff_vector.pt`), output effects files (via `lang_probing_src.utils_input_output.load_effects_files`), word probes (`outputs/probes/word_probes/{Lang}_{Concept}_{Value}_l{layer}_n{n}.joblib`), FLORES (HF `gsarti/flores_101`), Llama-3.1-8B + SAE layer 16. CLI: `--experiment --concept --value --k --max_samples --batch_size --use_probe --probe_layer --probe_n --output_dir`. Output: `outputs/ablation_results/results_{experiment}.jsonl` (one line per (src, tgt, concept, value, k)). Ties to **H2 + H3** (same features, ablate in both mono and translation contexts). Conf: **high**.

**`scripts/run_ablation.py`** (9 KB, Nov 16) — **older, broken.** Imports `lang_probing_src.ablation` (module does not exist; only `ablate.py` and `zzz_ablation.py` are present). Would apply three modes (`simple` / `progressive` / `necessity`) to small example files (`examples/*.txt`). Not invoked by any other script. Appears dead-code-superseded by `ablate.py`. Conf: **medium**.

**`scripts/visualize_ablate.py`** (2 KB, Dec 22) — **v1 heatmap visualizer.** Loads one JSONL, pivots (src_lang × tgt_lang) per (experiment, concept, value, k), plots coolwarm heatmap. Output: `heatmap_{exp}_{concept}_{value}_{k}.png` to `--output_dir` (used for `img/ablate_prob/`). Conf: **high**.

**`scripts/visualize_ablate_bar.py`** (5 KB, Apr 13) — **v2/v3 barplot visualizer.** Loads all `results_*.jsonl` from `--input_dir`, infers experiment from filename, optionally filters by `--num_samples` / `--probe_layer`, produces `barplot_source_{concept}_{value}_{k}.png` and `barplot_target_*.png` per group. Used for `img/ablate_prob_bar/`, `img/ablate_v2_prob_bar/`, `img/ablate_v3_prob_bar/`. Conf: **high**.

#### Input / output feature family

**`scripts/input_features.py`** (5 KB, Dec 1) — word-token-level experimental version: iterates sentences, masks to last-token-of-word via `word_ids()`, applies probe + SAE; does not save, prints/returns arrays. Superseded. Conf: **medium**.

**`scripts/sentence_input_features.py`** (4.7 KB, Dec 1) — **production sentence-level input features.** For each (language, concept, value), loads `ConlluDatasetPooled` (treebank="PUD"), mean-pools SAE activations per sentence, computes diff-in-means (positive vs negative concept examples). Output: `outputs/sentence_input_features/{Lang}/{Concept}/{Value}/diff_vector.pt` (torch tensor, shape `[32768]`). Implements .tex Q1 "Sentence-wise procedure". Conf: **high**.

**`scripts/sentence_input_features_visualize.py`** (9 KB, Dec 9) — visualizes diff vectors: top-k by magnitude, Jaccard similarity across languages, per-language magnitude bars, cross-language feature distribution histograms. Outputs to `img/sentence_input_features/` (`feature_language_distribution_{Concept}_{Value}_top{K}.png`, `jaccard_similarity_{Concept}_{Value}_top{K}.png`, `{Lang}_{Concept}_{Value}_magnitudes.png`). Conf: **high**.

**`scripts/input_output_features_visualize.py`** (7.5 KB, Dec 15) — **input vs output feature overlap** per (target_lang, concept, value). Loads input features (`outputs/sentence_input_features/…diff_vector.pt`) + output effects (`outputs/attributions/{timestamp}/effects_{src}_{tgt}.pt`), produces 3 plots per combo into `img/input_output/{Lang}_{Concept}_{Value}/`: `jaccard_topk.png`, `signal_input.png`, `signal_output.png`. Signal plots currently commented-out in main. **Core H2 evidence.** Conf: **high**.

**`scripts/jannik_output_features.py`** (9 KB, Dec 9) — purpose unclear from name alone; likely port of Jannik Brinkmann's multilingual-features output-feature analysis (README acknowledges his code was source for sentence-level probing). Writes to `outputs/attributions/` based on provenance sweep. Conf: **low** (not read deeply). See §6 Q1.

**`scripts/find_features.py`** (7.5 KB, Nov 16) — post-hoc analysis: loads trained probes, extracts top-K SAE features by absolute probe weight, computes cross-language shared-feature dict. Output: `outputs/features/{Concept}_{Value}.json`, `outputs/features/{Concept}_{Value}_shared.json`. Precedes the modern diff-in-means input-features pipeline; still functional. Conf: **medium**.

**`scripts/attribution_flores.py`** (7.6 KB, Nov 24) — **output-feature producer via attribution patching.** Loads probe (`PROBES_DIR`), converts sklearn LR → torch LinearLayer (fuses scaler), runs nnsight trace at layer 32 on FLORES sentence pairs, calls `attribution_patching_per_token` from the probe logits to SAE features, accumulates effects. Output: `outputs/attributions/{YYYYMMDD_HHMMSS}/effects_{Src}_{Tgt}.pt` + `config.json`. Uses `from src.config` — fragile relative import (config is in `src/lang_probing_src/`). CLI: `--language --layer --num_probe_samples --max_samples --batch_size --output_dir`. **Core H2 evidence** (gradient-attribution output features). Conf: **high**.

**`scripts/zzz_input_space.py`** (7.3 KB, Nov 17) — early exploratory input-feature script (diff-in-means + top-k); superseded by `sentence_input_features*.py`. `zzz_` prefix = deprecated. Conf: **medium**.

#### Perplexity–BLEU family (H1)

**`scripts/perplexity.py`** (5.6 KB, Feb 9) — computes corpus token-level perplexity on FLORES-101 devtest per language for a given model. CLI: `--model_id --batch_size`. Output: `outputs/perplexity_bleu/perplexity_results_{model}.csv`. Conf: **high**.

**`scripts/perplexity_comparison.py`** (17 KB, Feb 19) — computes **Perplexity Error Rate (PER)**: proportion of minimal-pair rows where model assigns lower perplexity to the WRONG sentence. Supports single-language (`--config`) or multi-language (`--multilang --languages`) runs. Intended for Multi-BLiMP (`jumelet/multiblimp`). Output: `outputs/perplexity_comparison/error_rates_by_language_{model}.json`, `perplexity_matrices_{model}.npz`. Conf: **high**.

**`scripts/linear_model_ppl_bleu.py`** (10 KB, Mar 2) — **fits the BLEU ~ a + β₁P_src + β₂P_tgt [+ β₃P_src·P_tgt] model.** Toggles: `--feature-transform {raw,log}`, `--include-interaction {yes,no}`. Uses numpy `lstsq` (closed-form OLS). Output: `outputs/perplexity_bleu/linear_models/linear_coeffs_{model}_{raw|log}_{joint|nojoint}.csv` + `linear_predictions_*.csv`. Slide history: R² ≈ 0.11 for Aya, 0.02 for Llama (raw joint); .tex claims "88% faithful rank-1 approximation" — **no rank-1 code exists**. Conf: **high** on impl; **low** on .tex claim reproduction.

**`scripts/visualize_perplexity_bleu.py`** (3 KB, Feb 9) — legacy v1 scatter + mixedlm (groups by src). Outputs `img/perplexity_bleu/perplexity_vs_bleu_{model}.png`. Conf: **medium**.

**`scripts/visualize_perplexity_bleu_sorted.py`** (6 KB, Feb 9) — adds a "sorted-by-target-difficulty" plot with twin-axis. Outputs `perplexity_vs_bleu_{model}_sorted.png` and `perplexity_sorted_vs_bleu_{model}_sorted.png`. Conf: **high**.

**`scripts/visualize_perplexity_bleu_correlation.py`** (7.7 KB, Feb 20) — **produces the current H1 figures.** Inbound/outbound BLEU × src/tgt PER with Pearson/Spearman, 2D joint competence scatter + contour. Outputs (in `img/perplexity_bleu/`): `{aya|llama}_source_competence.png`, `_target_competence.png`, `_joint_competence_scatter.png`, `_joint_competence_contour.png`. Conf: **high**.

**`scripts/visualize_perplexity_error_bar.py`** (3 KB, Feb 20) — per-language PER bar chart. Outputs `perplexity_plot_{model}.png` and `_sorted.png`. Conf: **high**.

**`scripts/combine_csvs.py`** (1 KB, Mar 2) — joins `bleu_results_{model}.csv` with `perplexity_results_{model}.csv` twice (on `src`, then `tgt`) → `outputs/perplexity_bleu/combined_results_{model}.csv`. Convenience / intermediate. Conf: **high**.

#### Steering vectors / PCA (LEGACY, but still partly used)

**`scripts/collect_steering_vectors.py`** (10 KB, Nov 16) — **canonical modern SV producer.** Reads pre-cached activations from `outputs/activations/language={L}/layer={N}/data.parquet`, filters by concept-tag presence, computes `mean(pos) − mean(neg)` per (concept, value, language, layer), writes partitioned parquet to `outputs/steering_vectors/all/concept=X/value=Y/language=L/layer=N/`. Also defines `generate_cosine_similarity_heatmap()` though whether it's invoked is unclear. Conf: **high**.

**`scripts/generate_steering_vectors.py`** (17 KB, Nov 16) — older, self-contained SV generator: traces model + extracts + diffs in one pass. Slower. Still functional. Conf: **medium**.

**`scripts/zzz_collect_steering_vectors.py`** (12 KB, Nov 16) — deprecated pickle-based version; superseded by `collect_steering_vectors.py`'s parquet output. Conf: **medium**.

**`scripts/pca_steering_vectors.py`** (12 KB, Nov 16) — loads all SVs via parquet row filters, runs sklearn PCA (n_components=2), scatter plots per (concept, value, [layer]). Output: `img/pca_steering_vectors/concept={C}_value={V}_language={L1-L2-…}_layer={N}.png`. Conf: **high**.

**`scripts/visualize_steering_vectors.py`** (19 KB, Nov 16) — multi-purpose: verifies SV tensors; generates per-concept-value cosine-similarity heatmaps across languages. Run script `run/run_visualization_steering_vectors.sh` loops over layers. **Likely producer of `img/sv_cs_heatmaps/` and `outputs/visualization_steering_vectors/`** — note: grep found no literal `sv_cs_heatmaps` string in scripts, so either the path is assembled via config/IMG_DIR or passed as CLI arg; two of the subagents disagreed on the producer. See §5 below. Conf: **medium**.

**`scripts/steer.py`** (10 KB, Nov 16) — applies a steering vector at a chosen layer during generation (`h += coeff · SV`); used for early qualitative examples from slides ("Ablating plural feature — The ducks … → The duck"). Writes to `outputs/steering/`. Conf: **medium**.

#### Counterfactual attribution (H2 prototype, newest — Apr 2026)

**`scripts/counterfactual_attribution.py`** (17 KB, Apr 13) — for each of ~30 minimal-pair English sentences in `data/grammatical_pairs.json` (`prefix`, `original_token`, `counterfactual_token`, `concept`), gradient-backprops `logP(orig) − logP(cf)` through SAE-decode at layer 16 to obtain per-feature `grad` and `grad × act` (indirect effect) rankings. Output: `outputs/counterfactual_attribution/{aggregated_by_concept.json, per_pair_results.json, skipped_pairs.json}` + optional `raw_gradients/{id}_{act,grad}.pt` (shape `[seq_len, 32768]`). 7 concepts covered (Number, Tense, Gender, Person, Mood, Aspect, Polarity); 1 skipped (`person_03`, counterfactual is 2 tokens). CLI: `--data_file --output_dir --top_k --save_raw_tensors`. Conf: **high**.

**`scripts/analyze_counterfactual_results.py`** (9 KB, Apr 13) — aggregates per-pair → per-concept, prints top-N features, cross-concept Jaccard, optional probe cross-reference. Stdout only. Conf: **high**.

**`scripts/visualize_counterfactual_results.py`** (14 KB, Apr 17) — produces `outputs/counterfactual_attribution/plots/`: per-concept `bar_top20_{concept}_{grad|grad_x_act}.png`, `jaccard_top50_{grad|grad_x_act}.png`, `metric_distribution_by_concept.png`, `feature_scatter_activation_vs_grad[_facet].png`. Conf: **high**.

#### Probing infrastructure (INFRA)

**`scripts/train_probes.py`** (8.7 KB, Nov 16) — **legacy sentence-level probe training** (sklearn LogisticRegression, balanced, liblinear, C=0.1). Saves `outputs/probes/{Lang}_{Concept}_{Value}.joblib`. Not actively used by downstream code; word-level probes are. Conf: **high**.

**`scripts/word_probes.py`** (11 KB, Nov 22) — **canonical probe trainer.** cuML-GPU LogisticRegression with 4-fold CV grid search over `C ∈ logspace(-4, 3, 16)`, L2 penalty, QN solver. Per-word token activations via `word_ids()`, MWT-aware. Caches parsed sentences at `outputs/probes/processed_sentences/{Lang}_{split}_processed.joblib`. Output probe files: `outputs/probes/word_probes/{Lang}_{Concept}_{Value}_l{layer}_n{max_samples}.joblib` (~300 files on disk). Also writes summary CSVs `outputs/probes/all_probe_results_{YYYY-MM-DD_HH:MM}.csv` (14 such CSVs from Nov 22 runs — grid-search iterations). Conf: **high**.

**`scripts/filter_probe_training_log.py`** (369 bytes, Nov 22) — strips non-INFO lines from `run/word_probes_for_in_out.out` → `…_filtered.out`. Trivial utility. Conf: **high**.

**`scripts/visualize_probe_results.py`** (7.5 KB, Nov 10) — reads `all_probe_results_tense.csv`, plots per-concept-value accuracy-vs-layer, test-accuracy distribution, and `all_concepts_c_value_vs_layer.png` in `img/probe_performance/`. Conf: **high**.

#### Activation / SAE collection (INFRA)

**`scripts/collect_activations.py`** (4.7 KB, Nov 16) — mean-pooled residual-stream activations from UD treebanks via nnsight, saved as partitioned parquet `outputs/activations/language={L}/layer={N}/data.parquet` (schema: `sentence_id, language, sentence_text, layer, tags, activation`). 23 languages × 9 layers on disk (207 parquets, ~70 MB each, Oct 27). **Note:** Line 112 hardcodes `for layer in [32]:` (with a "TODO: remove when done" comment) — contradicts the `COLLECTION_LAYERS` config that would iterate `[0,4,8,12,16,20,24,28,31]`. See §5. CLI: `--languages`. Conf: **high** on behavior, **low** on whether last run used the hardcoded layer or config.

**`scripts/collect_sae_activations.py`** (4.7 KB, Nov 16) — sibling script for FLORES inputs. Loads SAE but the shared `collect_sentence_activations()` function never applies it — same raw residual-stream output. Target dir `outputs/activations_flores/` is empty → never successfully run. Conf: **high**.

#### Token analysis (qualitative / debug, Mar–Apr 2026)

**`scripts/analyze_tokens.py`** (17 KB, Mar 30) — YAML-config-driven per-token ablation. For each experiment: builds prompt (free_form or flores), masks source/target positions, optionally probe-filters, loads top-K input/output/random feature vector, runs per-token ablation (nnsight), collects SAE activations + original/intervention logprobs per token, saves JSON. Note: expected probe filename format in code is `probe_layer{layer}_n{n}.joblib`, which **mismatches** actual `…_l{layer}_n{n}.joblib` from `word_probes.py` — probe loading silently disabled. CLI: `--config --output_dir`. Conf: **high**.

**`scripts/visualize_tokens.py`** (10 KB, Mar 30) — pure-Python HTML renderer of analyze_tokens JSON: three panels per experiment (colored SAE-activation tokens, colored logprob-delta tokens, per-token table). No JS/CSS deps. Output: `outputs/token_analysis_html/{name}.html`. Conf: **high**.

#### One-off / scratch

**`scripts/nnsight_practice.py`** (345 B, Oct 19) — 16-line nnsight demo (load Llama, trace layer 4). Throwaway. Conf: **high**.

**`scripts/zzz_scratchpad.py`** (1.2 KB, Nov 16) — counts tokens per UD treebank, throwaway. Conf: **high**.

**`scripts/aaa_run.md`** (4 lines, Mar 23) — v3 ablation command reference for `--use_probe --probe_layer 32 --probe_n 1024 --max_samples 256` + `visualize_ablate_bar.py` into `img/ablate_v3_prob_bar/`. Conf: **high**.

### 1.2 `src/lang_probing_src/` (library)

**`__init__.py`** — tiny; exports nothing meaningful.

**`config.py`** (4.1 KB) — centralized paths, model IDs (`MODEL_TO_ID = {"llama": "meta-llama/Llama-3.1-8B", "aya": "CohereForAI/aya-23-8B"}`), language lists (`LANGUAGES`=23, `LANGUAGES_NOVA`=8, `LANGUAGES_DEC`=9), `CONCEPTS_VALUES` dict, `LAYERS`=[0,4,8,12,16,20,24,28,32], `COLLECTION_LAYERS`=[…,31] (note 31 vs 32 discrepancy), SAE=`jbrinkma/sae-llama-3-8b-layer16`, layer 16, dim 32768. Conf: **high**.

**`ablate.py`** (15 KB, Apr 13) — `ablate()`, `ablate_batch()`, `get_probe_ablation_mask()`, `logits_to_probs()`. Nnsight-based SAE feature zero-ablation. Returns tensors / dicts; no file I/O. Conf: **high**.

**`zzz_ablation.py`** (7.5 KB, Oct 14) — deprecated earlier API (`ablate_features`, `activate_features`, `progressive_ablation`, `test_feature_necessity`). Only `run_ablation.py` (also dead) imports it. Conf: **medium**.

**`activations.py`** (18 KB, Dec 15) — 7 exported functions: `extract_mlp_activations`, `extract_sae_activations` (deprecated, TODO-commented: "incorrect, you shouldn't mean pool before encoding"), `extract_single_sentence_sae_activations`, `get_mean_sae_activation`, `extract_all_activations_for_steering`, `extract_mean_activations`, `collect_sentence_activations` (the actively-used one). Conf: **high**.

**`autoencoder.py`** (3.3 KB, Nov 24) — Gated-SAE: `encoder` (4096→32768), gate/mag biases (ReLU + Heaviside gate), `decoder` (32768→4096) + bias. `from_pretrained(path)` load pattern. Conf: **high**.

**`data.py`** (16 KB, Dec 1) — UD loading utilities (`get_all_treebank_files`, `get_training_files`, `get_test_files`, `get_available_concepts`), `ProbingDataset` (sentence-level multi-treebank), `balance_dataset`, `concept_filter`, FLORES loaders. Mixes legacy and current. Conf: **medium**.

**`dataloader_factory.py`** (2.6 KB) — `SentenceDataLoaderFactory` convenience class; lazy dataloader property. Not called by active `collect_activations.py`. Conf: **high**.

**`features.py`** (5.6 KB, Oct 14) — `find_top_correlating_features`, `find_top_positive_negative_features`, `get_shared_features_across_languages`, `analyze_feature_overlap` — utility functions used by `find_features.py`. Conf: **high**.

**`probe.py`** (3 KB, Oct 14) — sklearn probe lifecycle: `train_probe`, `evaluate_probe`, `save_probe`/`load_probe` (joblib), `get_probe_predictions` (incl. logits via `decision_function`), `get_probe_info`. Conf: **high**.

**`sentence_dataset_class.py`** (1.7 KB, Oct 26) — `SentenceDataset` + `collate_fn` with dynamic padding. Preserves UD tags alongside tokenized inputs. Conf: **high**.

**`utils.py`** (7.2 KB, Dec 22) — `setup_model()`, `get_device_info()`, JSON/dir helpers, misc. Conf: **high**.

**`utils_input_output.py`** (2.2 KB, Dec 15) — `load_effects_files()` and helpers for loading output-feature attributions. Used by `ablate.py` (`scripts/`). Conf: **medium** (small but critical glue).

**`word_probing_utils.py`** (7.2 KB, Nov 23) — `WordProbingDataset`, `WordProbingCollate`, `extract_word_activations` (word_id alignment + mean-pool token spans), `train_and_evaluate_probe` (cuML + GridSearchCV), `get_best_classifier`. Conf: **high**.

---

## 2. Output / figure provenance

### 2.1 `img/` subdirectories

| Subdir | # files | Producer | Pattern | In .tex? | In slides? | Tie-in |
|---|---|---|---|---|---|---|
| `img/ablate_prob/` | 12 | `visualize_ablate.py` (v1, via CLI `--output_dir`) | `heatmap_{exp}_{C}_{V}_{k}.png` | No | Yes (Dec slides) | H2/H3 (superseded by v2) |
| `img/ablate_prob_bar/` | 16 | `visualize_ablate_bar.py` (v2) | `barplot_{source\|target}_{C}_{V}_{k}.png` | No | No | H2/H3 |
| `img/ablate_v2_prob_bar/` | 4 | `visualize_ablate_bar.py` (v2 rerun, Jan) | same | No | No | H2/H3 |
| `img/ablate_v3_prob_bar/` | 2 | `visualize_ablate_bar.py` (v3, Mar 15; probe-targeted) | same | No | No | H2/H3 |
| `img/input_output/` | 29 subdirs × 3 | `input_output_features_visualize.py` | `{Lang}_{C}_{V}/{jaccard_topk,signal_input,signal_output}.png` | No | Yes ("Input / Output features") | **H2** |
| `img/pca_steering_vectors/` | 25 | `pca_steering_vectors.py` | `concept={C}_value={V}_language={…}_layer={N}.png` | No | Yes (extensive) | LEGACY / H2-adjacent |
| `img/perplexity_bleu/` | 15 | `visualize_perplexity_bleu{,_sorted,_correlation,_error_bar}.py` | `{aya\|llama}_{source,target,joint}_competence*.png`, `perplexity_plot_*.png`, `perplexity_vs_bleu_*.png` | **aya_{source,target}_competence.png** referenced via `images/…` (path mismatch) | Yes ("Perplexity Scores → BLEU Scores Linear Model") | **H1** |
| `img/probe_performance/` | 7 | `visualize_probe_results.py` | `{C}_{V}_accuracy_vs_layer.png`, `_test_accuracy_distribution.png`, `all_concepts_c_value_vs_layer.png` | No | Yes (probing grid search) | INFRA |
| `img/sentence_input_features/` | 46 | `sentence_input_features_visualize.py` (+ `zzz_input_space.py`) | `feature_language_distribution_{C}_{V}_top{K}.png`, `jaccard_similarity_{C}_{V}_top{K}.png`, `{Lang}_{C}_{V}_magnitudes.png` | No | Yes ("Sentence input features") | **H2** |
| `img/sv_cs_heatmaps/` | 18 | **Likely `visualize_steering_vectors.py`** (no literal grep hit for dir name; run via `run/run_visualization_steering_vectors.sh`). Subagents disagreed — see §5. | `sv_{C}_{V}.png` | No | Yes ("SV cossim heatmaps") | LEGACY |

### 2.2 `outputs/` subdirectories

| Subdir | Producer | Consumer | Status | Notes |
|---|---|---|---|---|
| `ablation_plots/` | `visualize_ablate_bar.py` (manual `--output_dir`) | – | Stale snapshot | duplicate of img/ablate_prob_bar contents |
| `ablation_results/` | `ablate.py` (default) | `visualize_ablate_bar.py`, `visualize_ablate.py` | Fresh | 7 `results_{config}.jsonl` |
| `ablation_results_3_23/` | `ablate.py` (per `aaa_run.md`) | `visualize_ablate_bar.py` (v3) | Fresh (Mar 23) | mono_{input,output,random} w/ probe filter |
| `ablation_results_debug/` | `ablate.py` (debug CLI) | – | Stale | 2 files, Mar 23 |
| `ablations_manual/` | (no script — manual) | – | Legacy, 1 file (Oct 14) | |
| `activations/` | `collect_activations.py` | `collect_steering_vectors.py`, `input_features.py`, `sentence_input_features.py`, `word_probes.py` (indirectly), others | Fresh (Oct 27) | 207 parquet files, 23 langs × 9 layers (if not short-circuited by line 112 hardcode) |
| `activations_flores/` | `collect_sae_activations.py` | `attribution_flores.py` (implicit) | Empty | Never populated |
| `attributions/` | `attribution_flores.py` and/or `jannik_output_features.py` | `input_output_features_visualize.py`, `ablate.py` (via `load_effects_files`) | Fresh | 53 timestamped subdirs, each with `config.json` + `effects_{src}_{tgt}.pt` |
| `attributions_notall/` | Variant of attribution_flores with a subset? | – | Unclear | 20 dirs, Dec 8 |
| `attributions_ud.tar.gz` | Manual export | – | Unreferenced | 661 MB, Dec 8 |
| `attributions_ud_all.tar.gz` | Manual export | – | Unreferenced | 1.9 GB, Dec 15 |
| `counterfactual_attribution/` | `counterfactual_attribution.py` | `analyze_counterfactual_results.py`, `visualize_counterfactual_results.py` | Fresh (Apr 13–17) | aggregated json + per-pair json + `raw_gradients/` + `plots/` |
| `features/` | `find_features.py` | – | Stale (Oct 14) | 19 JSON files per (concept, value) + `_shared.json` |
| `perplexity_bleu/` | `perplexity.py`, `linear_model_ppl_bleu.py`, external BLEU source | `visualize_perplexity_bleu*.py`, `combine_csvs.py` | Fresh (Mar 2) | `bleu_results*.csv`, `perplexity_results_{model}.csv`, `combined_results_{model}.csv`, `linear_models/linear_{coeffs,predictions}_{model}_{raw\|log}_{joint\|nojoint}.csv` |
| `perplexity_comparison/` | `perplexity_comparison.py` | `visualize_perplexity_bleu_correlation.py`, `visualize_perplexity_error_bar.py` | Fresh (Feb 19–20) | `error_rates_by_language_{model}.json`, `perplexity_matrices_{model}.npz` |
| `probes/` | `word_probes.py`, `train_probes.py` | `attribution_flores.py`, `ablate.py`, `visualize_probe_results.py` | Fresh (Nov 22) | 14 `all_probe_results_*.csv` (grid-search iterations same day), `word_probes/` (300 .joblib), `word_probes.zip`, `processed_sentences/` (per-lang/split cached pyconll parses), `lang10_all.csv`, `all_probe_results_{r1,tense}.csv` |
| `sae_activations/` | (`zzz_input_space.py` would, but never run) | – | Empty | placeholder |
| `sentence_input_features/` | `sentence_input_features.py` | `sentence_input_features_visualize.py`, `input_output_features_visualize.py`, `ablate.py` | Fresh (Dec 1) | 9 language subdirs with `{C}/{V}/diff_vector.pt` |
| `steering/` | `steer.py` | – | Legacy (Oct 20) | |
| `steering_vectors/` | `collect_steering_vectors.py` | `pca_steering_vectors.py`, `visualize_steering_vectors.py` | Active (Oct–Nov) | `all/` canonical, `all_corrupted/`, `nova/` (test subset), `test/` |
| `tags/` | – | – | Empty | |
| `token_analysis/` | `analyze_tokens.py` | `visualize_tokens.py` | Fresh (Apr 17) | JSON files + `mood/`, `multilang/`, `other_feats/` subdirs (per config) |
| `token_analysis_html/` | `visualize_tokens.py` | – | Fresh (Apr 17) | HTML files mirroring JSON |
| `visualization_steering_vectors/` | `visualize_steering_vectors.py` | – | Fresh (Oct 20) | ~180 PNGs: `cosine_similarity_layer{N}_{C}_{V}.png` |
| `zzz_dep_activations/` | earlier pipeline phase | – | Legacy | 19.7 GB monolithic `activations_all.parquet` + `test/` subdir |

### 2.3 .tex `\includegraphics` audit

`.tex` uses the prefix `images/`, but the repo uses `img/`. Literal references:

| .tex path | On disk? | Notes |
|---|---|---|
| `images/linear_effects_ranks.png` | **Missing** | No rank-1 approximation code exists. Claim of 88% faithful is in prose only. |
| `images/linear_effects_llama.png` | **Missing** | Same as above. |
| `images/noisy_vs_module.png` | **Missing** | Slides have a figure but not this exact filename; marked "potentially a good figure 1" in .tex. |
| `images/aya_source_competence.png` | Exists at `img/perplexity_bleu/aya_source_competence.png` | Path mismatch. |
| `images/aya_target_competence.png` | Exists at `img/perplexity_bleu/aya_target_competence.png` | Path mismatch. |
| `images/typological_similarity.png` | **Missing** | No lang2vec/WALS code in repo at all. |
| `images/cosine_sim_llama_normalized.png` | **Missing** | Slides reference a cosine-sim heatmap but no normalized-llama variant on disk. |

---

## 3. Current-framing ↔ code / figure / output map

### H1 — BLEU ≈ f(P_src, P_tgt)

- **Producing code:** `perplexity.py`, `perplexity_comparison.py`, `linear_model_ppl_bleu.py`, `combine_csvs.py`.
- **Figures:** `img/perplexity_bleu/{aya,llama}_{source,target,joint}_competence*.png` (current preferred), `perplexity_plot_*.png`, `perplexity_vs_bleu_llama[_sorted].png`.
- **Data:** `outputs/perplexity_bleu/*.csv`, `linear_models/*.csv`, `outputs/perplexity_comparison/*.{json,npz}`.
- **Tests:** `tests/test_perplexity_comparison.py` (mock-data unit tests).
- **.tex status:** §"A Motivating Observation" references `linear_effects_ranks.png` and `linear_effects_llama.png` **that do not exist** and an "88% faithful rank-1 approximation" **with no code producing it**. "Correlating Multi-BLiMP" subsection is empty; PER-from-MultiBLiMP is computed by `perplexity_comparison.py` but not prominent in final plots.
- **Repo ahead of paper:** the `{aya,llama}_{source,target,joint}_competence*.png` figures exist on disk and match the current framing; .tex hasn't caught up beyond two `\includegraphics` calls with wrong path prefix.

### H2 — Shared "semantic hub" (input/output feature overlap)

- **Producing code:**
  - Input features: `sentence_input_features.py` (sentence-level diff-in-means in SAE latent space).
  - Output features: `attribution_flores.py` (gradient attribution from probe logits).
  - Comparison: `input_output_features_visualize.py`.
  - Ablation causal-check: `ablate.py` (mono vs multi; input vs output; + random baselines).
  - Counterfactual (English-only, Apr 2026): `counterfactual_attribution.py` + `analyze_*` + `visualize_*`.
- **Figures:** `img/input_output/{Lang}_{C}_{V}/*.png`, `img/sentence_input_features/*.png`, `img/ablate_v3_prob_bar/*.png`, `outputs/counterfactual_attribution/plots/*.png`.
- **Data:** `outputs/sentence_input_features/…diff_vector.pt`, `outputs/attributions/{timestamp}/effects_{src}_{tgt}.pt`, `outputs/ablation_results/*.jsonl`, `outputs/counterfactual_attribution/*.json`.
- **Tests:** `tests/test_ablate_{batch,edge_cases,mocks,integration,statistics}.py`.
- **.tex status:** "Finding multilingual features" and "Assessing feature impact in a translation task setting" describe input/output features and ablations closely. Word-level procedure is specified in .tex but no active script implements it (sentence-level is what runs). Counterfactual-attribution "Potential Experiment" is implemented in code but the paper text hasn't incorporated the results.
- **Repo ahead of paper:** counterfactual attribution pipeline, `img/input_output/` (29 Jaccard-per-signal plots), input-vs-output Jaccard / feature-language-distribution plots — all exist, none rendered in the .tex via `\includegraphics`.

### H3 — Monolingual FT ≈ parallel-corpus FT

- **No finetuning code in this repo.** Grep for `LoRA|lora|peft|finetune|trainer|continual|xhosa|georgian` returned zero matches in `scripts/`, `src/`, `run/` (only TODO/jake/jannik annotations in `reference_paper.tex` lines 456–457 mention Xhosa).
- **.tex §"Adding a New Language to a Pretrained LLM"** is an empty subsection stub.
- **Nothing in the repo can currently produce H3 evidence.** Slides indicate Jannik was assigned the finetuning pipeline; results, if any, live elsewhere.
- **Paper ahead of repo** for this claim.

### LEGACY analyses

- **Steering vectors + PCA** — extensive code (collect / generate / pca / visualize / steer), extensive figures (`img/pca_steering_vectors/`, `img/sv_cs_heatmaps/`, `outputs/visualization_steering_vectors/` ~180 PNGs), extensive outputs (`outputs/steering_vectors/`). .tex is silent. Slides cover this heavily from summer/fall.
- **Typological similarity / lang2vec / WALS** — zero code, zero data, zero figures. Orphaned only in .tex (`images/typological_similarity.png`) and slides ("lang2vec — uses WALS_syntax").
- **Representation-similarity (cosine heatmap across languages)** — captured in `outputs/visualization_steering_vectors/` (180 cosine-similarity PNGs) and `img/sv_cs_heatmaps/`. Not referenced in .tex.
- **Aya-vs-Llama** — implicit everywhere: `config.py MODEL_TO_ID`, every perplexity/BLEU plot has both models. No dedicated "comparison" script.
- **Qualitative steering examples** — `scripts/steer.py` + `examples/*.txt` (`number_plural_{english,spanish}.txt`, `tense_past_english.txt`, `steer_english.txt`). Not referenced from any active pipeline.

---

## 4. Experiment-family deep dives

### (a) Input/output features & feature sharing

**Goal:** H2. Show that the top SAE features for a grammatical concept (measured monolingually as diff-in-means) are the same as the top features attributing to a probe's prediction during translation (output features), across languages.
**State:** Running. Input features computed for 9 languages × multiple (concept, value) pairs (`outputs/sentence_input_features/`). Output features computed as 53 timestamped runs in `outputs/attributions/` (Dec 6–12). 29 overlap-plot subdirs in `img/input_output/` produced Dec 15.
**Scripts:** `sentence_input_features.py` → `sentence_input_features_visualize.py`; `attribution_flores.py` (and likely `jannik_output_features.py`) → `input_output_features_visualize.py`; `find_features.py` (older probe-weight-based feature discovery).
**Versions:** Only one "version" per pipeline, but `input_features.py` is a superseded word-level draft and `zzz_input_space.py` is an older exploratory variant.
**.tex coverage:** Section "Finding multilingual features" + "BLEU and Perplexity" (Q1/Q2) describes this in detail. Word-level procedure is specified but not implemented; in-context-translation with prefilled demonstrations (Q2 step 1) is hinted at but `attribution_flores.py` uses FLORES pairs directly without explicit few-shot demo construction. **Paper partially ahead of repo (word-level); repo partially ahead of paper (`img/input_output/` not referenced).**
**Open questions:** §6 Q1–Q3.

### (b) Ablation (v1 / v2 / v3)

**Goal:** H2 causal check — ablating the top-K features should reduce correct-reference probability in both monolingual and translation settings, approximately equally.
**State:** Complete through v3 (Mar 23). Seven experiment configs × multiple concept/value/k combos ran in `outputs/ablation_results/` (Dec 22) and `ablation_results_3_23/` (Mar 23).
**Versions:**
- **v1** (Dec 22) — `scripts/ablate.py` + `visualize_ablate.py`, heatmap (src × tgt) output to `img/ablate_prob/`, 256-sample runs, no probe filter.
- **v2** (Jan 12) — `visualize_ablate_bar.py` bar-chart redesign (x=language, hue=experiment). Reused v1 data. `img/ablate_prob_bar/`, `img/ablate_v2_prob_bar/`.
- **v3** (Mar 15 → Mar 23) — added `--use_probe --probe_layer 32 --probe_n 1024` to restrict ablation to probe-positive token positions; added `frac_active_at_ablated` field to JSONL; new `outputs/ablation_results_3_23/`; new figures `img/ablate_v3_prob_bar/`. Only `mono_*` experiments rerun at v3.
**Tests:** 5 test files (`test_ablate_batch`, `_edge_cases`, `_integration`, `_mocks`, `_statistics`). Integration test is skipped (GPU). 2 test files were modified Apr 13 alongside code.
**.tex coverage:** "Procedure for ablation experiments" subsection matches code items closely (Δp normalization, pre-filled target, probe-targeted positions, mono/multi, input/output, random baseline — all implemented). No ablation figure is `\includegraphics`'d.
**Open questions:** §6 Q4, Q5.

### (c) Perplexity ↔ BLEU linear model

**Goal:** H1. Show `BLEU ≈ a + β₁P_src + β₂P_tgt` (with or without interaction) fits well.
**State:** Fit and visualized for both Aya and Llama; multiple variants (raw/log, joint/nojoint) produced Mar 2. Slides report R² 0.11 Aya / 0.02 Llama (raw+joint).
**Scripts:** `perplexity.py` (corpus PPL) → `combine_csvs.py` → `linear_model_ppl_bleu.py`. In parallel, `perplexity_comparison.py` computes PER over minimal-pair datasets; `visualize_perplexity_bleu_correlation.py` uses PER for the current competence figures.
**.tex coverage:** §3 "A Motivating Observation" references two figures that do not exist and a "rank-1 approximation 88% faithful" claim whose code is absent. §"BLEU and Perplexity" shows `aya_{source,target}_competence.png` (path mismatch via `images/`). "Correlating Multi-BLiMP" is empty.
**Missing analyses:** rank-1 SVD approximation of the coefficient space; Word-Error-Rate variant (considered in slides, rejected in favor of PER); active Multi-BLiMP visualization.
**Open questions:** §6 Q6, Q7.

### (d) Steering vectors & PCA (LEGACY)

**Goal originally:** representation-similarity analysis across languages (fall 2025 slides). **Current relevance:** SV is diff-in-means in **residual-space**, while `sentence_input_features.py` is diff-in-means in **SAE latent space** — same method, different projection. Arguably the input-features pipeline is the descendant.
**State:** Complete. 11 concepts × multiple values × 8+ languages × 9 layers of SVs in partitioned parquet at `outputs/steering_vectors/all/`. PCA plots in `img/pca_steering_vectors/` (25 PNGs). Cosine-sim heatmaps in `img/sv_cs_heatmaps/` (18 PNGs) and `outputs/visualization_steering_vectors/` (~180 per-layer PNGs).
**Scripts:** `collect_steering_vectors.py` (modern, parquet), `generate_steering_vectors.py` (older, self-contained), `zzz_collect_steering_vectors.py` (deprecated), `pca_steering_vectors.py`, `visualize_steering_vectors.py`, `steer.py`.
**.tex coverage:** None.
**Open questions:** §6 Q8.

### (e) Counterfactual attribution (PROTOTYPE, Apr 2026)

**Goal:** H2, via gradient-based indirect-effect attribution on 30 English minimal-pair sentences.
**State:** Pipeline complete. Aggregated and visualized (Apr 17). 29/30 pairs processed (1 skipped — multitoken counterfactual).
**Scripts:** `counterfactual_attribution.py` → `analyze_counterfactual_results.py` → `visualize_counterfactual_results.py`.
**.tex coverage:** §"Potential Experiment" (bottom of .tex) itemizes the exact method. **Repo ahead of paper** — no paper subsection discusses the results yet.
**Open questions:** §6 Q9.

### (f) Probing infrastructure

**Goal:** INFRA. Train and serve linear probes for every (language, concept, value, layer) combination needed downstream.
**State:** Current authoritative trainer is `word_probes.py` (word-level, cuML GPU grid search). 300 joblib probes on disk at `outputs/probes/word_probes/` in `{Lang}_{Concept}_{Value}_l{layer}_n{n}.joblib` format. Sentence-level `train_probes.py` is legacy. Probe performance plotted in `img/probe_performance/` (7 PNGs).
**.tex coverage:** Related Work lists "Training mass-mean probes" as a stub — this terminology does **not** match the actual method (sklearn / cuML `LogisticRegression` with grid-searched `C`, `class_weight='balanced'`, QN/liblinear solvers). If the framing is literally mass-mean, the code doesn't match. If it's logistic, the .tex is misleading.
**Open questions:** §6 Q10, Q11.

### (g) Activation / SAE collection

**Goal:** INFRA. Cache residual-stream activations from UD treebanks (and ideally FLORES) per language × layer.
**State:** UD path ran Oct 27; `outputs/activations/` has 23 languages × 9 layers partitioned parquet (~70 MB each). FLORES path `outputs/activations_flores/` never populated. `outputs/sae_activations/` never populated — SAE is applied on-the-fly, not cached. Line 112 of `collect_activations.py` hardcodes `for layer in [32]:` with a TODO — unclear how this reconciles with 9 layers on disk (possibly 9-layer data was collected before the hardcode was introduced, or the hardcode was toggled).
**Scripts:** `collect_activations.py`, `collect_sae_activations.py`, `src/lang_probing_src/activations.py`, `autoencoder.py`, `sentence_dataset_class.py`, `dataloader_factory.py` (unused by collection scripts), `data.py`.
**.tex coverage:** None of the collection mechanics (partitioning, pooling-within-trace, layer selection rationale) are in the Methods section.
**Open questions:** §6 Q12.

### (h) Finetuning / language-addition

**Status: absent from this repo.** No grep matches for `LoRA|peft|xhosa|georgian|finetune|trainer|continual`. Slides assign it to Jannik; `reference_paper.tex` §"Adding a New Language" is an empty stub. H3 evidence is not currently producible from this repo alone.

### (i) Legacy analyses in slides, orphaned in repo

- Typological similarity (lang2vec / WALS / SSWL / Ethnologue): **no code**, no data, no figures. `images/typological_similarity.png` in .tex is dead.
- Aya-vs-Llama dedicated comparison (BLEU matrices per model, representation similarity heatmaps): **no dedicated script**, though model-split outputs exist across many scripts (perplexity/BLEU scripts parameterize by `--model`).
- `projectnb/mcnet/jbrin/lang-probing/cos_img/cos_img_steering_vectors.png` in the Nov git history — that literal path (treating `projectnb` as a subdir) was committed accidentally and does not exist as a directory in the repo now.
- Qualitative early-ablation examples ("The ducks → The duck") — survive as `examples/*.txt` (4 files), read only by the dead `run_ablation.py`.

---

## 5. Things you should know (findings)

1. **The .tex is systematically under-specified relative to the repo.** The `\includegraphics` path prefix is `images/` but the actual dir is `img/` — 5 of 7 referenced figures are **missing** on disk (including the headline "Mark's linear model" figures `linear_effects_ranks.png`, `linear_effects_llama.png`, and the potential fig-1 `noisy_vs_module.png`). The other 2 exist but under `img/perplexity_bleu/`.
2. **`collect_activations.py` line 112 hardcodes `for layer in [32]:`** with a "TODO: remove when done" comment. `COLLECTION_LAYERS` in `config.py` is `[0,4,8,12,16,20,24,28,31]` (note 31, not 32, vs. `LAYERS = [0,4,8,12,16,20,24,28,32]`). 207 parquets on disk suggest all 9 layers were collected at some earlier point; unclear whether the hardcode was toggled between runs. Worth checking before rerunning.
3. **`attribution_flores.py` uses `from src.config`** (wrong module — should be `lang_probing_src.config`). The top of the file has a `sys.path.insert` hack; may or may not work depending on CWD.
4. **`outputs/probes/` has 14 `all_probe_results_2025-11-22_*.csv`** files, all from the same day, roughly hourly — these are grid-search / rerun iterations. Only the latest is probably authoritative; the rest are checkpoints. `filter_probe_training_log.py` only filters *run-log* text, not these CSVs.
5. **Probe filename format mismatch:** `word_probes.py` writes `{Lang}_{C}_{V}_l{layer}_n{n}.joblib`; `analyze_tokens.py` expects `…_probe_layer{layer}_n{n}.joblib`. Probe loading in the token-analysis pipeline is silently disabled as a result.
6. **`run_ablation.py` is dead.** It imports `lang_probing_src.ablation` which does not exist (there's `ablate.py` and `zzz_ablation.py`). `src/lang_probing_src/zzz_ablation.py` and `scripts/run_ablation.py` are a legacy pair that was never updated after the v1→v3 refactor.
7. **`scripts/jannik_output_features.py`** — purpose unclear from file listing alone; the output-feature dir `outputs/attributions/` has many runs but I was not able to pin which were from `attribution_flores.py` vs `jannik_output_features.py`. See §6 Q1.
8. **`outputs/attributions/` vs `outputs/attributions_notall/`** — two parallel output dirs, the latter with 20 subdirs (Dec 8) probably representing a specific subset (perhaps only probe-positive samples). No script explicitly writes to the `_notall` variant.
9. **Tar archives `outputs/attributions_ud{,_all}.tar.gz` (~2.5 GB total)** are unreferenced by any code. Likely manual exports (e.g., for sharing with a collaborator); not part of the pipeline.
10. **`sv_cs_heatmaps/` producer uncertain:** one subagent said `visualize_steering_vectors.py` writes there, another reported that `collect_steering_vectors.py` had unreachable code after a `return` that would have written the heatmaps. Grep finds no literal hit for `sv_cs_heatmaps` in scripts — consistent with the path being assembled from `IMG_DIR + concept-value`. Worth quickly verifying which script actually produced the 18 PNGs there.
11. **`zzz_` files are not uniformly "dead."** Audit:
    - `scripts/zzz_scratchpad.py` — throwaway (token counting).
    - `scripts/zzz_input_space.py` — superseded exploratory diff-in-means.
    - `scripts/zzz_collect_steering_vectors.py` — superseded by `collect_steering_vectors.py`.
    - `src/lang_probing_src/zzz_ablation.py` — superseded library, still imported by the dead `run_ablation.py`.
    - `outputs/zzz_dep_activations/` — 19.7 GB monolithic parquet from an earlier collection schema.
12. **`aaa_` files:** just `scripts/aaa_run.md` (v3 run commands). Named so it sorts first.
13. **`examples/`** — 4 small seed files (`number_plural_{english,spanish}.txt`, `tense_past_english.txt`, `steer_english.txt`), read only by the dead `run_ablation.py`. The newer `ablate.py` sources examples from FLORES, not from these files.
14. **`logs/`** — 29 files, all `collect_activations_*.log` from Oct 27. Not referenced by any active code.
15. **Multi-BLiMP PER is computed but not prominently shown.** `perplexity_comparison.py` → `error_rates_by_language_{model}.json` feeds `visualize_perplexity_bleu_correlation.py` (the current H1 figures). The .tex has an empty "Correlating Multi-BLiMP" subsection.
16. **Repo ahead of paper:** `img/input_output/` (29 Jaccard/signal plots, H2), all ablation bar charts (H2/H3), the counterfactual-attribution pipeline and plots (H2), the perplexity-BLEU competence figures beyond the 2 in .tex (H1), and most of the legacy SV/PCA figures (H2-adjacent). None of these are referenced in `\includegraphics`.
17. **Paper ahead of repo:** §"Adding a New Language" (H3 finetuning — entire subsection is empty and no code exists); rank-1 SVD approximation and the "88% faithful" claim for the linear model; typological-similarity figure `typological_similarity.png`; cosine-sim-normalized-Llama figure. The word-level input-feature procedure with priority-based negative sampling (lines 243–254) has no script.
18. **Slides ahead of both:** Aya-vs-Llama BLEU matrix as a dedicated figure; lang2vec / WALS typology; BostonHacks hackathon prep (completely orphan).
19. **Name drift:** `LAYERS=[…,32]` vs `COLLECTION_LAYERS=[…,31]` in the same config; the v3 `aaa_run.md` passes `--probe_layer 32` (consistent with `LAYERS`) but activations-on-disk partition names end at layer 31. Probe at layer 32 vs activations at layer 31 is a subtle mismatch worth verifying.
20. **`.gitignore` + `projectnb/mcnet/jbrin/lang-probing/cos_img/`** — the Nov 12 commit `83cde27` committed a file at the literal relative path `projectnb/mcnet/jbrin/lang-probing/cos_img/cos_img_steering_vectors.png`. No such directory exists in the current tree, so the file was either untracked subsequently or this was an untracked path accident.

---

## 6. Questions (for you to answer; each unlocks multiple scripts)

1. **`scripts/jannik_output_features.py`** — What is the intended purpose relative to `attribution_flores.py`? Are both writing to `outputs/attributions/`? Which one should be treated as canonical for "output features"?
2. **Word-level input/output feature procedure** — The .tex spells out a word-level procedure with priority-based negative sampling; active code (`sentence_input_features.py`) is sentence-level. Is the word-level version intentionally deferred, or is there a script elsewhere?
3. **`outputs/attributions_notall/` vs `outputs/attributions/`** — What's the difference? Is `_notall` a subset (e.g., only samples where the probe fires), or a separate experimental variant?
4. **v3 ablation:** `aaa_run.md` only covers `mono_{input,output,random}`. Were the `multi_*` experiments rerun with `--use_probe` too, or is v3 intentionally monolingual-only?
5. **Ablation output naming:** `EXP_CONFIGS` uses `multi_input_random` / `multi_output_random`; disk has `results_multi_random_src.jsonl` / `results_multi_random_tgt.jsonl`. Which is canonical, or is one a pre-rename artifact?
6. **Linear model rank-1 approximation:** Was the "88% faithful for Llama" number from an SVD of the coefficient space, a residual analysis, or a hand-computed summary statistic? Is the code for it somewhere I missed?
7. **Perplexity CSVs:** `outputs/perplexity_bleu/perplexity_results_{aya,llama}.csv` are ~600 bytes each — these look like tiny pilot runs. Are the `combined_results_{aya,llama}.csv` (28 KB each, Mar 2) the current source of truth for downstream plots?
8. **Steering vectors current role:** are the SV/PCA figures still considered live evidence for H2, or has the project fully transitioned to SAE-space input features, with SVs kept as background?
9. **Counterfactual attribution scope:** The current 30 pairs are all English. Does the plan include multilingual counterfactuals, or is the design English-only as a prototype?
10. **Probing terminology:** The .tex Related Work says "Training mass-mean probes" (a stub). The code trains logistic-regression probes. Is the .tex wording a legacy from an earlier mass-mean-vector plan, or do you intend to describe the LR probes as "mass-mean" loosely?
11. **Probe layer:** `aaa_run.md` uses `--probe_layer 32`. Activations on disk are partitioned up to `layer=31`. Was the layer-32 probe trained on activations that were collected post hoc (outside the Oct 27 run), or is there a layer numbering off-by-one?
12. **`collect_activations.py` line 112:** is the `for layer in [32]:` hardcode currently active, or was it patched back to `COLLECTION_LAYERS` before the 23-lang × 9-layer collection? The 207 parquets imply the latter.

---

## 7. Top findings (surprises and biggest confidence gaps)

**Most surprising / least certain:**
1. **The .tex is badly out of sync with disk.** 5 of 7 referenced figures are missing; the `images/` vs `img/` path prefix is wrong throughout; the "rank-1 approximation 88% faithful" claim has no producing code I could find.
2. **`run_ablation.py` + `examples/*.txt` + `src/lang_probing_src/zzz_ablation.py` form an orphan triple** — all imported-together, all dead. The IMPLEMENTATION_SUMMARY.md (which itself is Oct 2025 and stale) still holds them up as the primary workflow.
3. **The finetuning pipeline for H3 is completely absent from this repo.** Not partial, not scaffolded — zero. Any H3 evidence will come from a different codebase (likely Jannik's).
4. **`outputs/attributions/` has 53 runs** over a week in December, plus a separate `attributions_notall/` with 20 more, plus two ~GB-scale tar.gz backups — but I could not cleanly attribute producers (`attribution_flores.py` vs `jannik_output_features.py`) from code alone. The provenance of the main H2 evidence (output features) is less transparent than the input-features side.
5. **`collect_activations.py` has a hardcoded `for layer in [32]:`** that contradicts its config and its own on-disk output; this was presumably temporary (per the TODO comment) but the code state is confusing. If you rerun collection, check this first.

**Biggest gaps between repo and .tex:**
- **Repo far ahead of .tex on H2 visualizations:** `img/input_output/` (29 subdirs with Jaccard/signal plots), all ablation bar charts, the entire counterfactual-attribution prototype (code + JSON + plots). None of these are in `\includegraphics`.
- **.tex ahead of repo on H1 figure claims:** `linear_effects_ranks.png`, `linear_effects_llama.png`, the "88% rank-1" number, and the "Correlating Multi-BLiMP" subsection all have no code support.
- **.tex ahead of repo on H3:** §"Adding a New Language" references finetuning results that do not exist in this repo.
- **Legacy gap:** the typological-similarity analysis lives *only* in slides and one dead `\includegraphics`. No code or data supports it. If keeping it in the paper, it has to be reconstructed or cut.
