# Overnight ledger — 2026-04-22 session

Multilingual extension of counterfactual-attribution for SAE features on Llama-3.1-8B at layer 16.
Languages: English (baseline), French, Spanish, Turkish, Arabic.
Data sources: Multi-BLiMP (`jumelet/multiblimp`), existing 30 handcrafted English pairs, template-generated supplement pairs.

Hypothesis tags:
- **H2** — shared multilingual features; input/output feature overlap.
- **H4** — translation reuses monolingual LM circuits; features transfer across languages.
- **INFRA** — tooling (dashboards, bug-audit, data loaders).

## Angles

### cross_concept  (`#3`)
- **Hypotheses:** H2, H4
- **Status:** pending
- **Question:** Does the same SAE feature carry multiple grammatical concepts (e.g. tense=past AND num=plur)?
- **Method:** Binomial null under i.i.d. top-100 sampling over 32768 features; Bonferroni-corrected p<0.01.
- **Findings:** _pending_
- **Caveats:** _pending_
- **TODOs:** Cross-lingual cross-concept (features multi-concept in >1 language).
- **Figures:** _pending_
- **Dashboard links:** _pending_

### arabic_dual_english  (`#4`)
- **Hypotheses:** H4
- **Status:** pending
- **Question:** Do features identified via Arabic Number=Dual attribution fire more on English sentences mentioning "two/both/pair" than on numeric-free controls?
- **Method:** Welch's t + Cohen's d on whole-sentence-mean and ±3-token-window activations; firing-density-matched null of 20 random features.
- **Findings:** _pending_
- **Caveats:** _pending_
- **TODOs:** _pending_
- **Figures:** _pending_
- **Dashboard links:** _pending_

### sign_flip  (`#5`)
- **Hypotheses:** H2 challenger
- **Status:** pending
- **Question:** Does a feature attribute positively to French Fem but negatively to Spanish Fem (and vice versa), revealing language-specific usage of the "same" SAE feature?
- **Method:** 2×2 subplot grid of signed grad×act over top-200 features (Fem/Fem, Masc/Masc, Fem/Masc, Masc/Fem); annotate top-10 opposite-sign features.
- **Findings:** _pending_
- **Caveats:** _pending_
- **TODOs:** _pending_
- **Figures:** _pending_
- **Dashboard links:** _pending_

### input_vs_output  (`#6` — holy grail)
- **Hypotheses:** H2, H4
- **Status:** pending
- **Question:** Features with high input-activation gap but low ablation effect (or vice versa) — especially when the asymmetry flips across languages (feature ablates hard in English but not French past-tense, etc.).
- **Method:** Per-feature scatter of Δact_input vs. Δablation; cross-lingual ablation matrix (source cell × ablation target cell).
- **Findings:** _pending_
- **Caveats:** _pending_
- **TODOs:** _pending_
- **Figures:** _pending_
- **Dashboard links:** _pending_

### decoder_logit_lens
- **Hypotheses:** INFRA
- **Status:** pending
- **Question:** For top features, what vocabulary tokens does `W_dec[f] @ W_unembed.T` promote most?
- **Method:** Single `[4096, vocab]` matmul, cached; per-feature top-20 tokens.
- **Findings:** _pending_
- **Caveats:** Logit-lens signal is weak for mid-stream layers; useful for dashboard context only.
- **Figures:** _pending_

### max_activating_tokens
- **Hypotheses:** H2
- **Status:** pending
- **Question:** What token contexts does each top feature fire strongest on, across all 5 languages' FLORES text?
- **Method:** Top-1% activating tokens per feature aggregated across FLORES devtest.
- **Findings:** _pending_
- **Figures:** _pending_

### coactivation_clusters
- **Hypotheses:** H2
- **Status:** pending
- **Question:** Within a cell's top-20 features, are they one tight cluster or independent signals?
- **Method:** Pearson correlation of activations across 1000 FLORES sentences → NetworkX spring layout.
- **Findings:** _pending_
- **Figures:** _pending_

### firing_density
- **Hypotheses:** H2
- **Status:** pending
- **Question:** Are top-attribution features narrow (concept-specific, sparse) or broad (general-syntactic, dense)?
- **Method:** 2D scatter of firing rate vs. attribution-rank specificity per cell.
- **Findings:** _pending_
- **Figures:** _pending_
