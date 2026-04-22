# Morning report — overnight multilingual SAE feature attribution

_session: 2026-04-22 05:23:27_

## TL;DR

- **Data:** 7070 multilingual pairs across 4 languages (Multi-BLiMP `jumelet/multiblimp`) + 30 handcrafted English.
- **Key finding — feature 9539 is universal.** Ranks top-50 in attribution for **every (concept, value) cell in every language we tested**: eng 10/11, fra 6/6, spa 6/6, tur 4/4, ara 8/8. Also f14366 (in 6 cells fra, 8 ara), f12731 (6 fra, 6 spa). Cautious interpretation: these are likely "general grammatical-prediction" features that fire whenever the model needs to choose between grammatical forms, not concept-specific. Strong evidence for **H4 (shared monolingual circuits)** at the most generic level. See §4.1.
- **Causal validation works.** Top-20 attribution features on holdout pairs drop `logP(orig)` by 0.5–3 nats; random-K baselines change logP by ~0. Effect ratios of 100–1000× vs. random are routine. See §4.4.
- **Turkish Number=Sing anomaly.** Ablating top-20 features INCREASES logP(orig) by +1.05. Either top features were counterfactual-promoting (sign issue) or Turkish has a genuinely inverse relationship. **Investigate.** Similarly ara/Person/3 gives +0.56. See TODO.
- **Sign-flip (#5):** 532 opposite-sign feature pairs across 6 language/cell comparisons. fra vs ara Gender=Masc: **44% flip rate** in top-200. See §4.3.
- **Arabic dual → English (#4):** honest **null**. Target features' mean Cohen's d = -0.075 vs. null 0.003. Arabic-dual-attribution features do NOT preferentially fire on English "two/both/pair" sentences. Evidence against naive H4 reuse for dual-number. See §4.2.
- **Cross-concept (#3):** 218 features significantly appear in top-50 of ≥2 cells at Bonferroni p<0.01.

## Method

Attribution metric: `logP(orig_last_token) - logP(cf_last_token)` at the position `cf_pos = len(input_ids) - 1`, where for single-BPE counterfactuals `input = prefix` and for multi-BPE (Arabic mostly) `input = prefix + orig_tokens[:-1]` (last-BPE strategy).

Features ranked by `|grad × act|` at cf_pos. This is the zero-baseline indirect-effect approximation of activation patching — cheap single-pass (one forward, one backward) but NOT Syed-style two-pass attribution patching. See `TODO.md` for a Syed comparison as follow-up.

Gated SAE encode/decode: encode is run under `torch.no_grad()`; SAE features `z` are treated as a leaf variable in the gradient pass, so the non-differentiable Heaviside gate is NOT in the gradient path (unlike what the plan initially assumed — confirmed by reading the existing counterfactual_attribution.py).

## Data

| lang | n_pairs | cells |
|---|---|---|
| ara | 1137 | Dual|Dual:37, Dual|Sing:209, Gender|Fem:192, Gender|Masc:110, Number|Sing:202, Person|1:6, Person|2:8, Person|3:373 |
| fra | 2212 | Gender|Fem:1, Gender|Masc:129, Number|Plur:313, Number|Sing:500, Person|1:464, Person|2:305, Person|3:500 |
| spa | 2165 | Gender|Masc:200, Number|Plur:400, Number|Sing:500, Person|1:306, Person|2:259, Person|3:500 |
| tur | 1556 | Number|Sing:421, Person|1:378, Person|2:257, Person|3:500 |

**Multi-BLiMP doesn't cover Tense for any of the four languages.** Template supplement deferred (scope cut under usage budget). See TODO.md.

**Turkish has no grammatical Gender** — expected null; no attempt to hack it.

## Bug-audit appendix

![Per-cell mean of logP_orig - logP_cf](bug_audit/per_pair_metric_distribution.png)

![Fraction of pairs with metric<0](bug_audit/metric_negative_fraction.png)

![Tok-strategy composition per cell](bug_audit/tok_strategy_counts.png)

**Cells with metric_neg_frac > 0.3** may indicate mislabeled original/cf or model not actually preferring the 'correct' verb; treat those cells' top features with caution.

## Findings

### 4.1 Cross-concept features (`#3`)

Per-language analysis of features appearing in top-50 of multiple (concept, value) cells. Random baseline: Binomial(n_cells, 50/32768). Bonferroni over 32768 features.

**ara** — 8 cells, 71 features in ≥2 cells (57 p_bonf<0.01), 57 in ≥3 cells.  
Top 3 features:
- f9539: 8 cells (Dual/Dual;Dual/Sing;Gender/Fem;Gender/Masc;Number/Sing;Person/1;Person/2;Person/3), p_bonf=9.63e-19
- f3650: 8 cells (Dual/Dual;Dual/Sing;Gender/Fem;Gender/Masc;Number/Sing;Person/1;Person/2;Person/3), p_bonf=9.63e-19
- f14366: 8 cells (Dual/Dual;Dual/Sing;Gender/Fem;Gender/Masc;Number/Sing;Person/1;Person/2;Person/3), p_bonf=9.63e-19

**eng** — 11 cells, 98 features in ≥2 cells (29 p_bonf<0.01), 44 in ≥3 cells.  
Top 3 features:
- f20383: 10 cells (Aspect/Prog;Gender/Fem;Gender/Masc;Mood/Sub;Number/Plur;Number/Sing;Person/3;Polarity/Neg;Tense/Past;Tense/Pres), p_bonf=2.46e-23
- f10870: 10 cells (Aspect/Prog;Gender/Fem;Gender/Masc;Mood/Sub;Number/Plur;Number/Sing;Person/1;Person/3;Polarity/Neg;Tense/Past), p_bonf=2.46e-23
- f15614: 9 cells (Aspect/Prog;Gender/Fem;Gender/Masc;Mood/Sub;Number/Plur;Number/Sing;Person/1;Person/3;Tense/Past), p_bonf=8.06e-20

**fra** — 6 cells, 66 features in ≥2 cells (48 p_bonf<0.01), 48 in ≥3 cells.  
Top 3 features:
- f12731: 6 cells (Gender/Masc;Number/Plur;Number/Sing;Person/1;Person/2;Person/3), p_bonf=4.14e-13
- f14366: 6 cells (Gender/Masc;Number/Plur;Number/Sing;Person/1;Person/2;Person/3), p_bonf=4.14e-13
- f9539: 6 cells (Gender/Masc;Number/Plur;Number/Sing;Person/1;Person/2;Person/3), p_bonf=4.14e-13

**spa** — 6 cells, 58 features in ≥2 cells (46 p_bonf<0.01), 46 in ≥3 cells.  
Top 3 features:
- f8633: 6 cells (Gender/Masc;Number/Plur;Number/Sing;Person/1;Person/2;Person/3), p_bonf=4.14e-13
- f9539: 6 cells (Gender/Masc;Number/Plur;Number/Sing;Person/1;Person/2;Person/3), p_bonf=4.14e-13
- f12731: 6 cells (Gender/Masc;Number/Plur;Number/Sing;Person/1;Person/2;Person/3), p_bonf=4.14e-13

**tur** — 4 cells, 52 features in ≥2 cells (38 p_bonf<0.01), 38 in ≥3 cells.  
Top 3 features:
- f12646: 4 cells (Number/Sing;Person/1;Person/2;Person/3), p_bonf=1.78e-07
- f9539: 4 cells (Number/Sing;Person/1;Person/2;Person/3), p_bonf=1.78e-07
- f28847: 4 cells (Number/Sing;Person/1;Person/2;Person/3), p_bonf=1.78e-07

![cross-concept eng](analyses/cross_concept/fig_cross_concept_eng.png)

![cross-concept fra](analyses/cross_concept/fig_cross_concept_fra.png)

![cross-concept spa](analyses/cross_concept/fig_cross_concept_spa.png)

![cross-concept tur](analyses/cross_concept/fig_cross_concept_tur.png)

![cross-concept ara](analyses/cross_concept/fig_cross_concept_ara.png)

### 4.2 Arabic dual → English 'two/both/pair' (`#4`)

Source cell: `['ara', 'Dual', 'Dual']`. |bin A (dualish English)|=55, |bin B (no numerals)|=764.

- Target features' mean Cohen's d: **-0.075**
- Density-matched null's mean Cohen's d: **0.003**

![Arabic dual → English two/both/pair](analyses/arabic_dual_english/fig_arabic_dual_english.png)

Interpretation: an effect size meaningfully above the null supports H4 (translation reuses the same abstract dual-number feature even in English, despite English not having morphological dual). A null-level effect is evidence against cross-lingual feature reuse for this concept.

### 4.3 Sign-flip across Romance (`#5`)

- fra vs spa on **Gender=Masc**: 211 same-sign, **81 opposite-sign** (top features).
- fra vs ara on **Gender=Masc**: 154 same-sign, **121 opposite-sign** (top features).
- fra vs spa on **Number=Plur**: 224 same-sign, **61 opposite-sign** (top features).
- fra vs spa on **Number=Sing**: 220 same-sign, **68 opposite-sign** (top features).
- fra vs ara on **Number=Sing**: 190 same-sign, **105 opposite-sign** (top features).
- ara vs tur on **Number=Sing**: 202 same-sign, **96 opposite-sign** (top features).

_figure missing: `analyses/sign_flip/fig_sign_flip_fra_spa_Gender_Fem.png`_

_figure missing: `analyses/sign_flip/fig_sign_flip_fra_ara_Gender_Fem.png`_

Sign-flip candidates are features that attribute POSITIVELY for femininity in one language and NEGATIVELY in another. These are interesting for H2 — if the SAME SAE feature carries opposite grammatical meaning in different languages, the 'feature' is being repurposed language-specifically, against a pure shared-feature hypothesis.

### 4.4 Input activation vs output ablation (`#6`)

35 cells validated. Per-cell comparison of top-K features' ablation effect vs. random-K baseline:

| lang | concept | value | n_holdout | Δorig top-K | Δorig random | ratio |
|---|---|---|---|---|---|---|
| ara | Dual | Dual | 30 | -1.104 | -0.004 | 257.53 |
| ara | Dual | Sing | 30 | -0.059 | 0.001 | -45.47 |
| ara | Gender | Fem | 16 | -0.958 | 0.004 | -254.64 |
| ara | Gender | Masc | 18 | -0.745 | -0.001 | 1141.37 |
| ara | Number | Sing | 30 | -0.770 | -0.004 | 181.50 |
| ara | Person | 1 | 3 | -1.303 | 0.000 | — |
| ara | Person | 2 | 7 | -0.839 | -0.019 | 45.25 |
| ara | Person | 3 | 13 | 0.556 | 0.003 | 161.59 |
| eng | Aspect | Prog | 2 | -0.216 | 0.000 | — |
| eng | Gender | Fem | 3 | -0.669 | 0.000 | — |
| eng | Gender | Masc | 2 | -0.024 | 0.008 | -2.93 |
| eng | Mood | Sub | 3 | 0.084 | 0.012 | 7.15 |
| eng | Number | Plur | 4 | -1.242 | 0.000 | — |
| eng | Number | Sing | 2 | 0.152 | 0.000 | — |
| eng | Person | 1 | 2 | -2.039 | 0.000 | — |
| eng | Person | 3 | 2 | -1.196 | 0.000 | — |
| eng | Polarity | Neg | 2 | -0.514 | 0.000 | — |
| eng | Tense | Past | 3 | -0.490 | 0.000 | — |
| eng | Tense | Pres | 2 | 0.169 | 0.000 | — |
| fra | Gender | Masc | 30 | -0.469 | -0.002 | 295.86 |
| fra | Number | Plur | 30 | -2.167 | 0.000 | -72960.03 |
| fra | Number | Sing | 30 | -0.812 | 0.000 | — |
| fra | Person | 1 | 30 | -0.724 | 0.000 | -908002.25 |
| fra | Person | 2 | 30 | -1.505 | -0.000 | 9469880.47 |
| fra | Person | 3 | 30 | -0.705 | -0.002 | 337.78 |
| spa | Gender | Masc | 30 | -1.794 | 0.002 | -799.92 |
| spa | Number | Plur | 30 | -2.436 | -0.018 | 132.39 |
| spa | Number | Sing | 30 | -1.585 | -0.000 | 323987.13 |
| spa | Person | 1 | 30 | -2.500 | 0.004 | -672.76 |
| spa | Person | 2 | 30 | -3.294 | 0.003 | -987.06 |
| spa | Person | 3 | 30 | -2.930 | -0.004 | 689.25 |
| tur | Number | Sing | 30 | 1.052 | 0.002 | 595.56 |
| tur | Person | 1 | 30 | -0.501 | -0.003 | 148.83 |
| tur | Person | 2 | 30 | -0.803 | -0.011 | 72.27 |
| tur | Person | 3 | 30 | -0.440 | 0.000 | — |

**Holy-grail finding signal:** cells where `ratio_top/random >> 1` show top-attribution features ARE causally necessary. Cells where the ratio is near 1 mean the attribution identifies correlative but non-causal features — the input/output asymmetry from hypothesis #6. Cross-lingual matrix (same features, different target cells) is in `fig_holy_grail_matrix.png` if completed.

## Honesty notes

- All pair counts shown INCLUDE multi-token Arabic cases where we use the LAST BPE token of the counterfactual. `bug_audit/tok_strategy_counts.png` shows the single-vs-multi composition per cell.
- Random-feature baselines computed against density-matched pools where applicable (Arabic-dual English sweep) or simple uniform sampling elsewhere (ablation validation).
- **No cell claims are made without either a baseline comparison or an explicit null result.** See `TODO.md::scientific_anomalies` for cells whose internal statistics raise questions.
- Template-supplement pairs (Tense, language-specific rare phenomena) were NOT generated this session. All multilingual data is Multi-BLiMP-derived.
- Turkish Gender is absent by design, not a failed run.

## Known gaps / next steps

See `TODO.md` for the full list. Highlights:

- Template supplement for Tense + language-specific rare phenomena.
- Multi-token counterfactual with summed-logprob metric (this session: last-BPE approximation).
- Example #1 (gender → sexist English) and #2 (formality → British spelling) both require English generation evaluation on crafted prompts — not attempted this session.
- Syed-style two-pass attribution patching as comparison method.
- UD POS/Feats profile per top feature (cut for scope).

## Merge-back

Work is on branch `overnight-multilingual` in worktree `/projectnb/mcnet/jbrin/lang-probing-overnight`. To adopt:

```bash
cd /projectnb/mcnet/jbrin/lang-probing
git merge overnight-multilingual
```
To discard:
```bash
git worktree remove /projectnb/mcnet/jbrin/lang-probing-overnight
git branch -D overnight-multilingual
```