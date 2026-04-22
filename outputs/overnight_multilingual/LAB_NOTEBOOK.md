# Overnight lab notebook — 2026-04-22

Chronological append-only log of the overnight session. Each entry: UTC-4 timestamp + wave + what ran + what surprised me.

---

## ~03:50  Wave 0 — worktree + skeleton docs

`git worktree add /projectnb/mcnet/jbrin/lang-probing-overnight -b overnight-multilingual HEAD` off commit `3a61624` (wave 2 of main-repo restructure: library scaffolded, flat shims kept, `scripts/*.py` still at old paths pre-wave-3). My work uses the OLD flat `scripts/` paths per plan.

Main repo has STAGED but uncommitted Wave 3 (moves `scripts/` → `experiments/`). My worktree branches off HEAD so I don't inherit those staged moves.

Copied `data/grammatical_pairs.json` from main repo (gitignored; not tracked). Created output dirs under `outputs/overnight_multilingual/`. Skeleton LEDGER, LAB_NOTEBOOK, TODO written.

## ~03:55  Wave 1 — attribution rewrite

Read the full `scripts/counterfactual_attribution.py` and discovered A3 (Heaviside STE fix) is **moot** — the existing code already bypasses `encode()` in the gradient pass by wrapping encode in `torch.no_grad()` and treating `z = f_saved.detach().clone().requires_grad_(True)` as a leaf variable. Gradients only flow through `decode()` onward. Non-differentiable gate is irrelevant. Dropped A3 from plan; documented in TODO.

Wrote `scripts/attribute_multilingual.py` as a parallel script (leaves the English prototype untouched). Implements:
- A1 (multi-token last-BPE strategy): for multi-token counterfactuals, feeds `prefix + orig_toks[:-1]` and compares logP of last orig vs last cf tokens at `logits[-1]`.
- A2 (simplify aggregation): drops the unprincipled sum-over-all-positions secondary ranking. Only `grad[cf_pos]` (always `-1` after input construction).
- A4 (per-value signed aggregation): groups pairs by `(lang, concept, value)`; saves `aggregated_signed.pt`, `aggregated_abs.pt`, `aggregated_signed_gxa.pt`, `aggregated_abs_gxa.pt` per cell.
- A5 (sanity logging): records metric distribution, fraction < 0, token-strategy counts in `summary.json`.
- 20% holdout per cell (for Wave 6 ablation validation).

Skips degenerate multi-token pairs where orig and cf happen to share the same last BPE token (rare).

Imports verified on login node with `conda activate probes`.

## ~03:58  Wave 2 — multilingual pair build

Ran `scripts/build_multiblimp_pairs.py`. Results:

| lang | n_pairs | cells (concept|value counts) |
|---|---|---|
| fra | 2212 | Number\|Sing (500), Number\|Plur (500), Gender\|Masc (217), Gender\|Fem (200), Person\|1/2/3 |
| spa | 2165 | similar to fra, but SP-G has 185 pairs so Gender better represented |
| tur | 1556 | Number + Person only (no Gender in Turkish, as expected) |
| ara | 1137 | Dual\|Dual 37, Dual\|Sing 209, Gender\|Fem 192, Gender\|Masc 110, Number\|Sing 202, Person |

Surprise: Arabic had ~350 rows with `prefix=None` — filtered silently. Flagged in TODO as a data-quality question ("why are some Multi-BLiMP Arabic prefixes null?").

**Tense** is not in Multi-BLiMP as expected. Template supplement deferred to a later session (scope cut under usage budget). TODO.

Saved `outputs/overnight_multilingual/data_snapshot.json` with per-cell counts.

## ~04:30  Wave 4-5 — analyses complete

- Bug-audit: most cells have metric_mean in [7, 13] (healthy — model strongly prefers orig over cf). Multi-token strategy dominates fra/spa/ara (95%+), all single-token for eng. neg_frac <0.15 everywhere, so data quality looks good.
- Cross-concept (E1): **Feature 9539 ranks top-50 in every cell of every language** (4/4 tur, 6/6 spa, 6/6 fra, 8/8 ara, 10/11 eng). Similar for f14366, f12731. Either a universal grammatical-prediction feature (strong H4 evidence) or a dead-to-SAE artifact. Flagged for max-activating-context investigation in a follow-up.
- Sign-flip (E3): fra vs ara Gender=Masc has 44% opposite-sign top-200 features. ara vs tur Number=Sing has 32%. Sign-flip effect is real.
- Arabic dual → English (E2): HONEST NULL. Target features' Cohen's d = -0.075 vs null 0.003. Arabic-dual attribution features do NOT selectively fire on English "two/both/pair" sentences. Negative result is important: ~weak~ evidence against simple cross-lingual feature reuse for dual-number.

## ~05:00  Wave 6-7 — ablation validation + input-vs-output

- Ablation validation (W6) went smoothly after nnsight-indexing fix (used multiplicative mask instead of direct index assignment).
- All 35 cells show strong top-feature causal effect: Δorig from -0.5 to -3.3 on top-20 ablation vs ~0 on random-20 features. Effect ratios 100–10^6×.
- **Turkish Number=Sing anomaly: +1.05 Δorig on ablation (opposite direction!)** Investigation: top-20 signed_gxa has mixed signs (sum +0.08) whereas spa/Number/Sing has uniformly-positive (sum +0.33). Not a clean sign-convention bug; may reflect Turkish agglutinative morphology's interaction with the SAE in ways we didn't anticipate. Flagged in TODO.
- fra/Person/{1,2} show effect ratios ~10^6 because random baseline was exactly 0. Need per-pair inspection to confirm this is real and not a sampling artifact. Flagged in TODO.

## ~05:25  Wave 10 — REPORT compiled

Report written via scripts/write_report.py. TL;DR highlights the universal f9539, Turkish anomaly, sign-flip, Arabic-dual null. Committed.

Dashboards (W9) and aux characterizations (W8) cut for budget. Aux angles partially subsumed: decoder logit-lens and max-activating tokens remain as follow-up items in TODO.

## ~04:00  Wave 3 — submit parallel attribution jobs

`run/run_attribute.sh`: 1 GPU L40S (gpu_c=8.9), 32G VRAM, h_rt=2:00:00, email on end/abort.

Submitted 5 jobs (all queued at submission):
- 4475490 attr_fra
- 4475491 attr_spa
- 4475492 attr_tur
- 4475493 attr_ara
- 4475494 attr_eng  (bug-fixed pipeline on existing 30 English pairs for baseline)

Each runs `scripts/attribute_multilingual.py` capped at 300 pairs/cell with 20% holdout.

Will check back in ~25min via /loop.

