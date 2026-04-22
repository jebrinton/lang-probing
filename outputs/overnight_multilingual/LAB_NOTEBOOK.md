# Overnight lab notebook — 2026-04-22

Chronological append-only log of the overnight session. Each entry: UTC-4 timestamp + wave + what ran + what surprised me.

---

## ~03:50  Wave 0 — worktree + skeleton docs

`git worktree add /projectnb/mcnet/jbrin/lang-probing-overnight -b overnight-multilingual HEAD` off commit `3a61624` (wave 2 of main-repo restructure: library scaffolded, flat shims kept, `scripts/*.py` still at old paths pre-wave-3). My work uses the OLD flat `scripts/` paths per plan.

Main repo has STAGED but uncommitted Wave 3 (moves `scripts/` → `experiments/`). My worktree branches off HEAD so I don't inherit those staged moves.

Copied `data/grammatical_pairs.json` from main repo (gitignored; not tracked). Created output dirs under `outputs/overnight_multilingual/`. Skeleton LEDGER, LAB_NOTEBOOK, TODO written.

Next: Wave 1 — apply bug fixes A1–A5 to `scripts/counterfactual_attribution.py`, smoke-test on 30 English pairs.
