# Lab notebook

**Append-only, chronological.** This file is the unfiltered daily log: what was done, what was tried, what surprised, what's unresolved. `LEDGER.md` is the curated view; this is the scratchpad.

Cross-reference experiments by folder name (e.g., "see LEDGER::ablation"). New entries go at the **top**; older entries scroll down.

---

## 2026-04-22 — restructure kickoff (Wave 0 + 1)

Started a multi-wave restructure of this repo. Plan at `~/.claude/plans/h2-is-about-the-idempotent-cascade.md`. Goal: library + thin experiments + curated ledger + chronological notebook + TODO.

### Inventory & reconnaissance

Ran three exploration passes:

1. **Experiments report.** Identified 10 experiments as first-class organizational units. The flat `scripts/` dir fragments them across 40+ files. New structure is `experiments/<name>/` with thin `run.py` + YAML configs + co-located `run/` job scripts.
2. **Code/output weirdness.** Beyond the INVENTORY.md catalog: duplicate `"zho_simpl"` key in `config.py:71,74` silently overwrites "Chinese (Simplified)" with "Chinese"; duplicate `import torch` in `src/lang_probing_src/ablate.py`; dead function `old_get_batch_positions_masks()` at `src/lang_probing_src/ablate.py:122-170`; 16+ hardcoded absolute paths that should use `config.*_DIR`; `collect_activations.py:37-42` silently swallows load errors and continues to a NameError; `collect_activations.py` uses `args` as a module-global pattern (works, but a smell).
3. **Scientific anomaly scan.** No NaN/Inf. Linear-model R² genuinely low (0.02 Llama, 0.10 Aya). Counterfactual per-concept sample sizes alarmingly small (Polarity=2, Aspect/Mood=3). Ablation mono-random baseline has 66.7% exact zeros — may indicate sparse sampling or probe-coverage issues. Probe accuracies cluster high (mean 96.68%, max 99.98%) — likely reflects lexical morphology more than deep grammar. Filed as TODOs.

### Layer 31 vs 32 — not a bug

Investigated the suspected layer-31/32 mismatch. Verdict: **naming artifact across two PyTorch indexing conventions**.

- Llama-3.1-8B has 32 transformer layers.
- `outputs.hidden_states` is a 33-element tuple: `hidden_states[0]` is the embedding layer, `hidden_states[32]` is the output of the final transformer block.
- `model.model.layers` is a 32-element `ModuleList` indexed 0..31. `model.model.layers[31]` is the final transformer block.
- So `hidden_states[32]` and `model.model.layers[31]` refer to the same tensor.
- Probes (named `_l32_*.joblib`) are trained on `hidden_states[32]`.
- Cached activations (partitioned `layer=31`) are collected from `model.model.layers[31]`.
- `src/lang_probing_src/ablate.py:130-133` explicitly clamps `probe_layer=32` → `min(32, 31) = 31`.

System is internally consistent. Action items (filed to TODO): remove the misleading `for layer in [32]:` in `scripts/collect_activations.py:112`, add a docstring comment to `ablate.py:130-133` explaining the convention.

One minor find: older `_l30_*` probes exist for English (created Nov 22 13:45-13:55, before the `_l32` convention settled at 14:55). No cached layer-30 activations exist, so these can only be stragglers from an earlier experiment. Low-priority cleanup.

### Skeleton docs written

- `LEDGER.md` — per-experiment curated view, 10 experiments + cross-notes.
- `LAB_NOTEBOOK.md` — this file.
- `TODO.md` — consolidated from Agent B (code), Agent C (scientific), and INVENTORY.md.
- `README.md` — updated as index (still to do).

### Unresolved / surprised by

- **Random ablation baseline has 66.7% exact zeros.** Expected near-zero mean with nonzero variance. Exact zeros suggest either no samples entered the ablation mask (probe never fired for that language/concept), or a bug where the "random" feature set collided with a no-op branch. To investigate in Wave 5 cleanup.
- **The linear-model R² for Llama (0.02) is weak enough that the H1 claim might need rephrasing.** Aya at 0.10 is better but still low. The `.tex` claim of "88% faithful rank-1 approximation" is unverified here (code lives elsewhere); reproducing it in this repo is a Wave 5 TODO.
- **Probe accuracies surprisingly high.** 96.68% mean is suspicious — word-level grammatical morphology may be decidable from surface lexical features alone. Would want to compare against shuffled-label baselines; filed as a stretch TODO.
