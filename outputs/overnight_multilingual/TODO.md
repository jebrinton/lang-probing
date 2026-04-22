# Overnight TODOs and oddities

Session: 2026-04-22 overnight multilingual counterfactual attribution run.

## Blockers

_(none yet — populated if a wave fails)_

## Silent-failure / wrong-output (fixed this session)

- [ ] **Heaviside gate non-differentiable** in `src/lang_probing_src/activations/sae.py` (`GatedAutoEncoder.encode`, formerly at `src/lang_probing_src/autoencoder.py:65`). `(pi_gate > 0).float()` produces no grad_fn; grad does NOT flow through the gate, only the magnitude ReLU. Attribution rankings therefore miss gate-driven effects and over-weight magnitude-driven features. **Fix this session:** isolated `sae_encode_ste` helper in the attribution script with sigmoid surrogate backward pass; do NOT modify the autoencoder (would regress ablate.py).
- [ ] **Counterfactual position hardcoded to last token** at `scripts/counterfactual_attribution.py:151`. Breaks mid-sentence counterfactuals (most of Multi-BLiMP). **Fix this session:** `cf_position_idx` field per-pair; `logits[:, cf_pos, :]`.
- [ ] **Grad aggregation contaminated by early-token signal** at `scripts/counterfactual_attribution.py:395-398`. `grad_sum` over all positions mixes pre-concept noise with the concept-bearing position. **Fix this session:** default to `grad[cf_pos]`; optional windowed sum behind flag.
- [ ] **No per-value signed aggregation**, blocking sign-flip analysis (`#5`). **Fix this session:** separate `(lang, concept, value)` cells with signed + abs + signed_gxa tensors.

## Silent-failure (NOT addressed this session; main-repo concern)

- [ ] Main repo `scripts/analyze_tokens.py` expects probe filename `probe_layer{layer}_n{n}` but `word_probes.py` writes `l{layer}_n{n}`. Probe filtering in token-analysis silently disabled. Flagged; not my scope tonight.
- [ ] Main repo `scripts/attribution_flores.py:26` has bad import `from src.config` (should be `lang_probing_src.config`). Flagged; not my scope tonight.

## Scientific anomalies — investigate

_(filled during run)_

## Scientific — elsewhere (follow-up sessions)

- [ ] **Example `#1` — gender → sexist English.** Requires English generation eval on crafted prompts ("The doctor said ___", "I'll order takeout at ___"). Top Romance-gender features → ablate during English generation → measure distributional shift on gendered pronoun / stereotype tokens.
- [ ] **Example `#2` — formality → British spelling.** Requires English minimal pairs on spelling ("color/colour", "realize/realise"). Top Arabic/German formality features → check if ablation shifts logprob on British variants.
- [ ] **Multi-token counterfactual with summed-logprob metric.** Current session uses LAST BPE token only. Extension: sum logprob over full multi-token counterfactual span, backprop the sum. Care needed on which positions contribute to the loss.
- [ ] **UD POS/Feats profile per feature.** Cut from this session for scope. For each top feature, tabulate UPOS tags + morphological feature values of its top-1% activating tokens across UD-PADT (ar), UD-GSD (fr/es), UD-BOUN (tr). Reveals whether a feature is generic verbal vs. specifically past-tense morpheme.
- [ ] **Syed-style two-pass attribution patching** as a comparison method. Current session uses single-pass grad×act (zero-baseline indirect effect). Paper: arxiv.org/abs/2602.16080.
- [ ] **Aruna's attention-head identity ablation** on MT prompts. No contrastive pair required; rank heads by logprob drop on r_0 after setting h_i to identity. For a session where contrastives are unavailable (true MT task).

## Nice-to-have

- [ ] Cross-lingual cross-concept: features multi-concept in >1 language.
- [ ] Cross-lingual co-activation graph (union across all 5 langs).
- [ ] Multi-BLiMP coverage expansion for concepts not present: Turkish Gender, French past-participle gender agreement with preceding direct object.

## Cross-reference questions (unresolved)

- [ ] **STE change magnitude.** Bug-audit `before_after_ste.png` will show how much ranking shifts. If >50% of top-50 features change in any cell, the STE fix is consequential enough to upstream into `autoencoder.py` in a follow-up session.
- [ ] **Per-cell metric<0 fraction.** If a cell has metric<0 (model prefers counterfactual) for >30% of pairs, either the pair labeling is swapped or the concept is not well-modeled. Flag in REPORT.
- [ ] **Arabic multi-token skip/last-token rate.** Will appear in `data_snapshot.json`. If >50%, affects H4 evidence strength; qualify claims accordingly.
- [ ] **Layer 31 vs layer 32 probe question** (pre-existing main-repo open item per their own TODO). Does not affect this session's layer-16 SAE work.
