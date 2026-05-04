# GCM Translation Attribution — Scientific Report

This is the scientific complement to the [README](README.md) (which covers
how to run things). This report explains *what we're doing and why*, the
methodology in detail, sanity-check results, and the eventual findings, with
attached figures. Keep this around as a reference for future you and for
colleagues who want to understand the experiment cold.

**Status:** v0.1 — implementation complete, sanity checks pending the
overnight sweep. Findings sections are placeholders; they'll be filled in
when results land.

---

## 1. Executive summary

We apply Generative Causal Mediation (GCM, Sankaranarayanan et al.,
arXiv:2602.16080) — a gradient-based estimator of the indirect effect of
patching one model component — to FLORES translation pairs. For each pair
of (source sentence, counterfactual source sentence), we ask: *which
attention head, and which sparse-autoencoder feature at layer 16, most
strongly distinguishes the model's preference between the two gold target
translations?* We sweep eng/spa/deu/fra/tur/ara/hin/heb in all 56 ordered
directions on Llama-3.1-8B + a layer-16 gated SAE, with 100 pairs per
direction. The output is a per-component IE ranking per direction, plus
cross-direction "universal translation heads" / "universal translation
features" candidates.

This is the first experiment in this repo to localize translation
computations at the **head level** (vs. the SAE-only granularity used in
[counterfactual_attribution](../counterfactual_attribution/)), and the
first to apply GCM specifically.

---

## 2. Background

### 2.1 The thesis this experiment serves

The broader research arc (see top-level [README](../../README.md) and
[LEDGER](../../LEDGER.md)) hypothesizes that LLMs translate by reusing the
same monolingual circuits they already use for ordinary language modeling,
mediated through a multilingual "semantic hub" of shared grammatical
features. The four sub-hypotheses are:

- **H1** — BLEU is explained by source + target language-modeling
  competence with no language-pair-specific interaction.
- **H2** — Input and output feature spaces overlap across languages
  (the "noisy channel is multilingual").
- **H3** — Adding a language ≈ improving monolingual capability.
- **H4** — Translation uses the same monolingual circuits.

This experiment is **direct evidence for H4** at the head level. Until now,
the H4 evidence in the repo was at the SAE-feature level — informative,
but coarse-grained: a single feature can be active for many computations,
and SAE feature activity is mostly correlational. By targeting attention
heads (the model's actual *computational primitives*), and by using a
**causal** gradient estimator rather than activation correlations, we get
a sharper test: if the same heads matter across many language directions,
that's strong evidence for shared circuitry rather than coincidental
feature firing.

### 2.2 What GCM is, in one paragraph

For a contrastive setup `(p_orig, r_orig, r_cf)` — one prompt, two
candidate responses — the *indirect effect* of a model component
$z$ is how much patching $z$ from one input to another changes the
metric

$$M(z) = \log \pi(r_{\text{cf}} \mid p_{\text{orig}}, z) - \log \pi(r_{\text{orig}} \mid p_{\text{orig}}, z).$$

Activation patching (ACP) measures this directly by running the model with
$z$ swapped — one forward pass per component, expensive at scale. GCM's
key contribution is that under a 1st-order Taylor expansion of $M$ in $z$,
the indirect effect of swapping $z_{\text{orig}} \to z_{\text{cf}}$ is

$$\widehat{\mathrm{IE}}(z) = \nabla_z M \big|_{z = z_{\text{orig}}} \cdot (z_{\text{orig}} - z_{\text{cf}}),$$

which can be computed for *every* component simultaneously with one
forward + one backward pass. The paper validates this approximation
empirically and we replicate that check (see §5.1).

### 2.3 Why translation as the application

GCM's natural setting is contrastive prompts where someone wrote both
responses by hand (e.g. "talk in prose" vs "talk in verse"). Translation
gives us something better: thousands of *real, naturally-occurring*
contrastive pairs from FLORES, where both responses are gold translations
of real sentences and the contrast varies along whatever axis FLORES
sentences vary along (topic, register, syntax, lexis). This is a richer
signal than hand-written contrasts and is directly aligned with the
research question — "what computations distinguish translation A from
translation B?"

---

## 3. Method

### 3.1 The metric and our sign convention

Both responses are scored against the **same prompt** $p_{\text{orig}}$:

$$M(z) = \underbrace{\log \pi(r_{\text{cf}} \mid p_{\text{orig}}, z)}_{m_{\text{cf}}} - \underbrace{\log \pi(r_{\text{orig}} \mid p_{\text{orig}}, z)}_{m_{\text{orig}}},$$

where each $\log \pi(r \mid p)$ is the joint log-probability of the
response sequence under teacher-forcing (sum of per-token log-probs).
$z_{\text{cf}}$ is cached separately by running the model on
$p_{\text{cf}}$ — i.e. the counterfactual activation at the same
component-position is the value that component would have had if the
model had seen the cf source sentence instead.

Sign: positive $\widehat{\mathrm{IE}}$ means *moving from $z_{\text{cf}}$
to $z_{\text{orig}}$ at this component increases the model's preference
for $r_{\text{cf}}$ over $r_{\text{orig}}$*. So $|\mathrm{IE}|$ measures
component importance for distinguishing the two translations, and the
sign tells you which side this component favors when in its orig state.

### 3.2 What counts as a "component"

Two flavors, both attributed in every run:

**(a) Per-attention-head outputs.** For each layer $L$ and each query head
$h$ in Llama-3.1-8B (32 layers × 32 q-heads = 1024 components per
direction), we treat the slice of `o_proj.input[:, last_src_idx, head*head_dim : (head+1)*head_dim]`
as the component. Note GQA: Llama-3.1-8B has 32 q-heads but only 8
kv-heads, so q-heads come in groups of 4 sharing a kv-cache — expect
results to cluster in 4s.

**(b) Layer-16 SAE features.** A 32 768-feature gated autoencoder trained
on the L16 residual stream of this model
(`jbrinkma/sae-llama-3-8b-layer16`). Provides a sparser, more
interpretable basis to compare against the SAE-level results from the
rest of the repo.

### 3.3 Patching position

We patch at **the last source token only** — the position right before
the model has to start producing the target translation. For our prompt
template that's the trailing-space token after `Spanish:` (or the
space-prefixed first translation token, depending on the target script).

This is a deliberate departure from the GCM paper's recipe (which patches
at all source positions and sums). Two reasons:

1. **Interpretability.** A single anchor position makes the IE
   comparable across pairs and across language directions. Patching at
   varying numbers of source positions (sentences differ in length)
   would mix in a confound.
2. **The semantic-hub hypothesis says the source-side computation has
   already condensed by this position.** If the noisy-channel/semantic-hub
   picture is right, the "translation-relevant" content of the source
   sentence has already been summarized at the last source token by the
   time the target side begins. Patching there is the natural test of
   that hypothesis.

A follow-up worth doing: re-run with the full GCM "all source positions"
patching to see how much signal we're leaving on the table. See §6.

### 3.4 Prompt template — and the first-response-token detail

The 2-shot prompt ends with a literal trailing space:

```
English: <shot1_src>
Spanish: <shot1_tgt>
English: <shot2_src>
Spanish: <shot2_tgt>
English: <query_src>
Spanish: 
```

Single-newline separators throughout. Two FLORES rows (0 and 1) are held
out from the pair-sampling pool to serve as fixed shots.

The metric sums log-probs over **every** token of the gold target
translation, including the first. This is correct for our template:

- For Latin scripts (es/de/fr/tr) Llama-3 BPE merges the trailing-space
  with the next word, so the first response token is a single
  space-prefixed-word like `" Hola"`. Scoring it = predicting
  "what's the first word of the Spanish translation given the prompt
  ends with `Spanish: `?"
- For Arabic/Hebrew/Hindi/CJK, BPE doesn't produce space-prefixed
  language tokens, so the trailing space ends up as its own token at the
  prompt boundary, and the first response token is the bare first
  language token. Scoring it = predicting "what's the first character of
  the Arabic translation given everything before?"

In both cases the first response token carries real information and
should be scored. (The GCM paper's reference implementation skips it
because they use chat-template tokenization where the first response
token is a deterministic newline; we don't have that artifact.)

### 3.5 The two-trace gradient pattern

A naive implementation would put both response-scoring passes
(`prompt_orig + r_orig` and `prompt_orig + r_cf`) into one nnsight
`model.trace()` block via two `tracer.invoke(...)` calls. We don't,
because nnsight 0.5 batches multi-invoke into a single forward pass
using `tokenizer.pad` with `padding_side="left"` — and our two
sequences differ in length by however much the responses differ. The
left-padding shifts the patched index for the shorter sequence, so the
intervention lands on a padding token instead of the intended last source
token. Silently wrong.

Instead we run **two separate single-invoke `model.trace()` blocks**, one
per response. Inside each, we use the canonical nnsight 0.5 backward
pattern:

```python
with model.trace(input_ids, **TRACER_KWARGS) as tracer:
    # ... patch z_leaf at last_src_idx ...
    m = sum_response_logprobs(...)
    m_value = m.save()
    with m.backward():
        grad_proxy = z_leaf.grad.save()
```

PyTorch accumulates gradients on `z_leaf` across traces. We zero
`z_leaf.grad` between the two traces and capture each separately, then
combine: $\nabla M = \nabla m_{\text{cf}} - \nabla m_{\text{orig}}$.

### 3.6 The heads patching mechanism

Writing to `module.input` is unsupported in nnsight ("the `.input`
property returns the first positional argument and is read-only" — per
the official agent guide). To patch a per-head slice of `o_proj`'s
input, we exploit the fact that `o_proj` is linear: replacing a slice
of the input from $z_{\text{orig}}$ to $z_{\text{leaf}}$ produces an
*equivalent* delta in the output of $(z_{\text{leaf}} - z_{\text{orig}}) W_o^\top$,
which we then add to `o_proj.output` (which IS supported). The
mathematical effect is identical; the implementation is canonical.

### 3.7 Sanity checks built into the run

Four guardrails fire on every pair:

1. **`sanity_orig_drift`** — at the linearization anchor $z_{\text{leaf}} = z_{\text{orig}}$, the patched run should produce a metric identical (up to bf16 round-off) to the unpatched run. Drift > 0.5 nat means the patch identity is broken.
2. **Per-token-mean sign sanity** — clean (unpatched) `m_orig` should be greater than clean `m_cf` per-token, since the model should genuinely prefer the gold orig translation over a random other FLORES translation given the orig source. Length-bias-free comparison.
3. **`decoded_last_src_orig` / `decoded_last_src_cf`** — both should be the same character (the trailing space or the colon, depending on tokenization), confirming the patch position is consistent across orig and cf prompts.
4. **NaN-aware aggregation** — failed pairs are stored as NaN sentinels so all the per-pair tensor stacks remain index-aligned with `per_pair_records.json`.

Two more guardrails outside the per-pair loop:

5. **Finite-difference linearization test** ([test_gcm_sae_finite_difference_sanity](../../tests/test_gcm_translation.py)) — perturb $z_{\text{leaf}}$ by $\varepsilon \cdot \delta z$ and assert $M(z + \varepsilon \delta z) - M(z) \approx \varepsilon \cdot \nabla M \cdot \delta z$ on a single pair. Decisive faithfulness check for the per-component gradient.
6. **ACP vs ATP correlation** ([validate_faithfulness.py](validate_faithfulness.py)) — for top-K heads from ATP, run the full activation-patching ground truth (one forward per (head, pair) with the head's slice actually replaced by $z_{\text{cf}}$) and report Pearson + Spearman correlation. The point is to confirm that the ATP ranking ≈ the ACP ranking — i.e. that what gradients say matters is also what actually matters when you do the intervention.

### 3.8 Sweep parameters

| Parameter | Value |
|---|---|
| Source/target languages | English, Spanish, German, French, Turkish, Arabic, Hindi, Hebrew |
| Number of directions | 56 (8 × 7 ordered, excluding identity) |
| Pairs per direction | 100 |
| Pair-sampling seed | 42 (deterministic across directions) |
| Shot indices | FLORES rows 0 and 1, held out from pair pool |
| FLORES split | dev (~997 sentences) |
| Max response tokens | 128 (token-space truncation in `tokenize_pair`) |
| Model | meta-llama/Llama-3.1-8B (bf16) |
| SAE | jbrinkma/sae-llama-3-8b-layer16, 32 768 features, gated |
| GPU | L40S or A40, gpu_c ≥ 8.6, 32 GB |
| Per-pair compute | ~6 forward passes (caches) + 4 forward+backward (grad traces) ≈ 4 s |
| Per-direction wall-time | ~12 min |
| Sweep wall-time (4-task array) | ~3 h |

---

## 4. Statistical analysis plan

### 4.1 Per-direction

- **Top-20 heads** by `mean(|IE|)` across the 100 pairs, with std error.
- **Top-50 SAE features** by the same.
- **Sign-mean** plot at the head level — for each (layer, head), the mean signed IE across pairs. Direction-of-effect.
- **`sanity_orig_drift` distribution** — should be tight around 0; >0.5 nat in any pair indicates a per-pair patch-identity failure.

### 4.2 Cross-direction (after `analyze.py`)

- **Universal heads:** for each (layer, head), how many of the 56 directions have it in their top-20? A head appearing in ≥28 (half) is a candidate "universal translation head". Visualized as a heatmap on the (layer, head) grid.
- **Sign consistency of universal heads:** for each universal head, the per-direction mean signed IE — does it have a consistent sign or does it flip across language directions? Consistent sign ≈ "this head encodes 'how strongly is this the right translation,' regardless of language." Flipping sign ≈ "this head encodes a language-specific feature."
- **Universal SAE features:** same analysis at the feature level. Cross-reference the result with the existing universal features (f9539, f14366, f12731) from [counterfactual_attribution/aggregated_by_concept.json](../counterfactual_attribution/aggregated_by_concept.json) — overlap would be strong corroboration of a "multilingual semantic hub."
- **Layer distribution:** at what depths do universal vs language-specific heads cluster?

### 4.3 Faithfulness

- **Pearson r and Spearman ρ** between ATP IE and ACP IE on the top-20-by-ATP heads, sampled over 10 pairs, for at least eng→spa.
- **Top-K agreement:** of the top-20 heads by ATP, how many are in the top-20 by ACP? Order-statistics version of the correlation.

---

## 5. Findings

*Placeholder — populated when the sweep + ACP run completes.*

### 5.1 Sanity-check results
TBD. Expected patterns:
- Finite-difference test: relative error <5% (per [test_gcm_sae_finite_difference_sanity](../../tests/test_gcm_translation.py)).
- ACP vs ATP Pearson r > 0.9 on eng→spa.
- `sanity_orig_drift` < 0.05 nat per pair (just bf16 round-off).
- Per-token-mean orig-preferred > 0.95 of pairs.

### 5.2 Per-direction head results
TBD. Will include the top-20 head table and a representative head-IE
heatmap (e.g. eng→spa) showing what a typical attention map of GCM IE
looks like.

### 5.3 Per-direction SAE feature results
TBD. Will include the top-50 feature bar chart, color-coded by sign of
mean IE, with annotations for any feature that's also in the
`counterfactual_attribution` universal list.

### 5.4 Cross-direction "universal heads"
TBD. Will include the universal-heads heatmap and a discussion of which
layers / head clusters appear most often.

### 5.5 Translation-vs-monolingual circuit overlap
TBD. The follow-up cross-reference: are the heads / features that GCM
flags as important for translation also important in monolingual settings?
This is the H4-direct test. Probably needs another small experiment using
the same prompt template stripped of the target side.

---

## 6. Limitations and follow-ups

- **Last-source-token-only patching** — under-attributes if the
  translation-relevant signal is spread across the source sentence
  (which it probably is, especially for long sentences). The reference
  implementation patches at all source positions and sums. A follow-up
  pass with the full source-positions patching would let us quantify
  this.
- **2-shot prompt is a confound.** Heads attending to the shot examples
  may dominate IE for some pairs in unintended ways. We may want a
  no-shot or 1-shot variant once we have the 2-shot baseline.
- **ACP only on top-K from ATP** — a fully-faithful pairing would compute
  ACP on every (head, pair), but that's 56 directions × 100 pairs × 1024
  heads = 5.7M forward passes, prohibitive. The top-20 sample is the
  standard mitigation, but skews the correlation toward the
  high-magnitude end of the distribution.
- **GQA q-head clustering.** With 32 q-heads and 8 kv-heads, q-heads come
  in groups of 4 sharing a kv-cache; "different" q-heads in the same
  group may not be functionally distinct. Reporting should keep this in
  mind.
- **Single model / single SAE.** All conclusions are Llama-3.1-8B + L16
  SAE-specific. Generalizing requires repeating the same recipe on
  Aya-23-8B (already cached in this repo for some experiments) and
  multiple SAE layers.
- **Base model, not instruction-tuned.** Translation is being elicited
  via 2-shot prompting on a base model, which works but is noisier than
  using a chat-tuned model. Not a fundamental issue for the H4 test
  (which is about which circuits are reused, not translation quality).

---

## 7. Reproducibility

- **Code:** [experiments/gcm_translation/](.) — `gcm_core.py` (attribution), `run.py` (per-direction CLI), `flores_pairs.py` (sampling + prompt), `analyze.py` (cross-direction aggregation), `visualize.py` (plots), `validate_faithfulness.py` (ACP).
- **Tests:** [tests/test_gcm_translation.py](../../tests/test_gcm_translation.py), 15 tests (9 CPU + 6 GPU).
- **Red-team log:** [REDTEAM.md](REDTEAM.md).
- **Determinism:** seed 42, deterministic shot indices, deterministic pair sampling per (src, tgt).
- **Sweep submission:** `qsub -t 1-56 experiments/gcm_translation/run/run_gcm_sweep.sh`.

---

## 8. References

- Sankaranarayanan, A., Zur, A., Geiger, A., Hadfield-Menell, D.
  *Generative Causal Mediation* (arXiv:2602.16080).
- Reference implementation: [roonbug/gcm-interp](https://github.com/roonbug/gcm-interp) (cleanup branch).
- Llama-3.1-8B (Meta).
- jbrinkma/sae-llama-3-8b-layer16 (gated SAE on the L16 residual stream).
- FLORES-101 (Goyal et al., 2022).
- nnsight 0.5.12 — official guidance at https://github.com/ndif-team/nnsight/blob/main/CLAUDE.md and https://github.com/ndif-team/nnsight/blob/main/NNsight.md.

---

## 9. Figures

Figures live in [./img/](./img/) once the sweep completes. The intended
inventory is below; placeholders will become real `![](path)` markdown
links as figures arrive.

- **Per-direction heatmaps**
  `img/<src>__<tgt>_heads_heatmap.png` — mean-abs IE per (layer, head)
  `img/<src>__<tgt>_heads_signed_heatmap.png` — signed mean IE
  `img/<src>__<tgt>_top_heads_bar.png` — top-20 heads, error-barred
  `img/<src>__<tgt>_top_sae_bar.png` — top-50 SAE features, color by sign
- **Faithfulness**
  `img/<src>__<tgt>_acp_vs_atp.png` — scatter, Pearson r in title
- **Cross-direction**
  `img/universal_heads_heatmap.png` — directions-in-top-K per (layer, head)
- *(more, once we know what the data looks like)*
