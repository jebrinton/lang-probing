# gcm_translation — red-team findings

Three independent agents reviewed `gcm_core.py`, `run.py`, `flores_pairs.py`, and the qsub scripts against (a) the GCM paper's Eq. 1, (b) the official nnsight 0.5 docs + the working reference patterns in `experiments/counterfactual_attribution/run.py` and `src/lang_probing_src/features/attribution.py`, and (c) general code correctness. Verified against the actual installed nnsight at `~/.conda/envs/probes/lib/python3.11/site-packages/nnsight/`.

## CRITICAL — must fix before re-submitting any job

### C1. `M.backward()` is called *outside* the `model.trace()` context — non-canonical nnsight pattern
- **Where:** `gcm_core.py:180, 286`
- **Evidence:** nnsight 0.5 monkey-patches `Tensor.backward` (`__init__.py:156`) to dispatch to a `BackwardsTracer` *only inside* a `with`-block. Outside the trace, it falls through to plain torch autograd, which depends on the saved tensors still chaining back to `z_leaf` after the interleaver `__exit__` cleans up. None of the working references in this repo do this. The canonical pattern is `value.backward()` *inside* a `tracer.invoke(...)` block, with `z_leaf.grad.save()` set up first.
- **Fix:** Move `M.backward()` inside the trace; capture grads via `.save()`:
  ```
  with model.trace(...) as tracer:
      with tracer.invoke(orig): ...; m_orig = ...
      with tracer.invoke(cf):   ...; m_cf = ...
      M = (m_cf - m_orig).save()
      z_leaf_grad_proxy = z_leaf.grad.save()
      M.backward()
  grad = z_leaf_grad_proxy
  ```

### C2. Multi-invoke + different sequence lengths → left-padding shifts `last_src_idx` and `response_start`
- **Where:** `gcm_core.py:163–177` (SAE), `gcm_core.py:268–283` (heads), and inside `sum_response_logprobs`
- **Evidence:** nnsight `LanguageModel._batch` (`language.py:271`) calls `tokenizer.pad`. The default `padding_side="left"` is set in both `_load_tokenizer` (`language.py:159`) AND in `lang_probing_src/utils.py:69`. The two invokes (`prompt_orig + r_orig` vs `prompt_orig + r_cf`) have different lengths, so the shorter sequence is left-padded with EOS. Indices computed pre-padding are wrong by `pad_amount` for the shorter row.
- **Fix:** Two options. Either (a) batch the two by pre-padding to a common length and feed `[2, S_max]` to a *single* `tracer.invoke`, or (b) drop multi-invoke and use two separate `model.trace()` blocks per pair (also fixes C1 — backward inside each, accumulate grads from `+m_cf` and `-m_orig`). Option (b) matches the working reference (`counterfactual_attribution/run.py`).

### C3. Writing to `o_proj.input` is undocumented and likely no-ops
- **Where:** `gcm_core.py:272, 280`
- **Evidence:** Surveying nnsight's "Setting" docs page and all working references in this repo, the supported intervention pattern is to write to `module.output` (full-tensor `[:]` assignment), never to `module.input`. The `attribution_patching` tutorial reads from `.input` but never writes to it. In nnsight 0.5, slice-assigning the input proxy probably modifies a local copy without intercepting the actual `o_proj` forward call. Result: the patch silently no-ops and `z_leaf.grad` ends up zero or `None` for the heads branch.
- **Fix:** Patch one position downstream. Two viable approaches:
  - **(easier) Patch `self_attn.output`:** compute the equivalent residual perturbation in Python from the leaf via `o_proj.weight`: `delta = (z_leaf - z_orig) @ o_proj.weight.T`, then `self_attn.output[0][:, last_src_idx, :] += delta`. Per-head decomposition is then `grad_h * delta_h.sum(-1)` exactly as before, but you operate on `self_attn.output` which IS supported.
  - **(harder) Run o_proj manually:** In Python, compute `patched_o_proj_output = z_leaf @ o_proj.weight.T`, then write `o_proj.output[:] = patched_o_proj_output`. Read `o_proj.weight` outside the trace.

### C4. Sub-slice index assignment to `submodule.output[0]` may not propagate
- **Where:** `gcm_core.py:166, 173`
- **Evidence:** Working reference uses *full-tensor in-place* `submodule.output[0][:] = reconstructed`. The new code does `submodule.output[0][:, last_src_idx, :] = patched`. Whether sub-slice setitem on the proxy actually triggers the swap on the underlying forward output is undocumented; it may silently no-op.
- **Fix:** Build the full replacement tensor and assign with `[:]`:
  ```
  out0 = submodule.output[0]
  new_out = out0.clone()
  new_out[:, last_src_idx, :] = patched
  submodule.output[0][:] = new_out
  ```

### C5. Leading-space response prefix corrupts the first scored token in non-Latin scripts
- **Where:** `run.py:93–94`
- **Evidence:** Empirical: for Llama-3 tokenizer, `' مرحبا'` (Arabic with leading space) tokenizes very differently from `'مرحبا'`. Same for Hebrew, Hindi, Chinese, Japanese. The prompt ends with `Spanish:` (no trailing space); we then prepend `" "` to the response. For Latin scripts (es/de/fr/tr), this is the right BPE convention. For Arabic/Hebrew/Hindi/Chinese this introduces a spurious leading-space token, biasing `m_orig_clean` and `m_cf_clean` by 0.5–5 nats *in a language-dependent way* — invalidating cross-language comparisons that are the whole point of the sweep.
- **Fix:** Terminate the prompt template with `"Spanish: "` (trailing space) and pass the bare FLORES text as the response (no leading space). Edit `flores_pairs.py:make_prompt` and the `response_orig`/`response_cf` construction in `run.py:93–94`. Then `last_src_idx` will be the trailing space, not the colon — reflect that in the smoke-test sanity assertion.

### C6. `truncate_response` decode→retokenize is not BPE-round-trip-safe
- **Where:** `run.py:78–82`
- **Evidence:** Empirical: Hebrew first-5-tokens decode includes a stranded `"\xfd"` byte that re-tokenizes to a different ID. Truncation can silently shift response tokens. Bug only fires when `len(ids) > max_response_tokens` (default 128) — invisible in 5-pair smoke testing because typical FLORES sentences fit. In the 100-pair production sweep, the rare long sentences will silently corrupt their pair's metric.
- **Fix:** Tokenize once with `add_special_tokens=False`, truncate IDs in token space, and concatenate IDs directly without round-tripping through text. Restructure `tokenize_pair` to accept already-tokenized response IDs.

## IMPORTANT — likely correctness or operational issues

### I1. SAE leaf is bf16, not float32 — gradients quietly noisy
- **Where:** `gcm_core.py:149`
- **Fix:** `z_leaf = z_orig.detach().clone().to(torch.float32).requires_grad_(True)` — mirror the heads branch (`gcm_core.py:265`).

### I2. Missing `**TRACER_KWARGS` on gradient-pass `model.trace()`
- **Where:** `gcm_core.py:162, 267`
- **Evidence:** `TRACER_KWARGS = {'scan': False, 'validate': False}`. Without these, nnsight runs a scan/validate phase that interacts badly with intervention writes. All working references pass `**TRACER_KWARGS`.
- **Fix:** `with model.trace(**TRACER_KWARGS) as tracer:`.

### I3. `sae_ies`/`head_ies`/`pair_records` can desync on partial pair failure
- **Where:** `run.py:102–137`
- **Evidence:** Single try/except wraps both attributions. If SAE succeeds and heads then raises, `sae_ies` has the entry but `head_ies` doesn't — anyone joining `sae_stack[i] ↔ head_stack[i] ↔ pair_records[i]` post-hoc is silently misaligned.
- **Fix:** Per-component try/except; on failure, append a NaN sentinel of the right shape so all three lists stay index-aligned.

### I4. No explicit GPU memory release between pairs → OOM risk
- **Where:** `run.py:90–137`
- **Evidence:** Each pair: 1 grad-enabled trace × 2 invokes × (Llama-3.1-8B + SAE encode/decode + n_layers Save proxies). Reference `compute_attribution` does only ONE grad pass; this is heavier and runs twice (SAE + heads) per pair. With FLORES-long sentences, ~32 GB starts to feel tight by pair 50.
- **Fix:** `del z_leaf, grad, delta, ie, M; torch.cuda.empty_cache()` at end of each pair iteration.

### I5. Sign-sanity comparison biased by response length
- **Where:** `run.py:216–217`
- **Evidence:** `m_orig_clean` and `m_cf_clean` are sums of per-token logps over responses of *different* lengths. The longer response always has lower (more negative) sum. So `frac_orig_preferred` flips on length, not on quality.
- **Fix:** Compare per-token mean logp, not sum, for the sanity stat. (Keep sum for the IE metric; only the sanity stat needs normalizing.)

### I6. `model.config.num_attention_heads` may fail on nnsight LanguageModel
- **Where:** `gcm_core.py:231–233`
- **Evidence:** Other code in this repo uses `model.model.config.num_hidden_layers` (e.g. `extraction.py:316–317`), suggesting `model.config` forwarding is unreliable in some nnsight versions.
- **Fix:** Use `model.model.config.num_attention_heads` and `.hidden_size` for parity.

## MINOR — sanity and quality

### M1. Assert decoded last-source-token is `:` (or trailing space, post-C5 fix)
- **Where:** `gcm_core.py` end of `tokenize_pair`
- **Fix:** Add `assert pr_or.decoded_last_src in (":", " ")` and `assert pr_or.decoded_last_src == pr_cf.decoded_last_src` to catch any tokenization drift.

### M2. Hoist clean-metric forward passes out of the core functions (compute waste)
- **Where:** `gcm_core.py:140–145, 257–262`
- **Evidence:** With `--components both`, `m_orig_clean`/`m_cf_clean` are computed twice per pair. ~20% wasted forward passes.
- **Fix:** Compute once at the run.py level and pass in.

### M3. GQA: 32 "heads" are q-heads pre-o_proj concat, grouped in 4s by kv-head
- **Where:** `gcm_core.py:232`
- **Fix:** Document this in the analysis output. The reshape itself is correct (o_proj input has shape `[B, S, n_q_heads * head_dim]`).

### M4. No per-layer partial-gradient detection
- **Where:** `gcm_core.py:288–292`
- **Fix:** After backward, assert `(z_leaf.grad.view(n_layers, -1).abs().sum(-1) > 0).all()` to catch silent partial gradient flow.

### M5. `make_prompt` produces double-newline gaps between shots — confirm intent
- **Where:** `flores_pairs.py:104–108`
- **Fix:** Verify the gap is intentional or switch to single-newline join.

## Action plan

The smoke test (job 4650002, currently queued) will fail at C1 / C3, or run silently with C2/C4 corruption. Fix the **C1–C6** items before resubmitting any job. The **I1–I6** items can be batched in the same patch but are not show-stoppers. Add a **proper unit test**: 1 pair, assert `z_leaf.grad` is non-None, finite, and that small finite-difference perturbations of `z_leaf` change `M` by ≈ `grad · perturbation` (paper's linearization sanity).

---

## Update — comparison against `roonbug/gcm-interp` (cleanup branch)

Cross-referenced our implementation against an existing GCM reference (the GCM paper's collaborator's working repo). Findings:

**Confirmed equivalent or correct:**
- Score both responses against the same `p_orig` (delta_z from cf prompt's last_src position) — matches reference design.
- nnsight 0.5 `with M.backward():` pattern is the correct upgrade of the reference's nnsight 0.4 outside-trace `L.backward()`.
- Two-traces-with-grad-accumulation is mathematically equivalent to one-trace-with-summed-metric (linearity of gradient).
- GQA per-q-head reshape `[n_layers, n_q_heads, head_dim]` matches their per-`head_dim` slicing.
- Our fp32 leaf is numerically cleaner than their bf16 leaf.

**Intentional departures (documented in `gcm_core.py` module docstring):**
1. **Last-source-token patching only.** Reference patches at all source positions and sums grad·delta over them. We patch at the last source token only — per the original task spec. Effect: under-attribution if signal is spread across the source. May want a follow-up sweep over more positions.
2. **Score the first response token.** Reference skips it (chat-template artifact); our prompt template ends with `"Spanish: "` so the first response token is real content and should be scored.
3. **Heads patched at `o_proj.output`** via the equivalent delta `(z_leaf - z_orig) @ W_o.T`, since writing to `o_proj.input` is unsupported in nnsight. Reference uses `o_proj.output.retain_grad()` directly — different mechanism, same target.

**Missing — desirable follow-ups, not blockers for tonight:**
- **ACP (full activation patching) faithfulness check** on top-K heads from ATP. Reference includes this; we don't. Adding it as a post-hoc check on the top-20 heads would strengthen the claim that the gradient ranking is faithful to the actual interventional ranking.
- **`atp-zero` ablation variant** (z_cf = 0 baseline) — useful as a "remove this feature" baseline.

The finite-difference linearization sanity test (`test_gcm_sae_finite_difference_sanity` in `tests/test_gcm_translation.py`) already provides the most decisive faithfulness check on a per-pair basis.
