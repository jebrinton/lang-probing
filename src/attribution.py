import torch

from src.sparse_activations import SparseActivation

TRACER_KWARGS = {}


def attribution_patching_per_token(
    clean_prefix,
    model,
    probe,
    probe_submodule,
    submodules,
    dictionaries,
    steps=10,
):
    """
    For a single sentence (clean_prefix), compute per-feature attribution answering:
    "How much does this feature make the probe say TRUE at this same token only?",
    averaged over all tokens for which the probe predicts TRUE.

    Output effects[submodule].act has shape [n_sae_features] (per sentence).
    """

    # Ensure batch dim and move to cuda
    clean_prefix = torch.cat([clean_prefix], dim=0).to("cuda")

    # 1. Detect whether each submodule's output is a tuple, and record original dtypes
    is_tuple = {}
    orig_dtype = {}
    with model.trace(clean_prefix, **TRACER_KWARGS):
        for submodule in submodules:
            out = submodule.output
            is_tuple[submodule] = isinstance(out, tuple)
            if is_tuple[submodule]:
                orig_dtype[submodule] = out[0].dtype
            else:
                orig_dtype[submodule] = out.dtype

    # 2. Get clean dictionary activations + residuals for each submodule
    hidden_states_clean = {}
    with model.trace(clean_prefix, **TRACER_KWARGS), torch.no_grad():
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.output
            if is_tuple[submodule]:
                x = x[0]  # [batch, seq, d_model] or similar

            f = dictionary.encode(x)
            x_hat = dictionary.decode(f)
            residual = x - x_hat

            hidden_states_clean[submodule] = SparseActivation(
                act=f.save(),
                res=residual.save(),
            )

    hidden_states_clean = {k: v for k, v in hidden_states_clean.items()}

    # 3. Define zero (ablated) patch state
    hidden_states_patch = {
        k: SparseActivation(
            act=torch.zeros_like(v.act),
            res=torch.zeros_like(v.res),
        )
        for k, v in hidden_states_clean.items()
    }

    # 4. Clean forward pass through probe_submodule to find TRUE tokens
    with model.trace(clean_prefix, **TRACER_KWARGS), torch.no_grad():
        acts = probe_submodule.output
        if isinstance(acts, tuple):
            acts = acts[0]  # [batch, seq, d_model]

        bsz, seqlen, d_model = acts.shape
        assert bsz == 1, "This function currently assumes batch_size == 1."

        acts_flat = acts.reshape(-1, d_model)          # [bsz*seq, d_model]
        logits_flat = probe(acts_flat.float())         # [bsz*seq] or [bsz*seq, 1]
        if logits_flat.dim() == 2 and logits_flat.shape[1] == 1:
            logits_flat = logits_flat.squeeze(-1)

        logits = logits_flat.view(bsz, seqlen)         # [1, seq_len]

        # Indices of tokens where probe predicts TRUE (logit > 0)
        true_token_indices = (logits[0] > 0).nonzero(as_tuple=False).view(-1).tolist().save()

    if len(true_token_indices) == 0:
        # No TRUE tokens: return zero vectors per submodule
        effects = {}
        deltas = {}
        grads = {}
        total_effect = None
        for submodule in submodules:
            n_features = hidden_states_clean[submodule].act.shape[-1]
            effects[submodule] = SparseActivation(
                act=torch.zeros(n_features, device=clean_prefix.device),
                res=torch.zeros(hidden_states_clean[submodule].res.shape[-1], device=clean_prefix.device),
            )
        return effects, deltas, grads, total_effect

    # 5. Metric function: logit at a single target token
    def metric_fn_single_token(model, submodule, probe, token_idx: int):
        acts = submodule.output
        if isinstance(acts, tuple):
            acts = acts[0]  # [batch, seq, d_model]

        bsz, seqlen, d_model = acts.shape
        acts_flat = acts.reshape(-1, d_model)          # [bsz*seq, d_model]
        logits_flat = probe(acts_flat.float())         # [bsz*seq] or [bsz*seq, 1]
        if logits_flat.dim() == 2 and logits_flat.shape[1] == 1:
            logits_flat = logits_flat.squeeze(-1)
        logits = logits_flat.view(bsz, seqlen)         # [bsz, seq]

        # Metric = logit at this specific token
        return logits[:, token_idx].mean()             # scalar

    # 6. Aggregate per-token, same-position feature effects
    sentence_effects = {submodule: None for submodule in submodules}
    deltas = {}  # we'll just keep the last token's delta per submodule
    grads = {}   # and the last token's grad per submodule
    total_effect = None  # optional; not very meaningful here

    for token_idx in true_token_indices:
        # For each TRUE token, run IG and extract effects at that token only
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            clean_state = hidden_states_clean[submodule]
            patch_state = hidden_states_patch[submodule]

            grad_acts = []
            grad_ress = []

            for step in range(steps):
                alpha = step / steps

                # Linear path: clean -> patch (ablated)
                f = (1 - alpha) * clean_state + alpha * patch_state

                f.act.requires_grad_(True)
                f.res.requires_grad_(True)
                f.act.retain_grad()
                f.res.retain_grad()

                with model.trace(clean_prefix, **TRACER_KWARGS):
                    # Patch this submodule's output for this run
                    patched = dictionary.decode(f.act) + f.res
                    patched = patched.to(orig_dtype[submodule])

                    if is_tuple[submodule]:
                        submodule.output[0][:] = patched
                    else:
                        submodule.output = patched

                    # Metric: probe logit at *this* token only
                    value = metric_fn_single_token(
                        model,
                        probe_submodule,
                        probe,
                        token_idx,
                    ).save()

                value.backward(retain_graph=True)

                if f.act.grad is None or f.res.grad is None:
                    raise RuntimeError(
                        "No grad for f.act or f.res; they are not in the computation graph."
                    )

                grad_acts.append(f.act.grad.detach().clone())
                grad_ress.append(f.res.grad.detach().clone())

                # Clear grads for next step
                f.act.grad = None
                f.res.grad = None

            # Average grads over steps (IG)
            mean_grad_act = torch.stack(grad_acts, dim=0).mean(dim=0)
            mean_grad_res = torch.stack(grad_ress, dim=0).mean(dim=0)
            grad = SparseActivation(act=mean_grad_act, res=mean_grad_res)

            # Change in representation between patch and clean
            delta = (patch_state - clean_state).detach() if patch_state is not None else -clean_state.detach()

            # Full effect tensor (token × feature)
            effect_full = grad @ delta  # shape like clean_state: [1, seq, n_features] or [seq, n_features]

            # --- extract same-token-only feature vectors ---
            eff_act = effect_full.act

            # We assume batch size 1; handle [1, seq, feat] or [seq, feat]
            print("eff_act.shape", eff_act.shape)
            if eff_act.dim() == 3:
                # [1, seq, n_features] -> [n_features] at this token
                eff_act_token = eff_act[0, token_idx, :]
            elif eff_act.dim() == 2:
                # [seq, n_features] -> [n_features]
                eff_act_token = eff_act[token_idx, :]
            else:
                raise ValueError(f"Unexpected effect.act shape: {eff_act.shape}")

            # Accumulate across TRUE tokens
            if sentence_effects[submodule] is None:
                sentence_effects[submodule] = SparseActivation(
                    act=eff_act_token.clone(),
                    res=None,
                )
            else:
                sentence_effects[submodule].act += eff_act_token

            # Keep last grad/delta if you care
            grads[submodule] = grad
            deltas[submodule] = delta

    # 7. Average over TRUE tokens so we get a single vector per sentence
    num_true = len(true_token_indices)
    print("num_true", num_true)
    effects = {}
    for submodule in submodules:
        effects[submodule] = SparseActivation(
            act=sentence_effects[submodule].act.unsqueeze(0) / num_true,
            res=None,
        )
    print("effects[sae_submodule].act.shape", effects[submodule].act.shape)

    # total_effect not very meaningful here; leave as None or sum of feature vectors if you want
    total_effect = None

    return effects, deltas, grads, total_effect


def attribution_patching_per_token_agg(
    clean_prefix,
    model,
    probe,
    probe_submodule,
    submodules,
    dictionaries,
    steps=10,
):
    """
    Attribution patching where the metric is the probe's logit aggregated over
    only those tokens where the probe predicts TRUE on a clean forward pass.

    clean_prefix: token ids for a single sentence (1D or [1, seq])
    model: NN with .trace context manager (e.g. NNsight-wrapped model)
    probe: torch version of the (binary) logistic regression probe,
           taking [N, d_model] -> [N] or [N, 1] logits
    probe_submodule: module whose activations are fed into the probe
    submodules: list of modules to attribute over
    dictionaries: dict {submodule -> autoencoder/dictionary}
    steps: number of interpolation steps for integrated gradients
    """

    # Ensure batch dim and move to cuda
    clean_prefix = torch.cat([clean_prefix], dim=0).to("cuda")

    # 1. Detect whether each submodule's output is a tuple, and record original dtypes
    is_tuple = {}
    orig_dtype = {}
    with model.trace(clean_prefix, **TRACER_KWARGS):
        for submodule in submodules:
            out = submodule.output
            is_tuple[submodule] = isinstance(out, tuple)
            if is_tuple[submodule]:
                orig_dtype[submodule] = out[0].dtype
            else:
                orig_dtype[submodule] = out.dtype

    # 2. Get clean dictionary activations + residuals for each submodule
    hidden_states_clean = {}
    with model.trace(clean_prefix, **TRACER_KWARGS), torch.no_grad():
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.output
            if is_tuple[submodule]:
                x = x[0]  # [batch, seq, d_model] or similar

            # x is in model's dtype (likely float16); encode/decode should handle that
            f = dictionary.encode(x)
            x_hat = dictionary.decode(f)
            residual = x - x_hat

            hidden_states_clean[submodule] = SparseActivation(
                act=f.save(),
                res=residual.save(),
            )

    hidden_states_clean = {k: v for k, v in hidden_states_clean.items()}

    # 3. Define zero (ablated) patch state, same dtype/shape as clean_state
    hidden_states_patch = {
        k: SparseActivation(
            act=torch.zeros_like(v.act),
            res=torch.zeros_like(v.res),
        )
        for k, v in hidden_states_clean.items()
    }

    # 4. Compute the TRUE-token mask from a clean forward pass through probe_submodule
    with model.trace(clean_prefix, **TRACER_KWARGS), torch.no_grad():
        acts = probe_submodule.output
        if isinstance(acts, tuple):
            acts = acts[0]  # [batch, seq, d_model]

        bsz, seqlen, d_model = acts.shape
        acts_flat = acts.reshape(-1, d_model)          # [bsz*seq, d_model]

        # Probe expects float32; convert but don't change model's computation graph here
        logits_flat = probe(acts_flat.float())         # [bsz*seq] or [bsz*seq, 1]
        if logits_flat.dim() == 2 and logits_flat.shape[1] == 1:
            logits_flat = logits_flat.squeeze(-1)

        logits = logits_flat.view(bsz, seqlen)         # [bsz, seq]

        # Tokens where probe predicts TRUE (logit > 0)
        true_token_mask = (logits > 0).float().detach().save()  # [bsz, seq]

    # 5. Metric function: only uses tokens where probe predicted TRUE on clean run
    def metric_fn(model, submodule, probe, true_token_mask):
        acts = submodule.output
        if isinstance(acts, tuple):
            acts = acts[0]  # [batch, seq, d_model]

        bsz, seqlen, d_model = acts.shape
        acts_flat = acts.reshape(-1, d_model)          # [bsz*seq, d_model]

        # Probe in float32
        logits_flat = probe(acts_flat.float())         # [bsz*seq] or [bsz*seq, 1]
        if logits_flat.dim() == 2 and logits_flat.shape[1] == 1:
            logits_flat = logits_flat.squeeze(-1)

        logits = logits_flat.view(bsz, seqlen)         # [bsz, seq]

        # Only keep tokens that were TRUE on the clean run
        masked_logits = logits * true_token_mask       # [bsz, seq]
        denom = true_token_mask.sum()
        if denom == 0:
            # No TRUE tokens: metric is defined as zero; grads will be zero
            return masked_logits.sum() * 0.0

        metric = masked_logits.sum() / denom           # scalar
        return metric

    total_effect = None
    effects = {}
    deltas = {}
    grads = {}

    # 6. Integrated-gradients-style loop per submodule
    for submodule in submodules:
        dictionary = dictionaries[submodule]
        clean_state = hidden_states_clean[submodule]
        patch_state = hidden_states_patch[submodule]

        grad_acts = []
        grad_ress = []

        for step in range(steps):
            alpha = step / steps

            # Linear path: clean -> patch (ablated)
            f = (1 - alpha) * clean_state + alpha * patch_state

            f.act.requires_grad_(True)
            f.res.requires_grad_(True)
            f.act.retain_grad()
            f.res.retain_grad()

            with model.trace(clean_prefix, **TRACER_KWARGS):
                # Patch this submodule's output for this run
                patched = dictionary.decode(f.act) + f.res

                # VERY IMPORTANT: cast patched activations back to the model's original dtype
                patched = patched.to(orig_dtype[submodule])

                if is_tuple[submodule]:
                    submodule.output[0][:] = patched
                else:
                    submodule.output = patched

                # Metric uses probe_submodule's activations and fixed TRUE-token mask
                value = metric_fn(model, probe_submodule, probe, true_token_mask).save()

            value.backward(retain_graph=True)

            if f.act.grad is None or f.res.grad is None:
                raise RuntimeError(
                    "No grad for f.act or f.res; they are not in the computation graph."
                )

            grad_acts.append(f.act.grad.detach().clone())
            grad_ress.append(f.res.grad.detach().clone())

            # Clear grads for next step
            f.act.grad = None
            f.res.grad = None

        # Average grads over steps (IG approximation)
        mean_grad_act = torch.stack(grad_acts, dim=0).mean(dim=0)
        mean_grad_res = torch.stack(grad_ress, dim=0).mean(dim=0)
        grad = SparseActivation(act=mean_grad_act, res=mean_grad_res)

        # Change in representation between patch and clean
        delta = (patch_state - clean_state).detach() if patch_state is not None else -clean_state.detach()

        # Attribution effect of this submodule (already aggregated over TRUE tokens)
        effect = grad @ delta   # SparseActivation.__matmul__

        effects[submodule] = effect
        deltas[submodule] = delta
        grads[submodule] = grad

        if total_effect is None:
            total_effect = effect
        else:
            total_effect = total_effect + effect

    return effects, deltas, grads, total_effect


def attribution_patching(
    clean_prefix,
    model,
    probe,
    probe_submodule,
    submodules,
    dictionaries,
    steps=10,
    metric_kwargs=dict(),
):

    clean_prefix = torch.cat([clean_prefix], dim=0).to("cuda")

    def metric_fn(model, submodule, probe, token_idx=0):
        acts = submodule.output
        if isinstance(acts, tuple):
            acts = acts[0]   # [batch, seq, d_model]

        # pick the target token
        acts_token = acts[:, token_idx, :]      # [batch, d_model]

        # cuML logistic regression: use logits, NOT predict_proba
        true_logit = probe(acts_token.float())  # [batch]
        return true_logit.mean()
    
    # first run through a test input to figure out which hidden states are tuples
    is_tuple = {}
    with model.trace("_", **TRACER_KWARGS):
        for submodule in submodules:
            is_tuple[submodule] = True # type(submodule.output) == tuple

    hidden_states_clean = {}
    with model.trace(clean_prefix, **TRACER_KWARGS), torch.no_grad():
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.output
            if is_tuple[submodule]:
                x = x[0]
            f = dictionary.encode(x)
            x_hat = dictionary.decode(f)
            residual = x - x_hat
            hidden_states_clean[submodule] = SparseActivation(act=f.save(), res=residual.save())
    hidden_states_clean = {k : v for k, v in hidden_states_clean.items()}

    hidden_states_patch = {
        k : SparseActivation(act=torch.zeros_like(v.act), res=torch.zeros_like(v.res)) for k, v in hidden_states_clean.items()
    }
    total_effect = None

    effects = {}
    deltas = {}
    grads = {}

    for submodule in submodules:
        dictionary = dictionaries[submodule]
        clean_state = hidden_states_clean[submodule]
        patch_state = hidden_states_patch[submodule]
        
        grad_acts = []
        grad_ress = []
        metrics = []

        for step in range(steps):
            alpha = step / steps

            f = (1 - alpha) * clean_state + alpha * patch_state

            f.act.requires_grad_(True)
            f.res.requires_grad_(True)
            f.act.retain_grad()
            f.res.retain_grad()

            with model.trace(clean_prefix, **TRACER_KWARGS):
                if is_tuple[submodule]:
                    submodule.output[0][:] = dictionary.decode(f.act) + f.res
                else:
                    submodule.output = dictionary.decode(f.act) + f.res

                value = metric_fn(model, probe_submodule, probe, **metric_kwargs).save()

            value.backward(retain_graph=True)

            if f.act.grad is None or f.res.grad is None:
                raise RuntimeError(
                    "No grad for f.act or f.res; they are not in the computation graph."
                )

            grad_acts.append(f.act.grad.detach().clone())
            grad_ress.append(f.res.grad.detach().clone())
            metrics.append(value.item())

            f.act.grad = None
            f.res.grad = None

        # average grads over steps
        mean_grad_act = torch.stack(grad_acts, dim=0).mean(dim=0)
        mean_grad_res = torch.stack(grad_ress, dim=0).mean(dim=0)

        grad = SparseActivation(act=mean_grad_act, res=mean_grad_res)

        # change in representation between patch and clean
        delta = (patch_state - clean_state).detach() if patch_state is not None else -clean_state.detach()

        # attribution effect of this submodule
        effect = grad @ delta   # assumes SparseActivation.__matmul__ defined

        effects[submodule] = effect
        deltas[submodule] = delta
        grads[submodule] = grad

        if total_effect is None:
            total_effect = effect
        else:
            total_effect = total_effect + effect

    return (effects, deltas, grads, total_effect)


def attribution_patching_loop(dataset, model, torch_probe, submodule, autoencoder):
    """Perform attribution patching on the dataset."""
    effects = {}
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, example in enumerate(tqdm(dataloader, desc="Attribution patching")):
        if i >= 128:
            break
        tokens = model.tokenizer(example["sentence"][0], return_tensors="pt", padding=False)
        e, _, _, _ = attribution_patching(tokens["input_ids"], model, torch_probe, [submodule], {submodule: autoencoder})
        if submodule not in effects:
            effects[submodule] = e[submodule].sum(dim=0)
        else:
            effects[submodule] += e[submodule].sum(dim=0)
    return effects