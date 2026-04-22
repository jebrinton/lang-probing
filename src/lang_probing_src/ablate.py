""""
\item $\Delta p(\text{reference})$: change in (log-)probability of the correct reference sequence before and after ablations. Pre-fill the correct sequence and measure the logits of the correct token at all target language positions
\begin{itemize}
\item Translation: give a 2-shot example
\item Non-translation: no examples, single sentence
\item Normalize by the original probability (new - old) / old
\item Try this with input and output features

\item Cache activations before ablating features. We'll show average activations in translation and monolingual contexts (prediction: no significant difference across the settings if noisy channel hypothesis is true)
"""

import torch
from nnsight import LanguageModel

# ablate ablate ablate
import torch
import torch.nn.functional as F
from einops import rearrange
import numpy as np

def get_log_probs(logits, input_ids):
    # Calculate probabilities
    # We are interested in the probability of the *next* token. 
    # Logits at index [i] predict token at [i+1].
    
    # Shift logits and labels for calculating loss/prob on the reference
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    
    # Calculate CrossEntropy (NLL) or raw LogProbs
    # This gives log_prob per token
    log_probs = -F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), 
        shift_labels.view(-1), 
        reduction='none'
    )
    log_probs = log_probs.view(shift_labels.shape)

    return log_probs


def ablate(model, submodule, autoencoder, tokenizer, input_ids, feature_indices, ablate_positions, prob_positions):
    """
    Ablate features in the prompt and measure the change in log-probability 
    of the reference tokens.

    Note: prob_positions.start should be at least 1
    """
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    
    # Extract start/end from the slice object
    # prob_positions corresponds to the indices of the tokens we want to measure
    target_start = prob_positions.start
    target_end = prob_positions.stop

    if target_start < 1:
        raise ValueError("prob_positions.start should be at least 1")

    # To predict token[i], we need logit[i-1].
    logit_start = target_start - 1
    logit_end = target_end - 1

    # Prepare the target token IDs we want to gather probabilities for
    # We move this to the same device as the model during the trace
    target_ids = input_ids[:, target_start:target_end]

    with model.trace() as tracer:
        with tracer.invoke(input_ids):
            acts = submodule.input
            encoded_acts = autoencoder.encode(acts)
            encoded_acts_clean = encoded_acts.clone()
            if feature_indices is not None:
                encoded_acts[:, ablate_positions, feature_indices] = 0

            decoded_acts_clean = autoencoder.decode(encoded_acts_clean)
            decoded_acts = autoencoder.decode(encoded_acts)
            # Ensure dtype matches (important for BFloat16/Float16 mixed precision)
            decoded_acts_clean = decoded_acts_clean.to(acts.dtype)
            decoded_acts = decoded_acts.to(acts.dtype)

            submodule.output = submodule.output + (decoded_acts - decoded_acts_clean)
            
            logits = model.lm_head.output
            relevant_logits = logits[:, logit_start:logit_end, :]

            relevant_log_probs = F.log_softmax(relevant_logits.float(), dim=-1)
            
            log_probs_intervention = relevant_log_probs.gather(
                dim=-1, 
                index=target_ids.to(logits.device).unsqueeze(-1)
            ).squeeze(-1).cpu().save()

        with tracer.invoke(input_ids):
            logits = model.lm_head.output
            relevant_logits = logits[:, logit_start:logit_end, :]

            relevant_log_probs = F.log_softmax(relevant_logits.float(), dim=-1)
        
            log_probs_original = relevant_log_probs.gather(
                dim=-1, 
                index=target_ids.to(logits.device).unsqueeze(-1)
            ).squeeze(-1).cpu().save()
    

    log_diff = log_probs_intervention - log_probs_original
    return torch.exp(log_diff) - 1 # return ratio of (new prob - old prob) / old prob


def get_probe_ablation_mask(model, input_ids, probe, probe_layer, ablate_region_mask, device="cpu"):
    """
    Get ablation mask from probe: True only where probe predicts concept=value (logit > 0)
    and the position is inside the ablation region (source/target).

    Args:
        model: nnsight LanguageModel
        input_ids: [Batch, Seq]
        probe: sklearn LogisticRegression (e.g. from probe.load_probe)
        probe_layer: int, layer index the probe was trained on
        ablate_region_mask: BoolTensor [Batch, Seq] (True = source or target segment)
        device: device for tensors

    Returns:
        BoolTensor [Batch, Seq]: True where we should ablate (region AND probe logit > 0)
    """
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    # Probe filenames use HuggingFace convention: hidden_states[layer_num] where layer_num can be 32
    # for the last layer. model.model.layers has indices 0..num_layers-1, so clamp to valid range.
    num_layers = len(model.model.layers)
    layer_index = min(probe_layer, num_layers - 1)
    with torch.no_grad():
        with model.trace() as tracer:
            with tracer.invoke(input_ids):
                out = model.model.layers[layer_index].output
                acts = out[0] if isinstance(out, tuple) else out
                acts = acts.cpu().save()
    acts = getattr(acts, "value", acts)
    B, S, H = acts.shape
    # NumPy does not support BFloat16; convert to float32 for probe.decision_function
    acts_np = acts.float().reshape(B * S, H).numpy()
    logits = probe.decision_function(acts_np)
    probe_logits = torch.tensor(logits, dtype=torch.float32, device=device).reshape(B, S)
    ablate_mask_probe = (probe_logits > 0) & ablate_region_mask
    return ablate_mask_probe


def logits_to_probs(logits, input_ids):
    # Shift logits/labels for Causal LM (logit[i] predicts input[i+1])
    # Compute in float32 to avoid bfloat16 quantization flipping signs when deltas are tiny.
    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = input_ids[:, 1:]

    # Compute log probs for the whole sequence
    all_log_probs = F.log_softmax(shifted_logits, dim=-1)
    
    # Gather the log prob of the *correct* next token at every position
    # Shape: [Batch, Seq-1]
    token_log_probs = all_log_probs.gather(
        dim=-1, 
        index=shifted_labels.unsqueeze(-1)
    ).squeeze(-1)

    return token_log_probs


def ablate_batch(model, submodule, autoencoder, tokenizer, input_ids, feature_indices, ablate_mask, prob_mask):
    """
    Ablate features and measure change in log-probability using batch masks.

    Returns:
        dict with:
        - result_intervention: 1D tensor, exp(Δ log p) - 1 per scored token (~relative prob change)
        - mean_logprob_delta, min_logprob_delta: scalars, mean/min of Δ log p on scored tokens
        - frac_active_at_ablated: fraction of ablated SAE coeffs > 0 pre-ablation (nan if no sites)
    """
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)

    # 1. Validation
    # We cannot predict the first token (no context), so ensure mask is False there
    prob_mask[:, 0] = False 
    
    feature_indices = np.asarray(feature_indices)
    with torch.no_grad():
        with model.trace() as tracer:
            with tracer.invoke(input_ids):
                acts = submodule.output
                encoded_acts = autoencoder.encode(acts)
                encoded_acts_clean = encoded_acts.clone()
                # Shape: [Batch, Seq, K]
                selected_features = encoded_acts[:, :, feature_indices]
                pre_abl_slice = selected_features.clone().cpu().save()
                mask_reshaped = rearrange(ablate_mask, 'b s -> b s 1')
                ablated_slice = selected_features.masked_fill(mask_reshaped, 0.0)
                encoded_acts[:, :, feature_indices] = ablated_slice
                decoded_acts = autoencoder.decode(encoded_acts)
                decoded_acts_clean = autoencoder.decode(encoded_acts_clean)
                submodule.output = submodule.output + (decoded_acts - decoded_acts_clean)
                logits = model.lm_head.output
                token_log_probs_interv = logits_to_probs(logits, input_ids).cpu().save()

            with tracer.invoke(input_ids):
                logits = model.lm_head.output
                token_log_probs_orig = logits_to_probs(logits, input_ids).cpu().save()

    valid_mask = prob_mask[:, 1:].cpu()  # aligns with logits_to_probs output: [B, S-1]
    log_probs_interv = getattr(token_log_probs_interv, "value", token_log_probs_interv)
    log_probs_orig = getattr(token_log_probs_orig, "value", token_log_probs_orig)

    token_log_diff_full = log_probs_interv - log_probs_orig  # [B, S-1]
    log_diff_intervention = token_log_diff_full[valid_mask]

    result_intervention = torch.exp(log_diff_intervention) - 1
    mean_log_d = float(log_diff_intervention.float().mean().item())
    min_log_d = float(log_diff_intervention.float().min().item())

    # Fraction of selected SAE features that were active (>0) at ablated positions, pre-ablation
    ablate_mask_cpu = ablate_mask.cpu()
    pre_abl = getattr(pre_abl_slice, "value", pre_abl_slice)
    if ablate_mask_cpu.any() and pre_abl is not None:
        active_at_ablated = (pre_abl[ablate_mask_cpu] > 0).float().mean().item()
    else:
        active_at_ablated = float("nan")

    return {
        "result_intervention": result_intervention,
        "mean_logprob_delta": mean_log_d,
        "min_logprob_delta": min_log_d,
        "frac_active_at_ablated": active_at_ablated,
    }


# old code that went between acts = submodule.input and submodule.output = submodule.output + (decoded_acts - decoded_acts_clean)
# # 1. Encode
#             # Note: This creates a large dense tensor [Batch, Seq, SAE_Dim]
#             encoded_acts = autoencoder.encode(acts)
            
#             # 2. Identify the specific contribution we want to REMOVE
#             # We want to subtract (Features * Decoder_Weights) for the specific indices
            
#             # Create a sparse-like perturbation tensor (Initialize with zeros)
#             # This is still memory heavy but we avoid the .clone() of the full state
#             perturbation = torch.zeros_like(encoded_acts)
            
#             # Select only the features to ablate
#             # Shape: [Batch, Seq, K]
#             selected_features = encoded_acts[:, :, feature_indices]
            
#             # Apply the ablation mask (True where we want to ablate)
#             # We keep the value only if the mask is True
#             mask_reshaped = rearrange(ablate_mask, 'b s -> b s 1')
#             features_to_remove = selected_features * mask_reshaped.float()
            
#             # Slot them into the perturbation tensor
#             perturbation[:, :, feature_indices] = features_to_remove
            
#             # 3. Decode only the perturbation and subtract it from the output
#             # Output_new = Output_old - Decode(Features_to_remove)
#             ablation_update = autoencoder.decode(perturbation)
#             submodule.output = submodule.output - ablation_update

# new code that might not work perfectly
                # # 1. Encode
                # # Note: This creates a large dense tensor [Batch, Seq, SAE_Dim]
                # encoded_acts = autoencoder.encode(acts)
                
                # # 2. Identify the specific contribution we want to REMOVE
                # # We want to subtract (Features * Decoder_Weights) for the specific indices
                
                # # Create a sparse-like perturbation tensor (Initialize with zeros)
                # # This is still memory heavy but we avoid the .clone() of the full state
                # perturbation = torch.zeros_like(encoded_acts)
                
                # # Select only the features to ablate
                # # Shape: [Batch, Seq, K]
                # selected_features = encoded_acts[:, :, feature_indices]
                
                # # Apply the ablation mask (True where we want to ablate)
                # # We keep the value only if the mask is True
                # mask_reshaped = rearrange(ablate_mask, 'b s -> b s 1')
                # features_to_remove = selected_features * mask_reshaped.float()
                
                # # Slot them into the perturbation tensor
                # perturbation[:, :, feature_indices] = features_to_remove
                
                # # 3. Decode only the perturbation and subtract it from the output
                # # Output_new = Output_old - Decode(Features_to_remove)
                # ablation_update = autoencoder.decode(perturbation)
                # submodule.output = submodule.output - ablation_update


def ablate_bleu(model, submodule, autoencoder, tokenizer, input_ids, feature_indices, ablate_positions, prob_positions):
    """
    Ablate features indicated in feature_indices across positions in ablate_positions
    Return the change in logprobs at prob_positions as a ratio of delta prob / original sequence prob
    """
    # for batch in dataloader:
        # using .all():
    layers = model.model.layers
    n_new_tokens = 15
    with model.generate(input_ids, max_new_tokens=n_new_tokens) as tracer:
        hidden_states = list().save() # Initialize & .save() nnsight list

        # Call .all() to apply intervention to each new token
        with tracer.iter[ablate_positions]:

            # Ablate SAE features
            acts = submodule.input
            encoded_acts = autoencoder.encode(acts)
            encoded_acts[:, :, feature_indices] = 0
            decoded_acts = autoencoder.decode(encoded_acts)
            
            # Cast the decoded acts to match the dtype of the original acts (BFloat16)
            decoded_acts = decoded_acts.to(acts.dtype)
            submodule.output = decoded_acts

            # Append desired hidden state post-intervention
            hidden_states.append(layers[-1].output)

        out = model.generator.output.save()

    decoded_prompt = model.tokenizer.decode(out[0][0:-n_new_tokens].cpu())
    decoded_answer = model.tokenizer.decode(out[0][-n_new_tokens:].cpu())

    print("Prompt: ", decoded_prompt)
    print("Generated Answer: ", decoded_answer)

    print("Hidden state length: ",len(hidden_states))
    print("Hidden state shape",hidden_states[0].shape)