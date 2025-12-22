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

            relevant_log_probs = F.log_softmax(relevant_logits, dim=-1)
            
            log_probs_intervention = relevant_log_probs.gather(
                dim=-1, 
                index=target_ids.to(logits.device).unsqueeze(-1)
            ).squeeze(-1).cpu().save()

        with tracer.invoke(input_ids):
            logits = model.lm_head.output
            relevant_logits = logits[:, logit_start:logit_end, :]

            relevant_log_probs = F.log_softmax(relevant_logits, dim=-1)
        
            log_probs_original = relevant_log_probs.gather(
                dim=-1, 
                index=target_ids.to(logits.device).unsqueeze(-1)
            ).squeeze(-1).cpu().save()
    

    log_diff = log_probs_intervention - log_probs_original
    return torch.exp(log_diff) - 1 # return ratio of (new prob - old prob) / old prob


def ablate_batch(model, submodule, autoencoder, tokenizer, input_ids, feature_indices, ablate_mask, prob_mask):
    """
    Ablate features and measure change in log-probability using batch masks.
    
    Args:
        ablate_mask: BoolTensor [Batch, Seq] (True where we ablate)
        prob_mask: BoolTensor [Batch, Seq] (True where we measure probability)
    """
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)

    # 1. Validation
    # We cannot predict the first token (no context), so ensure mask is False there
    prob_mask[:, 0] = False 

    with torch.no_grad():
        with model.trace() as tracer:
            with tracer.invoke(input_ids):
                acts = submodule.input
                # 1. Encode
                # Note: This creates a large dense tensor [Batch, Seq, SAE_Dim]
                encoded_acts = autoencoder.encode(acts)
                
                # 2. Identify the specific contribution we want to REMOVE
                # We want to subtract (Features * Decoder_Weights) for the specific indices
                
                # Create a sparse-like perturbation tensor (Initialize with zeros)
                # This is still memory heavy but we avoid the .clone() of the full state
                perturbation = torch.zeros_like(encoded_acts)
                
                # Select only the features to ablate
                # Shape: [Batch, Seq, K]
                selected_features = encoded_acts[:, :, feature_indices]
                
                # Apply the ablation mask (True where we want to ablate)
                # We keep the value only if the mask is True
                mask_reshaped = rearrange(ablate_mask, 'b s -> b s 1')
                features_to_remove = selected_features * mask_reshaped.float()
                
                # Slot them into the perturbation tensor
                perturbation[:, :, feature_indices] = features_to_remove
                
                # 3. Decode only the perturbation and subtract it from the output
                # Output_new = Output_old - Decode(Features_to_remove)
                ablation_update = autoencoder.decode(perturbation)
                submodule.output = submodule.output - ablation_update

                # 3. Compute All Logits
                logits = model.lm_head.output
                
                # Shift logits/labels for Causal LM (logit[i] predicts input[i+1])
                shifted_logits = logits[:, :-1, :]
                shifted_labels = input_ids[:, 1:]

                # Compute log probs for the whole sequence
                all_log_probs = F.log_softmax(shifted_logits, dim=-1)
                
                # Gather the log prob of the *correct* next token at every position
                # Shape: [Batch, Seq-1]
                token_log_probs_interv = all_log_probs.gather(
                    dim=-1, 
                    index=shifted_labels.unsqueeze(-1)
                ).squeeze(-1).cpu().save()

            with tracer.invoke(input_ids):
                # Repeat logic for Clean run
                logits = model.lm_head.output
                shifted_logits = logits[:, :-1, :]
                shifted_labels = input_ids[:, 1:]

                all_log_probs = F.log_softmax(shifted_logits, dim=-1)
                token_log_probs_orig = all_log_probs.gather(
                    dim=-1, 
                    index=shifted_labels.unsqueeze(-1)
                ).squeeze(-1).cpu().save()
    
    # 4. Filter with Prob Mask
    # We must slice the mask because our log_probs are shifted (Seq-1)
    # prob_mask[:, i] corresponds to input_ids[:, i], which is predicted by logits[:, i-1]
    valid_mask = prob_mask[:, 1:].cpu()

    # Select only the relevant tokens (flattens the output to 1D)
    log_probs_intervention = token_log_probs_interv[valid_mask]
    log_probs_original = token_log_probs_orig[valid_mask]

    log_diff = log_probs_intervention - log_probs_original
    return torch.exp(log_diff) - 1


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