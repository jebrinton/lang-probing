""""
\item $\Delta p(\text{reference})$: change in (log-)probability of the correct reference sequence before and after ablations. Pre-fill the correct sequence and measure the logits of the correct token at all target language positions
\begin{itemize}
    \item Translation: give a 2-shot example
    \item Non-translation: no examples, single sentence
    \item Normalize by the original probability (new - old) / old
    \item Try this with input and output features
    \item Also do a random baseline
    \item [2nd] Input features translation: ablate features at source sentence token indices, measure prob of target sequence before and after ablation
    \item Input features monolingual: ablate features for sentence, measure prob of sentence before and after ablation
    \item Output features translation: ablate features at target sequence token indices, measure prob of target sequence before and after ablation
    % possible for a model to perform grammatical computations before we start the ablation (during the source sequence), then copy it over in layers after the 16th layer and it won't get ablated
    \item Output features monolingual: ablate features for sentence, measure prob of sequence before and after ablation
"""

import argparse
from pathlib import Path
import itertools
import logging
import gc
import json

from nnsight import LanguageModel
from torch.utils.data import DataLoader
from datasets import load_dataset
import numpy as np
import torch

from lang_probing_src.utils import setup_model, get_device_info
from lang_probing_src.config import MODEL_ID, SAE_ID, NAME_TO_LANG_CODE, LANGUAGES_DEC
from lang_probing_src.ablate import ablate, ablate_batch
from lang_probing_src.utils_input_output import get_output_features_vector, get_input_features_vector, load_effects_files, get_language_pairs_and_concepts


EXP_CONFIGS = {
    # 1. Monolingual Input
    "mono_input": {
        "mode": "monolingual", "ablate_loc": "source", "prob_loc": "source", "feats": "input"
    },
    # 2. Monolingual Output
    "mono_output": {
        "mode": "monolingual", "ablate_loc": "source", "prob_loc": "source", "feats": "output"
    },
    # 3. Multilingual Input (Ablate Source, Measure Target)
    "multi_input": {
        "mode": "multilingual", "ablate_loc": "source", "prob_loc": "target", "feats": "input"
    },
    # 4. Multilingual Output (Ablate Target, Measure Target)
    "multi_output": {
        "mode": "multilingual", "ablate_loc": "target", "prob_loc": "target", "feats": "output"
    },
    # 5. Monolingual Random Baseline
    "mono_random": {
        "mode": "monolingual", "ablate_loc": "source", "prob_loc": "source", "feats": "random"
    },
    # 6. Multilingual Random Source Baseline
    "multi_random_src": {
        "mode": "multilingual", "ablate_loc": "source", "prob_loc": "target", "feats": "random"
    },
    # 7. Multilingual Random Target Baseline
    "multi_random_tgt": {
        "mode": "multilingual", "ablate_loc": "target", "prob_loc": "target", "feats": "random"
    },
}


def get_feature_indices(K, feature_vector):
    """Return the top K feature indices from a feature vector"""
    return np.argsort(feature_vector)[-K:] # get the largest K features

def get_batch_positions_masks(tokenizer, contexts, sources, targets, device="cpu"):
    """
    Constructs the batch input_ids and boolean masks for source and target positions.
    """
    prompts = []
    # 1. Construct text prompts based on provided parts
    for ctx, src, tgt in zip(contexts, sources, targets):
        if tgt:
            # Multilingual: "Context... Source >> Target"
            prompt = f"{ctx}{src} >> {tgt}"
        else:
            # Monolingual: Just Source
            prompt = src
        prompts.append(prompt)

    # 2. Tokenize batch
    encoding = tokenizer(prompts, padding=True, return_tensors="pt", return_offsets_mapping=True)
    input_ids = encoding["input_ids"].to(device)
    offset_mapping = encoding["offset_mapping"] # Shape: (batch, seq_len, 2)

    batch_size, seq_len = input_ids.shape
    source_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
    target_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)

    # 3. Create Masks based on character alignments
    for i in range(batch_size):
        ctx_len = len(contexts[i])
        src_len = len(sources[i])
        tgt_len = len(targets[i]) if targets[i] else 0
        
        src_start_char = ctx_len
        src_end_char = src_start_char + src_len
        
        separator_len = 4 if tgt_len > 0 else 0 
        tgt_start_char = src_end_char + separator_len
        tgt_end_char = tgt_start_char + tgt_len

        offsets = offset_mapping[i]
        
        for j, (start, end) in enumerate(offsets):
            if start == end == 0: continue 
            
            if start >= src_start_char and end <= src_end_char:
                source_mask[i, j] = True
            
            if tgt_len > 0 and start >= tgt_start_char and end <= tgt_end_char:
                target_mask[i, j] = True

    return input_ids, source_mask, target_mask


def old_get_batch_positions_masks(tokenizer, contexts, source_sentences, target_sentences, device="cuda"):
    """
    Tokenizes a batch of text and returns masks for specific segments.
    """
    full_texts = []
    char_ranges = []

    # 1. Construct full strings and calculate character offsets for each sample
    for ctx, src, tgt in zip(contexts, source_sentences, target_sentences):
        full_text = ctx + src + tgt
        full_texts.append(full_text)
        
        # Calculate char boundaries
        src_start = len(ctx)
        tgt_start = src_start + len(src)
        full_end = tgt_start + len(tgt)
        char_ranges.append((src_start, tgt_start, full_end))

    # 2. Batch Tokenize with padding and offsets
    # return_offsets_mapping gives us the char start/end for every token
    encoding = tokenizer(
        full_texts, 
        padding=True, 
        return_tensors="pt", 
        return_offsets_mapping=True,
    )

    input_ids = encoding["input_ids"].to(device)
    offset_mapping = encoding["offset_mapping"] # Shape: (batch, seq_len, 2)

    batch_size, seq_len = input_ids.shape
    
    # 3. Create Boolean Masks for Ablation and Probing (based off specific experiment settings)
    source_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
    target_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)

    # We must iterate locally to map char offsets to token indices because 
    # offset_mapping is specific to the tokenization of that specific string.
    # However, this loop is fast (CPU only, over batch size) compared to model inference.
    offset_mapping_np = offset_mapping.numpy()
    
    for i in range(batch_size):
        src_char_start, tgt_char_start, full_char_end = char_ranges[i]
        offsets = offset_mapping_np[i] # (seq_len, 2)
        
        # Vectorized check for this single sample
        # Token is in Source if: token_start >= src_start AND token_end <= tgt_start
        # Note: We use the logic from your original code (full containment usually safer)
        
        # Extract columns for faster comparison
        token_starts = offsets[:, 0]
        token_ends = offsets[:, 1]
        
        # Logic: Indices where the token falls strictly within the source char range
        # Note: (token_ends > 0) checks that it's not a special padding token (usually 0,0)
        src_indices = (token_starts >= src_char_start) & (token_ends <= tgt_start) & (token_ends > 0)
        tgt_indices = (token_starts >= tgt_char_start) & (token_ends <= full_char_end) & (token_ends > 0)
        
        source_mask[i] = torch.tensor(src_indices, device=device)
        target_mask[i] = torch.tensor(tgt_indices, device=device)

    return input_ids, source_mask, target_mask



def get_positions(tokenizer, context, source_sentence, target_sentence):
    # 1. Create the full prompt
    full_text = context + source_sentence + target_sentence

    # 2. Find character start/end of the Spanish substring
    # Note: You must ensure 'source_sentence' is exactly as it appears in full_text
    source_sentence_start = len(context)
    target_sentence_start = source_sentence_start + len(source_sentence)
    full_text_end = target_sentence_start + len(target_sentence)

    # 3. Tokenize with offsets
    encoding = tokenizer(full_text, return_offsets_mapping=True)
    offset_mapping = encoding["offset_mapping"] # List of (char_start, char_end) tuples
    input_ids = encoding["input_ids"]

    # 4. Find which tokens fall within the character range
    target_token_indices = []
    source_token_indices = []

    for idx, (start, end) in enumerate(offset_mapping):
        # We check if the token has significant overlap with the spanish substring
        # Using max(start, char_start) < min(end, char_end) checks for ANY overlap
        if start >= source_sentence_start and end <= target_sentence_start:
            source_token_indices.append(idx)
        elif start >= target_sentence_start and end <= full_text_end:
            target_token_indices.append(idx)

    # print the tokens at ablate_positions and prob_positions

    ablate_positions = slice(source_token_indices[0], source_token_indices[-1])
    prob_positions = slice(target_token_indices[0], target_token_indices[-1])

    # print the whole prompt
    
    # Print the tokens at ablate_positions and prob_positions
    ablate_tokens = tokenizer.decode(input_ids[ablate_positions], skip_special_tokens=True)
    prob_tokens = tokenizer.decode(input_ids[prob_positions], skip_special_tokens=True)

    logging.info(f"Context: {context}")

    # logging.debug("Ablate positions indices:", list(range(ablate_positions.start, ablate_positions.stop)))
    logging.info("Ablate position tokens:", ablate_tokens)
    # logging.debug("Prob positions indices:", list(range(prob_positions.start, prob_positions.stop)))
    logging.info("Prob position tokens:", prob_tokens)

    logging.info(f"Source sentence: {source_sentence}")
    logging.info(f"Target sentence: {target_sentence}")

    return input_ids, ablate_positions, prob_positions

def save_result_jsonl(filename, data_dict):
    """
    Appends a dictionary as a JSON line to the specified file.
    Flushes immediately to ensure data survival on crash.
    """
    file_path = Path(filename)
    with open(file_path, mode='a', encoding='utf-8') as f:
        f.write(json.dumps(data_dict) + "\n")
        f.flush()

def main(args):
    logging.basicConfig(level=logging.INFO)
    model, submodule, autoencoder, tokenizer = setup_model(MODEL_ID, SAE_ID)
    device, _ = get_device_info()
    
    # Create output directory if not exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define the specific output file for this experiment
    output_file = output_dir / f"results_{args.experiment}.jsonl"
    logging.info(f"Saving results to: {output_file}")
    
    # Setup Model
    model, submodule, autoencoder, tokenizer = setup_model(MODEL_ID, SAE_ID)
    
    # Configuration
    exp_cfg = EXP_CONFIGS[args.experiment]
    input_features_dir = Path("/projectnb/mcnet/jbrin/lang-probing/outputs/sentence_input_features/")
    effects_files = load_effects_files() if "output" in exp_cfg['feats'] else None

    # TODO: ensure that language pairs for which there is no concept=value get handled gracefully
    # Iterate Languages
    if exp_cfg['mode'] == 'multilingual':
        iterator = itertools.permutations(LANGUAGES_DEC, 2)
    elif exp_cfg['feats'] == 'output':
        iterator = [(None, lang) for lang in LANGUAGES_DEC]
    else:
        iterator = [(lang, None) for lang in LANGUAGES_DEC]

    for source_lang, target_lang in iterator:
        logging.info(f"Running {args.experiment}: {source_lang} -> {target_lang}")

        if source_lang is not None:
            source_dataset = load_dataset("gsarti/flores_101", NAME_TO_LANG_CODE[source_lang], split="devtest")
        else:
            source_dataset = None
        if target_lang is not None:
            target_dataset = load_dataset("gsarti/flores_101", NAME_TO_LANG_CODE[target_lang], split="devtest")
        else:
            target_dataset = None
        
        
        num_samples = args.max_samples

        # --- SELECT FEATURES ---
        feature_indices = []
        try:
            if exp_cfg['feats'] == 'input':
                feats_vec = get_input_features_vector(input_features_dir, source_lang, args.concept, args.value)
                feature_indices = np.argsort(feats_vec)[-args.k:] 
            elif exp_cfg['feats'] == 'output':
                if source_lang is not None:
                    feat_vec = get_output_features_vector(effects_files, (source_lang, target_lang), args.concept, args.value)
                else:
                    # TODO: clean this up
                    feat_vecs = [get_output_features_vector(effects_files, (source_lang, target_lang), args.concept, args.value) for source_lang in LANGUAGES_DEC]
                    feat_vec = np.mean(feat_vecs, axis=0)
                feature_indices = np.argsort(feat_vec)[-args.k:]
            elif exp_cfg['feats'] == 'random':
                # Use input vector to get correct SAE dimension size
                temp_vec = get_input_features_vector(input_features_dir, source_lang, args.concept, args.value)
                sae_dim = len(temp_vec)
                feature_indices = np.random.choice(sae_dim, args.k, replace=False)
        except Exception as e:
            logging.warning(f"Could not load features for {source_lang}->{target_lang}: {e}")
            continue

        # --- PREPARE BATCHES ---
        all_contexts, all_sources, all_targets = [], [], []

        for i in range(num_samples):
            idx_1 = (i + 1) % num_samples
            idx_2 = (i + 2) % num_samples
            
            if exp_cfg['mode'] == 'multilingual':
                ctx = f"{source_dataset[idx_2]['sentence']} >> {target_dataset[idx_2]['sentence']}\n{source_dataset[idx_1]['sentence']} >> {target_dataset[idx_1]['sentence']}\n"
                tgt = target_dataset[i]['sentence']
            else:
                ctx = ""
                tgt = None

            src = source_dataset[i]['sentence']
            all_contexts.append(ctx)
            all_sources.append(src)
            all_targets.append(tgt)

        # --- BATCH LOOP ---
        batch_means = []
        batch_mins = []

        for i in range(0, num_samples, args.batch_size):
            b_ctx = all_contexts[i : i + args.batch_size]
            b_src = all_sources[i : i + args.batch_size]
            b_tgt = all_targets[i : i + args.batch_size]

            # Get Masks
            input_ids, src_mask, tgt_mask = get_batch_positions_masks(
                tokenizer, b_ctx, b_src, b_tgt, device=device
            )
            
            ablate_mask = src_mask if exp_cfg['ablate_loc'] == 'source' else tgt_mask
            prob_mask = src_mask if exp_cfg['prob_loc'] == 'source' else tgt_mask

            if not ablate_mask.any() or not prob_mask.any():
                continue

            try:
                # Assuming feature_indices is numpy, convert to list or tensor if needed by ablate_batch
                # usually list is fine for nnsight logic, or tensor
                delta_p = ablate_batch(
                    model,
                    submodule,
                    autoencoder,
                    tokenizer,
                    input_ids=input_ids,
                    feature_indices=feature_indices,
                    ablate_mask=ablate_mask,
                    prob_mask=prob_mask
                )
                
                batch_means.append(delta_p.mean().item())
                batch_mins.append(delta_p.min().item())

            except Exception as e:
                logging.error(f"Batch failed for {source_lang}->{target_lang}: {e}")
                continue
            
            # Cleanup
            del input_ids, src_mask, tgt_mask, delta_p
            torch.cuda.empty_cache()

        # --- SAVE RESULTS ---
        if batch_means:
            final_mean = np.mean(batch_means)
            final_min = np.min(batch_mins)
            
            print(f"\n\n\n\n\n\n\n\n\n\n\n\n\n[{args.experiment}] {source_lang}->{target_lang} | Mean: {final_mean:.4f} | Min: {final_min:.4f}")
            
            result_entry = {
                "experiment": args.experiment,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "concept": args.concept,
                "value": args.value,
                "k": args.k,
                "mean_delta": float(final_mean),
                "min_delta": float(final_min),
                "num_samples": num_samples
            }
            
            save_result_jsonl(output_file, result_entry)
        else:
            logging.warning(f"No results for {source_lang}->{target_lang}")
        
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run language ablation experiments.")
    parser.add_argument("--experiment", type=str, required=True, choices=EXP_CONFIGS.keys(), help="Which experiment to run")
    parser.add_argument("--concept", type=str, default="Tense")
    parser.add_argument("--value", type=str, default="Past")
    parser.add_argument("--k", type=int, default=1, help="Number of features to ablate")
    parser.add_argument("--max_samples", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="outputs/ablation_results", help="Directory to save jsonl files")
    
    args = parser.parse_args()
    main(args)
