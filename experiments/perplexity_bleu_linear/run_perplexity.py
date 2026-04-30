import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import math

import pandas as pd
from datasets import load_dataset
import argparse
import pathlib
import os

from lang_probing_src.config import LANG_CODE_TO_NAME, MODEL_TO_ID, OUTPUTS_DIR

def calculate_perplexity(
    model_id, 
    dataset, 
    feature_column="text", 
    batch_size=64, 
    device="cuda",
    max_length=None
):
    # --- 1. Setup & Padding Fix (Constraint #1) ---
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # CRITICAL: Force right-padding. 
    # If left-padded, the model sees [PAD, A, B]. Generating the first token 'A' 
    # becomes impossible because it is conditioned on a PAD token, spiking loss.
    tokenizer.padding_side = "right" 
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if "cuda" in device else torch.float32
    ).to(device)
    model.eval()

    # --- 2. Counters for Corpus Perplexity (Constraint #4) ---
    # We do NOT average the perplexity of batches. We sum the NLL of the whole corpus.
    total_nll = 0.0
    total_tokens = 0

    # Define Loss function with NO reduction (Constraint #2)
    # This allows us to apply the mask *before* averaging.
    loss_fct = nn.CrossEntropyLoss(reduction='none')

    # --- 3. Batch Loop ---
    # Convert dataset to list for easier slicing if it's not indexable
    texts = dataset[feature_column]
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating PPL"):
        batch_texts = texts[i : i + batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=max_length or model.config.max_position_embeddings
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # --- 4. The "Shift" Logic (Constraint #3) ---
        # We must align the prediction at pos `t` with the actual token at `t+1`.
        
        # Logits: remove the LAST token (we don't have a next token to predict for it)
        shift_logits = logits[..., :-1, :].contiguous()
        
        # Labels: remove the FIRST token (we don't predict the first token, it's context)
        shift_labels = inputs.input_ids[..., 1:].contiguous()
        
        # Mask: remove the FIRST token (matches the labels alignment)
        shift_attention_mask = inputs.attention_mask[..., 1:].contiguous()

        # --- 5. Calculate Raw Loss ---
        # Flatten tensors to [batch_size * seq_len, vocab_size]
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        
        # Calculate loss for EVERY token (no mean yet)
        raw_loss = loss_fct(flat_logits, flat_labels)
        
        # Reshape back to [batch_size, seq_len] to apply mask
        raw_loss = raw_loss.view(shift_labels.size())
        
        # Apply the mask (Zero out loss for padding tokens)
        masked_loss = raw_loss * shift_attention_mask

        # --- 6. Accumulate (Constraint #2 & #4) ---
        # Sum the exact NLL for this batch
        total_nll += masked_loss.sum().item()
        
        # Count exactly how many non-pad tokens were in the shift
        total_tokens += shift_attention_mask.sum().item()

    # --- 7. Final Calculation ---
    if total_tokens == 0:
        return float("nan")
        
    # Corpus Perplexity = exp(Total NLL / Total Tokens)
    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll)
    
    return ppl


def main(args):
    results = []

    print(f"Starting evaluation on {len(LANG_CODE_TO_NAME.keys())} languages...")

    for lang_code in LANG_CODE_TO_NAME.keys():
        try:
            # 1. Load the specific language subset
            # FLORES-200 uses the language code as the config name (e.g., 'eng_Latn')
            ds = load_dataset(
                "gsarti/flores_101", 
                lang_code, 
                split="devtest",    
            )
            
            # 2. Calculate Perplexity
            # Note: FLORES usually stores the text in a column named 'sentence'
            ppl = calculate_perplexity(
                model_id=args.model_id,
                dataset=ds,
                feature_column="sentence", # Specific to FLORES
                batch_size=args.batch_size # Adjust based on your VRAM
            )
            
            print(f"[{lang_code}] PPL: {ppl:.4f}")
            results.append({"Language": lang_code, "Perplexity": ppl})

        except Exception as e:
            print(f"Error processing {lang_code}: {e}")
            results.append({"Language": lang_code, "Perplexity": "Error"})

    # 3. Present Results
    df_results = pd.DataFrame(results)
    out_dir = os.path.join(OUTPUTS_DIR, "perplexity_bleu_linear", "bleu_and_ppl")
    os.makedirs(out_dir, exist_ok=True)
    # Resolve a stable short model name from MODEL_TO_ID instead of fragile substring sniffing.
    model_name = next(
        (key for key, mid in MODEL_TO_ID.items() if mid == args.model_id),
        None,
    )
    if model_name is None:
        # Fall back to a sanitized form of the model id rather than silently mislabeling.
        model_name = args.model_id.replace("/", "_")
    df_results.to_csv(os.path.join(out_dir, f"perplexity_results_{model_name}.csv"), index=False)
    print("\n=== Final Results ===")
    print(df_results.to_markdown(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    main(args)
