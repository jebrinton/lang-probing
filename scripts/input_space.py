import os
import glob

from torch.utils.data import DataLoader
from lang_probing_src.config import UD_BASE_FOLDER
import argparse

import h5py
import json
import numpy as np
import torch
import itertools
from tqdm import tqdm

from torch.utils.data import Subset

from lang_probing_src.data import FloresDataset
from lang_probing_src.data import ConlluDataset
from lang_probing_src.data import collate_fn
from lang_probing_src.utils import setup_model
from lang_probing_src.config import OUTPUTS_DIR
from lang_probing_src.utils import ensure_dir

def save_tags(tags, tags_dir):
    """
    Saves tags to a JSONL file.
    
    Args:
        tags (list): A list of [n_samples] dictionaries, where each dict
                     holds the pooled linguistic tags for a sentence.
        jsonl_path (str): Filepath to save the .jsonl file (e.g., "en_tags.jsonl").
    """
    with open(os.path.join(tags_dir, f"tags.jsonl"), 'w', encoding='utf-8') as f:
        for tag_dict in tags:
            # Convert sets to lists for JSON serialization
            serializable_tags = {
                key: list(value) if isinstance(value, set) else value 
                for key, value in tag_dict.items()
            }
            f.write(json.dumps(serializable_tags) + '\n')

    return len(tags)

def save_activations(activations, sae_activations_dir):
    """
    Saves activations to an HDF5 file and tags to a JSONL file.
    
    Args:
        activations (np.array): The [n_samples, dict_size] array of SAE activations.
        h5_path (str): Filepath to save the .h5 file (e.g., "en_activations.h5").
    """
    with h5py.File(os.path.join(sae_activations_dir, f"activations.h5"), 'w') as f:
        f.create_dataset(
            'activations',
            data=activations,
            dtype='float32',
            compression='gzip'  # Use compression
        )

    return len(activations)

def extract_and_save_data(
    model, 
    submodule, 
    autoencoder, 
    dataloader, 
    sae_activations_dir,
    tags_dir,
    pooling_strategy="max", 
    tracer_kwargs=None
):
    if tracer_kwargs is None:
        tracer_kwargs = {} 

    all_activations_list = []
    all_tags_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting SAE activations"):
            text_batch = batch["sentence"]
            tags_batch = batch["tags"]
            
            # --- FIX START: Tokenize manually ---
            # Tokenize inputs manually so we hold the reference to the attention mask
            # We assume the model is already on the correct device, so we move inputs there.
            inputs = model.tokenizer(
                text_batch, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(model.device)
            
            attn_mask = inputs["attention_mask"]
            input_ids = inputs["input_ids"]
            
            # Pass explicit tensors to trace instead of raw strings
            with model.trace(input_ids, attention_mask=attn_mask, **tracer_kwargs):
                acts = submodule.output[0].save()
            
            # Note: We removed `attn_mask = model.inputs[0][1].value`
            # We use the `attn_mask` variable we created above.
            
            acts_tensor = acts.value # [batch_size, seq_len, hidden_dim]
            # --- FIX END ---

            batch_size, seq_len, hidden_dim = acts_tensor.shape
            
            # 1. Reshape for SAE: [B, S, H] -> [B*S, H]
            acts_2d = acts_tensor.view(batch_size * seq_len, hidden_dim)
            
            # 2. Encode all tokens
            sae_acts_2d = autoencoder.encode(acts_2d) 
            
            dict_size = sae_acts_2d.shape[-1]
            
            # 3. Reshape back to 3D
            sae_acts_3d = sae_acts_2d.view(batch_size, seq_len, dict_size)
            
            # 4. Pool the SAE features
            attn_mask_expanded = attn_mask.unsqueeze(-1)
            
            if pooling_strategy == "max":
                sae_acts_3d_masked = sae_acts_3d.masked_fill(
                    attn_mask_expanded == 0, 
                    -torch.inf
                )
                pooled_sae_activations, _ = torch.max(sae_acts_3d_masked, dim=1)
                
            elif pooling_strategy == "mean":
                sae_acts_3d = sae_acts_3d * attn_mask_expanded
                seq_lengths = attn_mask.sum(dim=1, keepdim=True).float()
                pooled_sae_activations = sae_acts_3d.sum(1) / (seq_lengths + 1e-9)

            else:
                raise ValueError(f"Unknown pooling_strategy: {pooling_strategy}")

            # 5. Store results
            all_activations_list.append(pooled_sae_activations.float().cpu().numpy())
            all_tags_list.append(tags_batch)

    # (Rest of function remains unchanged)
    print("Concatenating all batches...")
    final_activations = np.vstack(all_activations_list)
    final_tags = list(itertools.chain.from_iterable(all_tags_list))
    
    save_activations(final_activations, os.path.join(sae_activations_dir, f"activations.h5"))
    save_tags(final_tags, os.path.join(tags_dir, f"tags.jsonl"))
    
    return final_activations, final_tags


def extract_sae_activations(
    model, 
    submodule, 
    autoencoder, 
    dataloader,
    pooling_strategy="max", 
    tracer_kwargs=None
):
    all_activations_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting SAE activations"):
            text_batch = batch["sentence"]
            

def main(args):
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    sae_id = "jbrinkma/sae-llama-3-8b-layer16"
    model, submodule, autoencoder, _ = setup_model(model_id, sae_id=sae_id)
    sae_activations_dir = os.path.join(OUTPUTS_DIR, "sae_activations")
    tags_dir = os.path.join(OUTPUTS_DIR, "tags")
    ensure_dir(sae_activations_dir)
    ensure_dir(tags_dir)
    
    # load in PUD files
    for language in args.languages:
        pud_files = glob.glob(os.path.join(UD_BASE_FOLDER, f"UD_{language}-PUD", f"*_pud-ud-test.conllu"))
        pud_file = pud_files[0]
        dataset = ConlluDataset(pud_file)

        # test with small dataset
        dataset = Subset(dataset, range(64))

        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        extract_and_save_data(
            model=model,
            submodule=submodule,
            autoencoder=autoencoder,
            dataloader=dataloader,
            sae_activations_dir=sae_activations_dir,
            tags_dir=tags_dir,
            pooling_strategy="max",
            tracer_kwargs=None,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--languages", type=str, nargs="+", default=["English"], help="A list of languages to collect input space for.")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size for the dataloader.")
    # parser.add_argument("--sae_activations_dir", type=str, default="outputs/sae_activations", help="Directory to save the HDF5 files.")
    # parser.add_argument("--tags_dir", type=str, default="outputs/tags", help="Directory to save the JSONL files.")
    args = parser.parse_args()

    main(args)
