import torch
from nnsight import LanguageModel
from datasets import load_dataset
from torch.utils.data import DataLoader

from huggingface_hub import hf_hub_download
from lang_probing_src.autoencoder import GatedAutoEncoder
from lang_probing_src.config import NAME_TO_LANG_CODE

import joblib
import numpy as np

def setup_autoencoder():
    sae_id = "jbrinkma/sae-llama-3-8b-layer16"
    sae_filename = "llama-3-8b-layer16.pt"
    sae_path = hf_hub_download(repo_id=sae_id, filename=sae_filename)
    autoencoder = GatedAutoEncoder.from_pretrained(sae_path)
    return autoencoder

# --- Helper: Encapsulate the messy index logic ---
def get_last_token_mask(inputs, texts):
    mask = torch.zeros_like(inputs.input_ids, dtype=torch.bool)
    
    # Loop through batch to find word boundaries (last token of each word)
    word_ids_batch = [inputs.word_ids(i) for i in range(len(texts))]
    for b, w_ids in enumerate(word_ids_batch):
        # Shifted comparison: is current word_id different from next?
        for i in range(len(w_ids) - 1):
            if w_ids[i] is not None and w_ids[i] != w_ids[i+1]:
                mask[b, i] = True
        # Handle the very last token of the sequence
        if w_ids[-1] is not None: mask[b, len(w_ids)-1] = True
    return mask
    tracer_kwargs = {'scan': False, 'validate': False}
    with torch.no_grad():
        all_positive = []
        all_negative = []
        for batch in dataloader:
            inputs = tokenizer(batch["sentence"], padding=True, return_tensors="pt")
            extraction_mask = get_last_token_mask(inputs, batch["sentence"]) # shape: [batch_size, seq_len]

            with model.trace(batch["sentence"], **tracer_kwargs) as tracer:
                hidden_layer = model.model.layers[16].output
                acts = hidden_layer[extraction_mask].save() # shape: [seq_len, hidden_dim]

            # acts = acts.value
            print(acts.shape)

            # --- Post-Processing (Probe -> SAE -> Diff) ---
            # Acts shape: [N_Words, 4096]
            activations = acts.to(model.device) # Move back to GPU if .save() moved it

            # A. Get Probe Logits (Concept Present vs Absent)
            # Replace with: probe_logits = my_probe(activations)
            
            activations_cpu = activations.cpu().numpy()
            probe_logits = probe.predict_proba(activations_cpu)[:, 1]  # Probabilidad de la clase positiva
            probe_logits = torch.from_numpy(probe_logits).to(model.device)

            concept_mask = probe_logits > 0.5  # Umbral de 0.5 para probabilidad

            # B. Get SAE Features 
            # Replace with: sae_acts = my_sae.encode(activations)
            sae_acts = sae.encode(activations) # shape: [n_words, sae_dim]

            print(sae_acts.nonzero().shape)
            print(sae_acts)
            exit()

            # C. Calculate Difference in Means
            # Use the boolean mask to slice the SAE activations directly
            mean_present = sae_acts[concept_mask].mean(dim=0)
            mean_absent  = sae_acts[~concept_mask].mean(dim=0)
            all_positive.append(mean_present.cpu().numpy())
            all_negative.append(mean_absent.cpu().numpy())
        
        # diff_vector = mean_present - mean_absent # shape: [sae_dim]
        # all_diff_vectors.append(diff_vector.cpu().numpy())

    return diff_vector


# def input_features_treebank

def input_features_probe(model, sae, tokenizer, dataloader, probe):
    tracer_kwargs = {'scan': False, 'validate': False}
    with torch.no_grad():
        all_positive = []
        all_negative = []
        for batch in dataloader:
            sentence_batch = batch["sentence"]

            inputs = tokenizer(sentence_batch, padding=True, return_tensors="pt")
            extraction_mask = get_last_token_mask(inputs, sentence_batch) # shape: [batch_size, seq_len]

            with model.trace(sentence_batch, **tracer_kwargs) as tracer:
                hidden_layer = model.model.layers[16].output # remember that nnsight v0.5 changed the output to output instead of output[0]
                acts = hidden_layer[extraction_mask].save() # shape: [seq_len, hidden_dim]

            # A. Get Probe Logits (Concept Present vs Absent)
            acts_cpu = acts.cpu().numpy()
            probe_logits = probe.predict_proba(acts_cpu)[:, 1]  # Probabilidad de la clase positiva
            probe_logits = torch.from_numpy(probe_logits).to(model.device)
            concept_mask = probe_logits > 0.5  # Umbral de 0.5 para probabilidad

            all_positive.append(acts[concept_mask].cpu())
            all_negative.append(acts[~concept_mask].cpu())
        
        # D. Compute global means after processing all batches
        if all_positive:
            mean_present = torch.cat(all_positive, dim=0).mean(dim=0)  # shape: [sae_dim]
        else:
            raise ValueError(f"No positive examples found for {dataloader}")
        
        if all_negative:
            mean_absent = torch.cat(all_negative, dim=0).mean(dim=0)  # shape: [sae_dim]
        else:
            raise ValueError(f"No negative examples found for {dataloader}")
        
        # E. Calculate final difference vector
        diff_vector = mean_present - mean_absent  # shape: [sae_dim] -> [32768]

    return diff_vector

def generate_heatmap(all_diff_vectors):
    pass

def main():
    model = LanguageModel("meta-llama/Meta-Llama-3.1-8B-Instruct", device_map="cuda")
    sae = setup_autoencoder()
    sae.to("cuda")

    tokenizer = model.tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    LANGUAGES = ["English", "French", "German", "Spanish", "Turkish", "Arabic", "Hindi", "Hebrew", "Chinese", "Indonesian"]
    LANGUAGES = ["English", "French", "German", "Spanish", "Turkish"]

    
    all_diff_vectors = {}
    for language in LANGUAGES:
        dataset = load_dataset("gsarti/flores_101", NAME_TO_LANG_CODE[language], split="devtest")
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        probe_path = f"outputs/probes/word_probes/{language}_Tense_Past_l16_n1024.joblib"
        probe = joblib.load(probe_path)
        diff_vector = input_features_probe(model, sae, tokenizer, dataloader, probe)
        all_diff_vectors[language] = diff_vector.cpu().numpy()

if __name__ == "__main__":
    main()
