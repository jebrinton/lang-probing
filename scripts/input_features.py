import torch
from nnsight import LanguageModel
from datasets import load_dataset
from torch.utils.data import DataLoader

from huggingface_hub import hf_hub_download
from lang_probing_src.autoencoder import GatedAutoEncoder
from lang_probing_src.config import NAME_TO_LANG_CODE
from lang_probing_src.data import ConlluDataset, FloresDataset, collate_fn

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


def get_positive_negative_mask_from_probe(probe, acts_cpu):
    probe_logits = probe.predict_proba(acts_cpu)[:, 1]  # Probabilidad de la clase positiva
    probe_logits = torch.from_numpy(probe_logits).to(model.device)
    concept_mask = probe_logits > 0.5  # Umbral de 0.5 para probabilidad
    return concept_mask, ~concept_mask


def get_positive_negative_mask(tags_batch):
    positive_mask = tags_batch == "positive"
    negative_mask = tags_batch == "negative"
    return positive_mask, negative_mask


def input_features(model, sae, dataloader, probe=None):
    tracer_kwargs = {'scan': False, 'validate': False}
    with torch.no_grad():
        all_positive = []
        all_negative = []
        for batch in dataloader:
            sentence_batch = batch["sentence"]
            tags_batch = batch["tags"]

            print(len(sentence_batch), len(tags_batch))
            for sentence, tags in zip(sentence_batch, tags_batch):
                for i in range(len(sentence)):
                    print(f"{sentence[i]} {tags[i]}")

            inputs = model.tokenizer(sentence_batch, padding=True, return_tensors="pt")
            extraction_mask = get_last_token_mask(inputs, sentence_batch) # shape: [batch_size, seq_len]

            with model.trace(sentence_batch, **tracer_kwargs) as tracer:
                hidden_layer = model.model.layers[16].output # remember that nnsight v0.5 changed the output to output instead of output[0]
                acts = hidden_layer[extraction_mask].save() # shape: [seq_len, hidden_dim]

            if probe is not None:
                positive_mask, negative_mask = get_positive_negative_mask_from_probe(probe, acts.cpu().numpy())
            elif tags_batch is not None:
                positive_mask, negative_mask = get_positive_negative_mask(tags_batch)
            else:
                raise ValueError("Either tags_batch or probe must be provided")

            sae_acts = sae.encode(acts)

            all_positive.append(sae_acts[positive_mask].cpu())
            all_negative.append(sae_acts[negative_mask].cpu())
        
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
        # dataset = FloresDataset(language, split="devtest")
        dataset = ConlluDataset(language, treebank="PUD")

        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

        diff_vector = input_features(model, sae, dataloader)
        all_diff_vectors[language] = diff_vector.cpu().numpy()

if __name__ == "__main__":
    main()
