import torch
from nnsight import LanguageModel
# from datasets import load_dataset
from einops import reduce

from tqdm import tqdm
from pathlib import Path
import logging

from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download
from lang_probing_src.autoencoder import GatedAutoEncoder
from lang_probing_src.config import NAME_TO_LANG_CODE
from lang_probing_src.data import ConlluDataset, ConlluDatasetPooled, collate_fn



def setup_autoencoder():
    sae_id = "jbrinkma/sae-llama-3-8b-layer16"
    sae_filename = "llama-3-8b-layer16.pt"
    sae_path = hf_hub_download(repo_id=sae_id, filename=sae_filename)
    autoencoder = GatedAutoEncoder.from_pretrained(sae_path)
    return autoencoder


def load_diff_vector(language, concept, value):
    output_dir = Path("/projectnb/mcnet/jbrin/lang-probing/outputs/sentence_input_features")
    save_dir = output_dir / language / concept / value
    save_path = save_dir / "diff_vector.pt"

    if not save_path.exists():
        logging.error(f"Diff vector for {language}-{concept}-{value} not found at {save_path}")
        return None

    return torch.load(save_path)


def get_input_feature_vector(model, sae, dataloader):
    with torch.no_grad():
        all_sae_acts = []
        for batch in tqdm(dataloader, desc="Extracting SAE activations"):
            sentence_batch = batch["sentence"]

            with model.trace(sentence_batch):
                acts = model.model.layers[16].output.save() # shape: [batch, seq, hidden_dim]

            sae_acts = sae.encode(acts) # shape: [batch, seq, sae_dim]

            sent_sae_acts = reduce(sae_acts, "b s d -> d", "mean").to("cpu") # shape: [sae_dim]
            all_sae_acts.append(sent_sae_acts)
        
        all_sae_acts = torch.stack(all_sae_acts, dim=0) # shape: [batches, sae_dim]
        mean = all_sae_acts.mean(dim=0) # shape: [sae_dim]
        return mean


def main():
    logging.basicConfig(level=logging.INFO)
    
    output_dir = Path("/projectnb/mcnet/jbrin/lang-probing/outputs/sentence_input_features")
    output_dir.mkdir(exist_ok=True)

    model = LanguageModel("meta-llama/Meta-Llama-3.1-8B-Instruct", device_map="cuda")
    sae = setup_autoencoder()
    sae.to("cuda")

    tokenizer = model.tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    LANGUAGES = ["English", "French", "German", "Spanish", "Turkish", "Arabic", "Hindi", "Hebrew", "Chinese", "Indonesian"]

    CONCEPTS_VALUES = {"Tense": ["Past", "Pres", "Fut"], "Number": ["Sing", "Dual", "Plur"]}

    # for testing
    LANGUAGES = ["English", "French", "German", "Spanish", "Turkish"]
    LANGUAGES = ["Arabic", "Hindi", "Chinese", "Indonesian"]
    CONCEPTS_VALUES = {"Tense": ["Past", "Pres", "Fut"], "Number": ["Sing", "Dual", "Plur"]}
    batch_size = 64

    for language in LANGUAGES:
        dataset = ConlluDatasetPooled(language, treebank="PUD")
        for concept in CONCEPTS_VALUES.keys():
            for value in CONCEPTS_VALUES[concept]:

                save_dir = output_dir / language / concept / value
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / "diff_vector.pt"

                concept_filter = lambda sentence: concept in sentence["tags"].keys() and value in sentence["tags"][concept]
                positive_dataset = dataset.filter(concept_filter)
                negative_dataset = dataset.filter(lambda sentence: not concept_filter(sentence))

                if len(positive_dataset) == 0:
                    logging.info(f"Not enough positive data for {language}-{concept}-{value}. Skipping.")
                    continue
                if len(negative_dataset) == 0:
                    logging.info(f"Not enough negative data for {language}-{concept}-{value}. Skipping.")
                    continue

                logging.info(f"Number of positive samples for {language}-{concept}-{value} is {len(positive_dataset)}")
                logging.info(f"Number of negative samples for {language}-{concept}-{value} is {len(negative_dataset)}")

                positive_dataloader = DataLoader(positive_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
                negative_dataloader = DataLoader(negative_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

                positive_vector = get_input_feature_vector(model, sae, positive_dataloader)
                negative_vector = get_input_feature_vector(model, sae, negative_dataloader)

                diff_vector = positive_vector - negative_vector

                torch.save(diff_vector, save_path)
                logging.info(f"Saved diff vector for {language}-{concept}-{value} to {save_path}")


if __name__ == "__main__":
    main()