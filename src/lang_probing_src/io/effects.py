import os
import glob
import torch
import numpy as np
from tqdm import tqdm


def get_output_features_vector(effects_files, language_pair, concept, value):
    e = effects_files[language_pair][concept][value]
    e = e.cpu().numpy()
    return np.mean(e, axis=0)


def get_input_features_vector(output_dir, language, concept, value):
    save_dir = output_dir / language / concept / value
    save_path = save_dir / "diff_vector.pt"

    return torch.load(save_path).numpy()


def load_effects_files(out_dir="/projectnb/mcnet/jbrin/lang-probing/outputs/attributions"):
    # Iterate through the subfolders, each corresponding to a language pair
    effects_files = {}
    for subfolder in tqdm(os.listdir(out_dir)):
        if os.path.isdir(os.path.join(out_dir, subfolder)):

            # Find effects_*.pt file
            effects_file = glob.glob(os.path.join(out_dir, subfolder, "effects_*.pt"))
            if len(effects_file) == 0:
                raise ValueError(f"No effects file found for {subfolder}")
            effects_file = effects_file[0]
        
            # The suffix is a language pair, such as effects_English_French.pt
            source_lang, target_lang = effects_file.split("/")[-1].split("_")[1:]
            target_lang, _ = target_lang.split(".")
            language_pair = (source_lang, target_lang)

            print(f"Loading effects file for {language_pair}")

            # Load the effects file and, if necessary, overwrite with more recent version
            # Make sure they're on CPU
            effects = torch.load(effects_file)
            effects_files[language_pair] = effects

    return effects_files


def get_language_pairs_and_concepts(effects_files):
    language_pairs = set()
    concepts = {}
    for language_pair in effects_files:
        language_pairs.add(language_pair)
        for concept_key in effects_files[language_pair]:
            if concept_key not in concepts:
                concepts[concept_key] = set()
            for concept_value in effects_files[language_pair][concept_key]: 
                concepts[concept_key].add(concept_value)
    
    return sorted(language_pairs), concepts
