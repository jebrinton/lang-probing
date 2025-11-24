"""
Script to process Flores dataset with SAE feature extraction

Usage:
    python scripts/process_flores.py [--languages LANG1,LANG2] [--max_samples N] [--batch_size N]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import logging
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from nnsight import LanguageModel
import pickle
from sklearn.pipeline import Pipeline
import glob

from src.config import (
    MODEL_ID, SAE_ID, SAE_FILENAME, LAYER_NUM, ACTIVATIONS_DIR, LOGS_DIR, 
    TRACER_KWARGS, BATCH_SIZE as DEFAULT_BATCH_SIZE, PROBES_DIR
)
from src.utils import ensure_dir, setup_logging, save_json
from src.activations import ActivationDataset
from src.probe import load_probe
from src.autoencoder import GatedAutoEncoder
from src.attribution import attribution_patching, attribution_patching_per_token


LANGUAGES = ["English", "French", "German", "Spanish", "Turkish", "Arabic", "Hindi", "Hebrew", "Chinese", "Indonesian"]
CONCEPTS = ["Tense", "Number"]

def to_numpy(x):
    # Handles cuML / CuPy / NumPy / Python scalars
    try:
        # cuML objects often have this
        if hasattr(x, "to_output"):
            return x.to_output("numpy")
    except TypeError:
        pass

    # CuPy array
    try:
        import cupy as cp
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    except ImportError:
        pass

    # Fall back to normal NumPy conversion
    return np.asarray(x)


class LogisticRegressionPyTorch(nn.Module):
    def __init__(self, n_features):
        super(LogisticRegressionPyTorch, self).__init__()
        self.linear = nn.Linear(n_features, 1) 

    def forward(self, x):
        return self.linear(x) 
    

def convert_probe_to_pytorch(probe_pipeline):
    # Expecting a Pipeline(scaler, model)
    scaler = probe_pipeline.named_steps['scaler']
    probe  = probe_pipeline.named_steps['model']

    # Convert everything to NumPy
    W    = to_numpy(probe.coef_).ravel()        # (n_features,)
    b    = float(to_numpy(probe.intercept_))    # scalar
    mean = to_numpy(scaler.mean_)               # (n_features,)
    std  = to_numpy(scaler.scale_)              # (n_features,)

    # Fuse StandardScaler into weights:
    # x_scaled = (x - mean) / std
    # z = W·x_scaled + b  =>  W_new = W/std,  b_new = b - sum(W*mean/std)
    W_new = W / std
    b_new = b - np.sum(W * mean / std)

    torch_probe = LogisticRegressionPyTorch(len(W_new))

    with torch.no_grad():
        torch_probe.linear.weight.copy_(torch.tensor(W_new, dtype=torch.float32).unsqueeze(0))
        torch_probe.linear.bias.copy_(torch.tensor([b_new], dtype=torch.float32))

    return torch_probe.to("cuda")


def setup_autoencoder(repo_id, filename, device="cuda"):
    dict = GatedAutoEncoder.from_hub(repo_id=repo_id, filename=filename)
    dict.to(device)
    return dict


def load_flores_dataset(language="eng", split="dev"):
    logging.info(f"Loading Flores dataset (split: {split})")
    
    dataset = load_dataset("gsarti/flores_101", language, split=split)
    logging.info(f"Loaded {len(dataset)} samples from gsarti/flores_101")
    
    # Test the dataset structure
    sample = dataset[0]
    logging.info(f"Dataset sample keys: {list(sample.keys())}")
    
    return dataset


def attribution_patching_loop(dataset, model, torch_probe, sae_submodule, probe_submodule, autoencoder):
    """Perform attribution patching on the dataset."""
    effects = {}
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, example in enumerate(tqdm(dataloader, desc="Attribution patching")):
        tokens = model.tokenizer(example["sentence"][0], return_tensors="pt", padding=False)
        e, _, _, _ = attribution_patching_per_token(tokens["input_ids"], model, torch_probe, probe_submodule, [sae_submodule], {sae_submodule: autoencoder})
        if probe_submodule not in effects:
            effects[probe_submodule] = e[probe_submodule].sum(dim=0)
        else:
            effects[probe_submodule] += e[probe_submodule].sum(dim=0)

    return effects


def main(args):
    """Main processing loop"""
    
    # Setup logging
    setup_logging(LOGS_DIR, 'process_flores.log')

    output_dir = os.path.join(ACTIVATIONS_DIR, "flores")
    ensure_dir(output_dir)
    
    # Load dataset
    dataset = load_flores_dataset(language=args.language, split="dev")

    # Optional: Subsample dataset to max. samples
    if args.max_samples:
        dataset = dataset.select(range(args.max_samples))

    # Load model and tokenizer
    logging.info(f"Loading model: {MODEL_ID}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    model = LanguageModel(MODEL_ID, device_map=device, dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # Load SAE model
    logging.info(f"Loading SAE: {SAE_ID}")
    ae = setup_autoencoder(repo_id=SAE_ID, filename=SAE_FILENAME, device=device)

    for language in LANGUAGES:
        for concept in CONCEPTS:
            
            # Find all probe files for this language and concepts. Each concept might have multiple values (e.g. Tense_Pres, Tense_Past, Tense_Future, etc.)
            # The values should not be prespecified, but should be found in the probe files. The structure is {language}_{concept}_{value}_l{args.layer}_n{args.num_samples}.joblib
            probe_files = glob.glob(os.path.join(PROBES_DIR, f"{language}_{concept}_*_l{args.layer}_n{args.num_probe_samples}.joblib"))
            probe_files = [os.path.basename(f) for f in probe_files]
            probe_files = [f.split("_") for f in probe_files]
            probe_files = [f[2] for f in probe_files]
            probe_files = list(set(probe_files))
            logging.info(f"Found the following probe files: {probe_files} for language: {language} and concept: {concept}")

            for value in probe_files:

                logging.info(f"Processing value: {value} for language: {language} and concept: {concept}")

                # Load probe
                probe_path = os.path.join(PROBES_DIR, f"{language}_{concept}_{value}_l{args.layer}_n{args.num_probe_samples}.joblib")
                logging.info(f"Loading probe: {probe_path}")
                probe = load_probe(probe_path)
                torch_probe = convert_probe_to_pytorch(probe)

                # Perform attribution patching
                sae_submodule = model.model.layers[LAYER_NUM]
                probe_submodule = model.model.layers[LAYER_NUM]
                effects = attribution_patching_loop(dataset, model, torch_probe, sae_submodule, probe_submodule, ae)

                # Save effects
                save_json(
                    effects, 
                    os.path.join(output_dir, f"effects_{language}_{concept}_{value}_l{args.layer}_np{args.num_probe_samples}_ns{args.max_samples}.json")
                )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="eng")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--num_probe_samples", type=int, default=1024)
    parser.add_argument("--max_samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--output_dir", type=str, default="outputs/effects")
    args = parser.parse_args()
    main(args)
