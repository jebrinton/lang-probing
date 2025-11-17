import sys
import os
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tqdm import tqdm
# from nnsight import LanguageModel
from torch.utils.data import DataLoader
import torch
import glob
from transformers import AutoModel, AutoTokenizer
import pyconll
import joblib
import numpy as np
import pandas as pd
import random
import logging
from cuml.common import logger as cuml_logger
from lang_probing_src.word_probing_utils import WordProbingDataset, WordProbingCollate, train_and_evaluate_probe
from lang_probing_src.config import PROBES_DIR, UD_BASE_FOLDER
from lang_probing_src.data import get_available_concepts, get_training_files


def conllu_to_processed_sentences(conll_filepaths):
    """
    Convert a list of conllu filepaths to a list of processed sentences.
    Args:
        conll_filepaths (list[str]): List of conllu filepaths.
    Returns:
        list[list[dict]]: List of processed sentences. List of sentences, each sentence is a list of word dictionaries.
    """
    processed_sentences = []
    for conll_file in tqdm(conll_filepaths, desc=f"Parsing conllu files"):
        for sentence in pyconll.load.iter_from_file(conll_file):
            processed_ids = set()
            processed_sentence = []
            for i, word in enumerate(sentence):
                # already processed via MWT
                if word.id in processed_ids:
                    continue

                word_dict = {"id": word.id, "form": word.form, "feats": word.feats}

                if word.is_multiword():
                    # kinda some weird logic here
                    # first we get a tuple of the MWT span as ints,
                    # the convert back to strings to index into the sentence
                    # this is important! 
                    # in conlllu, sentence[i] ≠ sentence[str(i)] due to multiword tokens
                    span = tuple(int(x) for x in word.id.split("-"))
                    start, end = span
                    span_ids = [str(i) for i in range(start, end + 1)]
                    for id in span_ids:
                        word_dict["feats"].update(sentence[id].feats) # possible TODO: ensure that MWT feats can have multiple values
                        processed_ids.add(id)
                processed_sentence.append(word_dict)

            processed_sentences.append(processed_sentence)
    return processed_sentences


def get_processed_sentences(language, split="train"):
    """
    Get the processed sentences for a given language.
    Args:
        language (str): Language code.
        split (str): Split to get sentences for.
    Returns:
        list[list[dict]]: List of processed sentences.
    """
    cache_filename = f"{language}_{split}_processed.joblib"
    cache_path = os.path.join(PROBES_DIR, "processed_sentences", cache_filename)

    # 1. Check if cache exists
    if os.path.exists(cache_path):
        print(f"Loading {language}:{split} sentences from cache...")
        processed_sentences = joblib.load(cache_path)
        return processed_sentences

    # 2. If cache doesn't exist, run parsing code
    print(f"Parsing {language}:{split} from .conllu files (this may take a while)...")
    conll_filepaths = glob.glob(os.path.join(UD_BASE_FOLDER, f"UD_{language}*", f"*-ud-{split}.conllu"))
    processed_sentences = conllu_to_processed_sentences(conll_filepaths)
    joblib.dump(processed_sentences, cache_path)
    return processed_sentences


def extract_word_activations(hf_model, dataloader, layer_num, device="cuda"):
    all_word_activations = []
    all_word_labels = []
    
    for tokenized_batch, batch_word_labels in tqdm(dataloader, desc="Extracting activations"):
        batch_word_ids = [tokenized_batch.word_ids(i) for i in range(len(batch_word_labels))]

        tokenized_batch = {k: v.to(device) for k, v in tokenized_batch.items()}
        with torch.no_grad():
            outputs = hf_model(**tokenized_batch, output_hidden_states=True)
            
        acts = outputs.hidden_states[layer_num] # [batch_size, seq_len, hidden_dim]

        # 4. Loop, align, and pool
        for i in range(len(batch_word_labels)): # Loop over sentences
            sentence_acts = acts[i]
            sentence_word_ids = batch_word_ids[i]
            sentence_labels = batch_word_labels[i]
            
            num_words_in_sentence = len(sentence_labels)

            for word_index in range(num_words_in_sentence): # Loop over words
                
                # Find all tokens for this word
                token_span = [j for j, wid in enumerate(sentence_word_ids) 
                                if wid == word_index]
                
                if not token_span:
                    continue # Word was truncated or lost
                    
                # Get all token activations for this word
                token_activations = sentence_acts[token_span]
                
                # Mean-pool the tokens
                word_activation = token_activations.mean(dim=0)
                
                all_word_activations.append(word_activation.float().cpu().numpy())
                all_word_labels.append(sentence_labels[word_index])
    
    # Stack all word activations into a 2D array [num_total_words, hidden_dim]
    return np.vstack(all_word_activations), np.array(all_word_labels)


def train_probe(model, tokenizer, language, concept, value, layer_num=16, max_samples=None):
    train_sentences = get_processed_sentences(language, "train")
    test_sentences = get_processed_sentences(language, "test")

    random.seed(42)
    if max_samples is not None:
        train_sentences = random.sample(train_sentences, max_samples)
        test_sentences = random.sample(test_sentences, max_samples)
    
    train_dataset = WordProbingDataset(train_sentences, concept, value)
    test_dataset = WordProbingDataset(test_sentences, concept, value)
    
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False, collate_fn=WordProbingCollate(tokenizer))
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=WordProbingCollate(tokenizer))
    
    train_word_acts, train_word_labels = extract_word_activations(model, train_dataloader, layer_num)
    test_word_acts, test_word_labels = extract_word_activations(model, test_dataloader, layer_num)
    
    classifier, stats = train_and_evaluate_probe(
        train_word_acts, train_word_labels, 
        test_word_acts, test_word_labels, 
        seed=42
    )
    return classifier, stats # Return both


# processed_sentences = get_processed_sentences("Spanish", "train")
# dataset = WordProbingDataset(processed_sentences, "Number", "Plur")
# subset_indices = range(64)
# dataset = Subset(dataset, subset_indices)

# dataloader = DataLoader(
#     dataset, 
#     batch_size=16, 
#     shuffle=False, 
#     collate_fn=WordProbingCollate(tokenizer)
# )

# word_acts, word_labels = extract_word_activations(my_hf_model, dataloader, layer_num)

# # save word_acts and word_labels to a file
# np.savez("word_acts_and_labels.npz", word_acts=word_acts, word_labels=word_labels)

# # load word_acts and word_labels from a file
# word_acts, word_labels = np.load("word_acts_and_labels.npz")

# classifier = train_and_evaluate_probe(word_acts, word_labels, word_acts, word_labels, 42)

# classifier.fit(train_activations, train_labels)

# train_accuracy = classifier.score(word_acts, word_labels)
# test_accuracy = classifier.score(word_acts, word_labels)

# print(f"Train Accuracy: {train_accuracy:.2f}")
# print(f"Test Accuracy: {test_accuracy:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Train word probes for linguistic concepts")
    parser.add_argument("concept", help="The linguistic concept to probe for (e.g., Gender, Number, Tense)")
    args = parser.parse_args()

    ALL_CONCEPTS_VALUES = {
        "Number": ["Sing", "Dual", "Plur"],
        "Tense": ["Past", "Pres", "Fut"],
        "Gender": ["Masc", "Fem", "Neut"],
        "Polite": ["Infm", "Form"],
        "Case": ["Nom", "Acc", "Gen", "Dat", "Loc"],
    }

    # Filter to only the requested concept
    if args.concept not in ALL_CONCEPTS_VALUES:
        raise ValueError(f"Concept '{args.concept}' not found. Available concepts: {list(ALL_CONCEPTS_VALUES.keys())}")

    # config logging
    logging.basicConfig(level=logging.INFO)
    cuml_logger.set_level(logging.ERROR)

    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    my_hf_model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    CONCEPTS_VALUES = {args.concept: ALL_CONCEPTS_VALUES[args.concept]}
    LANGUAGES = ["English", "French", "German", "Spanish", "Turkish", "Arabic", "Chinese"]
    LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 32]
    max_samples = 1024

    all_results = []

    for concept in CONCEPTS_VALUES.keys():
        for value in CONCEPTS_VALUES[concept]:
            for language in LANGUAGES:
                # check that the concept and value are valid for the language
                if concept not in get_available_concepts(get_training_files(language, ud_base_folder=UD_BASE_FOLDER)):
                    continue
                if value not in get_available_concepts(get_training_files(language, ud_base_folder=UD_BASE_FOLDER))[concept]:
                    continue
                for layer_num in LAYERS:
                    logging.info(f"Training probe for {language} {concept}={value} at layer {layer_num}...")
                    
                    # 1. Get classifier and stats
                    classifier, stats = train_probe(my_hf_model, tokenizer, language, concept, value, layer_num, max_samples)
                    
                    # 2. Save the probe file
                    probe_filename = f"outputs/probes/word_probes/{language}_{concept}_{value}_l{layer_num}_n{max_samples}.joblib"
                    joblib.dump(classifier, probe_filename)
                    logging.info(f"Saved probe to {probe_filename}")
                    
                    # 3. Create a full record for our CSV
                    record = {
                        "language": language,
                        "concept": concept,
                        "value": value,
                        "layer": layer_num,
                        "n_samples": max_samples,
                        "probe_file": probe_filename, # Link to the saved model
                        **stats # Unpack the stats dict (train_acc, test_acc, etc.)
                    }
                    
                    # 4. Add the record to our list
                    all_results.append(record)

    logging.info("\nAll probes trained. Saving results to CSV...")
    
    # Convert the list of dictionaries to a pandas DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Re-order columns for readability
    columns = [
        "language", "concept", "value", "layer", "n_samples", 
        "train_accuracy", "test_accuracy", "cv_score", 
        "best_params", "probe_file"
    ]
    # Make sure we only use columns that exist
    final_columns = [col for col in columns if col in results_df.columns]
    results_df = results_df[final_columns]
    
    # Define the save path
    results_csv_path = f"outputs/probes/all_probe_results_{args.concept.lower()}.csv"
    results_df.to_csv(results_csv_path, index=False)
    
    logging.info(f"Results saved to {results_csv_path}")
    logging.info("\n--- Final Results Summary ---")
    logging.info(results_df.to_string())

if __name__ == "__main__":
    main()