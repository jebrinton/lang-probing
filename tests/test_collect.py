"""
Test script for activation collection system.
Tests on a small subset (English and Spanish, 10 sentences each, 2 layers).
"""

import os
import sys
import logging
import pandas as pd
import json
from functools import partial
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import ACTIVATIONS_DIR, SEED
from src.data import load_sentences_with_tags
from src.activations import collect_sentence_activations
from src.utils import setup_model, ensure_dir
from src.sentence_dataset_class import SentenceDataset, collate_fn

def test_activation_collection():
    """Test the activation collection pipeline on a small subset."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.info("Starting test activation collection...")
    
    # Test parameters
    test_languages = ["English", "Spanish"]
    test_layers = [0, 16]
    max_sentences_per_language = 10
    batch_size = 4
    
    # Load model
    logging.info("Loading model...")
    model, _, _, tokenizer = setup_model(
        model_id="meta-llama/Llama-3.1-8B",
        sae_id=None
    )
    
    # Collect sentences from all test languages
    logging.info("Loading sentences...")
    all_sentences = []
    for language in test_languages:
        sentences = load_sentences_with_tags(
            language=language,
            max_samples=max_sentences_per_language,
            seed=SEED,
            quickly=True
        )
        all_sentences.extend(sentences)
        logging.info(f"Loaded {len(sentences)} sentences for {language}")
    
    logging.info(f"Total sentences: {len(all_sentences)}")
    
    # Create dataloader
    logging.info("Creating dataloader...")
    dataset = SentenceDataset(all_sentences, tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=partial(collate_fn, tokenizer=tokenizer)
    )
    
    # Extract activations
    logging.info("Extracting activations...")
    results = collect_sentence_activations(
        model=model,
        dataloader=dataloader,
        layers=test_layers
    )
    
    logging.info(f"Collected {len(results)} activation records")
    
    # Convert to DataFrame
    logging.info("Creating DataFrame...")
    df_data = []
    for record in results:
        df_data.append({
            'sentence_id': record['sentence_id'],
            'language': record['language'],
            'sentence_text': record['sentence_text'],
            'layer': record['layer'],
            'tags': json.dumps(record['tags']),  # Serialize dict to JSON string
            'activation': record['activation']
        })
    
    df = pd.DataFrame(df_data)
    
    # Validate results
    logging.info("Validating results...")
    
    # Check row count
    expected_rows = len(all_sentences) * len(test_layers)
    assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}"
    logging.info(f"✓ Row count correct: {len(df)} rows")
    
    # Check activation shape
    sample_activation = df.iloc[0]['activation']
    assert sample_activation.shape == (4096,), f"Expected (4096,), got {sample_activation.shape}"
    logging.info(f"✓ Activation shape correct: {sample_activation.shape}")
    
    # Check tags format
    for idx, row in df.iterrows():
        tags_str = str(row['tags'])
        tags = json.loads(tags_str)
        assert isinstance(tags, dict), f"Tags should be dict, got {type(tags)}"
        for key, values in tags.items():
            assert isinstance(values, list), f"Tag values should be list, got {type(values)}"
    logging.info("✓ Tags format correct")
    
    # Check languages
    assert set(df['language'].unique()) == set(test_languages), "Language mismatch"
    logging.info(f"✓ Languages correct: {df['language'].unique()}")
    
    # Check layers
    assert set(df['layer'].unique()) == set(test_layers), "Layer mismatch"
    logging.info(f"✓ Layers correct: {df['layer'].unique()}")
    
    # Save test output
    test_output_dir = os.path.join(ACTIVATIONS_DIR, "test")
    ensure_dir(test_output_dir)
    output_path = os.path.join(test_output_dir, "activations_test.parquet")
    
    logging.info(f"Saving to {output_path}...")
    df.to_parquet(output_path, compression='snappy', index=False)
    
    # Verify we can load it back
    df_loaded = pd.read_parquet(output_path)
    assert len(df_loaded) == len(df), "Loaded DataFrame has different length"
    logging.info("✓ Successfully saved and loaded Parquet file")
    
    # Print summary
    logging.info("\n" + "="*50)
    logging.info("TEST SUMMARY")
    logging.info("="*50)
    logging.info(f"Languages: {test_languages}")
    logging.info(f"Sentences per language: {max_sentences_per_language}")
    logging.info(f"Total sentences: {len(all_sentences)}")
    logging.info(f"Layers: {test_layers}")
    logging.info(f"Total records: {len(df)}")
    logging.info(f"Activation shape: {sample_activation.shape}")
    logging.info(f"Output file: {output_path}")
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logging.info(f"File size: {file_size_mb:.2f} MB")
    logging.info("="*50)
    logging.info("✓ All tests passed!")
    
    return df

if __name__ == "__main__":
    test_activation_collection()

