"""
Main script for collecting mean activations from all languages.
Stores results in a single Parquet file for cross-language analysis.
"""

import os
import sys
import logging
import pandas as pd
import json
from datetime import datetime
from functools import partial
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import (
    ACTIVATIONS_DIR, LANGUAGES, MODEL_ID, SEED,
    MAX_SAMPLES_FOR_STEERING, COLLECTION_LAYERS, BATCH_SIZE
)
from src.data import load_sentences_with_tags
from src.activations import collect_sentence_activations
from src.utils import setup_model, ensure_dir, setup_logging
from src.sentence_dataset_class import SentenceDataset, collate_fn

def main():
    """Main function to collect activations for all languages."""
    
    # Setup logging
    log_filename = f"collect_activations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(os.path.join(os.path.dirname(__file__), '..', 'logs'), log_filename)
    
    logging.info("="*60)
    logging.info("ACTIVATION COLLECTION PIPELINE")
    logging.info("="*60)
    logging.info(f"Model: {MODEL_ID}")
    logging.info(f"Languages: {len(LANGUAGES)}")
    logging.info(f"Layers: {COLLECTION_LAYERS}")
    logging.info(f"Max samples per language: {MAX_SAMPLES_FOR_STEERING}")
    logging.info(f"Batch size: {BATCH_SIZE}")
    logging.info("="*60)
    
    # Load model
    logging.info("\nLoading model...")
    model, submodule, autoencoder, tokenizer = setup_model(
        model_id=MODEL_ID,
        sae_id=None  # No SAE needed for this task
    )
    logging.info("✓ Model loaded successfully")
    
    # Collect sentences from all languages
    logging.info(f"\nLoading sentences from {len(LANGUAGES)} languages...")
    all_sentences = []
    language_stats = {}
    
    for language in LANGUAGES:
        logging.info(f"\nProcessing {language}...")
        try:
            sentences = load_sentences_with_tags(
                language=language,
                max_samples=MAX_SAMPLES_FOR_STEERING,
                seed=SEED
            )
            all_sentences.extend(sentences)
            language_stats[language] = len(sentences)
            logging.info(f"✓ {language}: {len(sentences)} sentences")
        except Exception as e:
            logging.error(f"✗ Error loading {language}: {e}")
            language_stats[language] = 0
    
    total_sentences = len(all_sentences)
    logging.info(f"\n✓ Total sentences loaded: {total_sentences}")
    
    if total_sentences == 0:
        logging.error("No sentences loaded. Exiting.")
        return
    
    # Create dataloader
    logging.info("\nCreating dataloader...")
    dataset = SentenceDataset(all_sentences, tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        collate_fn=partial(collate_fn, tokenizer=tokenizer)
    )
    
    # Extract activations
    logging.info(f"\nExtracting activations for {len(COLLECTION_LAYERS)} layers...")
    logging.info(f"Expected total records: {total_sentences * len(COLLECTION_LAYERS)}")
    
    results = collect_sentence_activations(
        model=model,
        dataloader=dataloader,
        layers=COLLECTION_LAYERS
    )
    
    logging.info(f"✓ Collected {len(results)} activation records")
    
    # Convert to DataFrame
    logging.info("\nCreating DataFrame...")
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
    logging.info(f"✓ DataFrame created with {len(df)} rows")
    
    # Save to Parquet
    ensure_dir(ACTIVATIONS_DIR)
    output_path = os.path.join(ACTIVATIONS_DIR, "activations_all.parquet")
    
    logging.info(f"\nSaving to {output_path}...")
    df.to_parquet(output_path, compression='snappy', index=False)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logging.info(f"✓ Saved successfully ({file_size_mb:.2f} MB)")
    
    # Print final summary
    logging.info("\n" + "="*60)
    logging.info("COLLECTION SUMMARY")
    logging.info("="*60)
    logging.info(f"Total languages processed: {len([v for v in language_stats.values() if v > 0])}/{len(LANGUAGES)}")
    logging.info(f"Total sentences: {total_sentences}")
    logging.info(f"Total activation records: {len(df)}")
    logging.info(f"Layers extracted: {COLLECTION_LAYERS}")
    logging.info(f"Output file: {output_path}")
    logging.info(f"File size: {file_size_mb:.2f} MB")
    
    logging.info("\nLanguage breakdown:")
    for language, count in sorted(language_stats.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            logging.info(f"  {language}: {count} sentences")
    
    logging.info("\nLayer breakdown:")
    for layer in COLLECTION_LAYERS:
        layer_count = len(df[df['layer'] == layer])
        logging.info(f"  Layer {layer}: {layer_count} records")
    
    logging.info("="*60)
    logging.info("✓ COLLECTION COMPLETE")
    logging.info("="*60)

if __name__ == "__main__":
    main()

