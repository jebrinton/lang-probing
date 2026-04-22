"""
Script principal para entrenar probes lineales sobre activaciones SAE

Usage:
    python scripts/train_probes.py [--languages LANGS] [--concepts CONCEPTS]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm

from lang_probing_src.config import (
    MODEL_ID, SAE_ID, LANGUAGES, CONCEPTS, BATCH_SIZE, SEED, 
    MAX_ITER, C_VALUE, LAYER_NUM, PROBES_DIR, LOGS_DIR
)
from lang_probing_src.utils import setup_model, ensure_dir, setup_logging
from lang_probing_src.data import get_training_files, get_test_files, get_available_concepts, ProbingDataset, balance_dataset
from lang_probing_src.activations import extract_mlp_activations
from lang_probing_src.probe import train_probe, evaluate_probe, save_probe




def train_probe_for_concept(model, language, concept_key, 
                            concept_value, output_dir, args):
    """
    Entrena un probe para un concepto específico.
    
    Args:
        model: LanguageModel
        language: Nombre del idioma
        concept_key: Clave del concepto (e.g., 'Tense')
        concept_value: Valor del concepto (e.g., 'Past')
        output_dir: Directorio de salida
        args: Argumentos del script
        
    Returns:
        dict: Resultados del entrenamiento
    """
    logging.info(f"Processing {language} - {concept_key}: {concept_value}")
    
    # Verificar si el probe ya existe
    probe_filename = f"{language}_{concept_key}_{concept_value}.joblib"
    probe_path = os.path.join(output_dir, probe_filename)
    
    if os.path.exists(probe_path) and not args.overwrite:
        logging.info(f"Probe already exists. Skipping.")
        return None
    
    # Obtener archivos de datos
    train_files = get_training_files(language)
    test_files = get_test_files(language)
    
    if not train_files:
        logging.warning(f"No training files found for {language}")
        return None
    
    logging.info(f"Found {len(train_files)} training files for {language}")
    for f in train_files:
        logging.info(f"  - {os.path.basename(f)}")
    
    # Crear datasets
    try:
        train_dataset = ProbingDataset(train_files, concept_key, concept_value)
        
        # Verificar que hay suficientes datos
        if len(train_dataset) < 10:
            logging.warning(f"Not enough samples in train set: {len(train_dataset)}")
            return None
        
        # Balancear dataset
        train_dataset_balanced = balance_dataset(train_dataset, seed=SEED)
        
        if train_dataset_balanced is None or len(train_dataset_balanced) < args.min_samples:
            logging.warning(f"Not enough balanced samples: {len(train_dataset_balanced) if train_dataset_balanced else 0}")
            return None
        
        # Crear DataLoader
        train_dataloader = DataLoader(
            train_dataset_balanced, 
            batch_size=BATCH_SIZE, 
            shuffle=True
        )
        
        # Extraer activaciones MLP
        logging.info("Extracting MLP activations...")
        train_activations, train_labels = extract_mlp_activations(
            model, train_dataloader, LAYER_NUM
        )
        
        # Entrenar probe
        logging.info("Training probe...")
        probe = train_probe(
            train_activations, train_labels, 
            seed=SEED, max_iter=MAX_ITER, C=C_VALUE
        )
        
        # Evaluar en train set
        train_accuracy = evaluate_probe(probe, train_activations, train_labels)
        logging.info(f"Train accuracy: {train_accuracy:.4f}")
        
        # Evaluar en test set si está disponible
        test_accuracy = None
        if test_files:
            logging.info(f"Found {len(test_files)} test files for {language}")
            for f in test_files:
                logging.info(f"  - {os.path.basename(f)}")
                
            test_dataset = ProbingDataset(test_files, concept_key, concept_value)
            test_dataset_balanced = balance_dataset(test_dataset, seed=SEED)
            
            if test_dataset_balanced and len(test_dataset_balanced) >= args.min_samples:
                test_dataloader = DataLoader(
                    test_dataset_balanced,
                    batch_size=BATCH_SIZE,
                    shuffle=False
                )
                
                test_activations, test_labels = extract_mlp_activations(
                    model, test_dataloader, LAYER_NUM
                )
                
                test_accuracy = evaluate_probe(probe, test_activations, test_labels)
                logging.info(f"Test accuracy: {test_accuracy:.4f}")
            else:
                logging.warning(f"Not enough test samples after balancing: {len(test_dataset_balanced) if test_dataset_balanced else 0}")
        
        # Guardar probe
        save_probe(probe, probe_path)
        logging.info(f"Saved probe to {probe_path}")
        
        return {
            'language': language,
            'concept_key': concept_key,
            'concept_value': concept_value,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'n_train_samples': len(train_dataset_balanced),
            'n_test_samples': len(test_dataset_balanced) if test_dataset_balanced else 0
        }
        
    except Exception as e:
        logging.error(f"Error processing {language} - {concept_key}: {concept_value}: {str(e)}")
        return None


def main(args):
    """Main training loop"""
    
    # Setup logging
    setup_logging(LOGS_DIR, 'train_probes.log')
    
    logging.info("Starting probe training")
    logging.info(f"Model: {MODEL_ID}")
    logging.info(f"SAE: {SAE_ID}")
    
    # Load model
    logging.info("Loading model...")
    model, _, _, tokenizer = setup_model(MODEL_ID, SAE_ID)
    logging.info(f"Model loaded: {MODEL_ID}")
    
    # Determine languages and concepts to process
    languages = args.languages if args.languages else LANGUAGES
    concepts = args.concepts if args.concepts else CONCEPTS
    
    logging.info(f"Languages: {languages}")
    logging.info(f"Concepts: {concepts}")
    
    # Output directory
    output_dir = PROBES_DIR
    ensure_dir(output_dir)
    
    # Track results
    results = []
    
    # Loop over languages and concepts
    for language in tqdm(languages, desc="Languages"):
        # Get training files
        train_files = get_training_files(language)
        if not train_files:
            logging.warning(f"No training files found for {language}, skipping")
            continue
        
        logging.info(f"Processing {language} with {len(train_files)} training files")
        
        # Get available concepts for this language
        available_concepts = get_available_concepts(train_files)
        
        for concept_key in tqdm(concepts, desc=f"Concepts ({language})", leave=False):
            if concept_key not in available_concepts:
                logging.info(f"Concept {concept_key} not found in {language}, skipping")
                continue
            
            # Get all values for this concept
            concept_values = available_concepts[concept_key]
            
            for concept_value in concept_values:
                result = train_probe_for_concept(
                    model, language, 
                    concept_key, concept_value, output_dir, args
                )
                
                if result:
                    results.append(result)
    
    # Summary
    logging.info(f"\nTraining complete. Trained {len(results)} probes.")
    
    if results:
        avg_train_acc = sum(r['train_accuracy'] for r in results) / len(results)
        test_results = [r for r in results if r['test_accuracy'] is not None]
        avg_test_acc = sum(r['test_accuracy'] for r in test_results) / len(test_results) if test_results else 0
        
        logging.info(f"Average train accuracy: {avg_train_acc:.4f}")
        if test_results:
            logging.info(f"Average test accuracy: {avg_test_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train probes for SAE features")
    parser.add_argument("--languages", nargs='+', help="Languages to process")
    parser.add_argument("--concepts", nargs='+', help="Concepts to process")
    parser.add_argument("--min_samples", type=int, default=128, 
                       help="Minimum number of samples required")
    parser.add_argument("--overwrite", action='store_true',
                       help="Overwrite existing probes")
    
    args = parser.parse_args()
    main(args)

