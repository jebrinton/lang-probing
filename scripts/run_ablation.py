"""
Script para ejecutar experimentos de ablación de features SAE

Usage:
    python scripts/run_ablation.py --language LANG --concept_key KEY --concept_value VALUE \
                                    --examples examples.txt [--k K]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import logging
import json

from src.config import MODEL_ID, SAE_ID, PROBES_DIR, FEATURES_DIR, ABLATIONS_DIR, LOGS_DIR
from src.utils import setup_model, ensure_dir, load_json, setup_logging
from src.probe import load_probe
from src.features import get_feature_indices
from src.ablation import (
    ablate_features,
    activate_features,
    progressive_ablation,
    test_feature_necessity
)




def load_examples_from_file(filepath):
    """
    Carga ejemplos desde un archivo.
    
    Formato esperado:
    - Líneas vacías separan grupos
    - Primera línea del grupo: etiqueta (positive/negative)
    - Siguientes líneas: sentences
    
    Args:
        filepath: Path al archivo de ejemplos
        
    Returns:
        dict: {'positive': [str], 'negative': [str]}
    """
    examples = {'positive': [], 'negative': []}
    
    if not os.path.exists(filepath):
        logging.error(f"Examples file not found: {filepath}")
        return examples
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_label = None
    for line in lines:
        line = line.strip()
        
        if not line:
            current_label = None
            continue
        
        if line.lower() in ['positive', 'negative']:
            current_label = line.lower()
        elif current_label:
            examples[current_label].append(line)
    
    logging.info(f"Loaded {len(examples['positive'])} positive and {len(examples['negative'])} negative examples")
    
    return examples


def run_simple_ablation(model, submodule, autoencoder, probe, tokenizer, 
                       sentence, feature_indices, k):
    """
    Ejecuta ablación simple en una sentence.
    
    Args:
        model: LanguageModel
        submodule: Submodule del modelo
        autoencoder: SAE autoencoder
        probe: Probe entrenado
        tokenizer: Tokenizer del modelo
        sentence: Texto de la sentence
        feature_indices: Lista de índices de features
        k: Número de top features a ablatar
        
    Returns:
        dict: Resultados de la ablación
    """
    top_k_features = feature_indices[:k]
    
    result = ablate_features(
        model, submodule, autoencoder, probe, tokenizer,
        sentence, top_k_features
    )
    
    result['sentence'] = sentence
    result['n_features_ablated'] = k
    
    return result


def main(args):
    """Main ablation experiment"""
    
    # Setup logging
    setup_logging(LOGS_DIR, 'run_ablation.log')
    
    logging.info("Starting ablation experiment")
    logging.info(f"Language: {args.language}")
    logging.info(f"Concept: {args.concept_key}:{args.concept_value}")
    logging.info(f"Top K features: {args.k}")
    
    # Load model and SAE
    logging.info("Loading model and SAE...")
    model, submodule, autoencoder, tokenizer = setup_model(MODEL_ID, SAE_ID)
    
    # Load probe
    probe_dir = PROBES_DIR
    probe_filename = f"{args.language}_{args.concept_key}_{args.concept_value}.joblib"
    probe_path = os.path.join(probe_dir, probe_filename)
    
    if not os.path.exists(probe_path):
        logging.error(f"Probe not found: {probe_path}")
        return
    
    logging.info(f"Loading probe from {probe_path}")
    probe = load_probe(probe_path)
    
    # Load top features
    features_dir = FEATURES_DIR
    features_filename = f"{args.concept_key}_{args.concept_value}.json"
    features_path = os.path.join(features_dir, features_filename)
    
    if not os.path.exists(features_path):
        logging.error(f"Features file not found: {features_path}")
        return
    
    logging.info(f"Loading features from {features_path}")
    features_dict = load_json(features_path)
    
    if args.language not in features_dict:
        logging.error(f"No features found for language {args.language}")
        return
    
    feature_indices = get_feature_indices(features_dict[args.language])
    logging.info(f"Loaded {len(feature_indices)} feature indices")
    
    # Load examples
    if args.examples:
        examples = load_examples_from_file(args.examples)
    else:
        # Use default examples if none provided
        logging.warning("No examples file provided. Using default examples.")
        examples = {'positive': [], 'negative': []}
    
    # Output directory
    output_dir = ABLATIONS_DIR
    ensure_dir(output_dir)
    
    results = {
        'language': args.language,
        'concept_key': args.concept_key,
        'concept_value': args.concept_value,
        'k': args.k,
        'experiments': []
    }
    
    # Run experiments based on mode
    if args.mode == 'simple':
        # Simple ablation on all examples
        logging.info("Running simple ablation...")
        
        for label in ['positive', 'negative']:
            for sentence in examples[label]:
                logging.info(f"\nAblating ({label}): {sentence}")
                
                result = run_simple_ablation(
                    model, submodule, autoencoder, probe, tokenizer,
                    sentence, feature_indices, args.k
                )
                result['label'] = label
                
                logging.info(f"  Original logit: {result['original_logit']:.4f}")
                logging.info(f"  Ablated logit: {result['ablated_logit']:.4f}")
                logging.info(f"  Change: {result['logit_change']:.4f}")
                
                results['experiments'].append(result)
    
    elif args.mode == 'progressive':
        # Progressive ablation on first positive and negative example
        logging.info("Running progressive ablation...")
        
        for label in ['positive', 'negative']:
            if examples[label]:
                sentence = examples[label][0]
                logging.info(f"\nProgressive ablation ({label}): {sentence}")
                
                progressive_results = progressive_ablation(
                    model, submodule, autoencoder, probe, tokenizer,
                    sentence, feature_indices[:args.k]
                )
                
                for res in progressive_results:
                    res['label'] = label
                    res['sentence'] = sentence
                
                results['experiments'].extend(progressive_results)
                
                # Log summary
                logging.info(f"  Ablating 1 feature: {progressive_results[0]['logit_change']:.4f}")
                logging.info(f"  Ablating {args.k} features: {progressive_results[-1]['logit_change']:.4f}")
    
    elif args.mode == 'necessity':
        # Test feature necessity
        logging.info("Running feature necessity test...")
        
        necessity_results = test_feature_necessity(
            model, submodule, autoencoder, probe, tokenizer,
            examples['positive'], examples['negative'],
            feature_indices[:args.k]
        )
        
        results['necessity_test'] = necessity_results
        
        # Log statistics
        stats = necessity_results['statistics']
        logging.info(f"\nNecessity test results:")
        logging.info(f"  Positive examples - mean change: {stats['positive_mean_change']:.4f} ± {stats['positive_std_change']:.4f}")
        logging.info(f"  Negative examples - mean change: {stats['negative_mean_change']:.4f} ± {stats['negative_std_change']:.4f}")
        logging.info(f"  Effect size: {stats['effect_size']:.4f}")
    
    # Save results
    output_filename = f"{args.language}_{args.concept_key}_{args.concept_value}_{args.mode}_k{args.k}.json"
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation experiments on SAE features")
    parser.add_argument("--language", type=str, required=True,
                       help="Language to test")
    parser.add_argument("--concept_key", type=str, required=True,
                       help="Concept key (e.g., 'Tense')")
    parser.add_argument("--concept_value", type=str, required=True,
                       help="Concept value (e.g., 'Past')")
    parser.add_argument("--examples", type=str, default=None,
                       help="Path to file with example sentences")
    parser.add_argument("--k", type=int, default=10,
                       help="Number of top features to ablate")
    parser.add_argument("--mode", type=str, default='simple',
                       choices=['simple', 'progressive', 'necessity'],
                       help="Ablation mode")
    
    args = parser.parse_args()
    main(args)

