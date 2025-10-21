"""
Script para encontrar top features SAE correlacionadas con conceptos

Usage:
    python scripts/find_features.py [--k K] [--concept_key KEY] [--concept_value VALUE]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import logging
import glob
from collections import defaultdict
from tqdm import tqdm

from src.config import TOP_K_FEATURES, PROBES_DIR, FEATURES_DIR, LOGS_DIR
from src.utils import ensure_dir, save_json, setup_logging
from src.probe import load_probe
from src.features import (
    find_top_correlating_features,
    find_top_positive_negative_features,
    get_shared_features_across_languages
)




def parse_probe_filename(filename):
    """
    Parsea el nombre de archivo de un probe.
    
    Args:
        filename: Nombre del archivo (e.g., 'English_Tense_Past.joblib')
        
    Returns:
        tuple: (language, concept_key, concept_value) o None si no se puede parsear
    """
    basename = os.path.basename(filename)
    if not basename.endswith('.joblib'):
        return None
    
    basename = basename[:-7]  # Remove .joblib
    parts = basename.split('_')
    
    if len(parts) < 3:
        return None
    
    # Format: Language_ConceptKey_ConceptValue
    # Con nombres simples de idiomas (English, Spanish, Turkish)
    # el formato es más directo
    
    # Buscar el último '_' y el penúltimo '_'
    last_underscore = basename.rfind('_')
    second_last_underscore = basename.rfind('_', 0, last_underscore)
    
    if second_last_underscore == -1:
        return None
    
    language = basename[:second_last_underscore]
    concept_key = basename[second_last_underscore+1:last_underscore]
    concept_value = basename[last_underscore+1:]
    
    return language, concept_key, concept_value


def find_features_for_concept(concept_key, concept_value, probe_dir, k):
    """
    Encuentra features para un concepto específico en todos los idiomas.
    
    Args:
        concept_key: Clave del concepto
        concept_value: Valor del concepto
        probe_dir: Directorio con los probes
        k: Número de top features
        
    Returns:
        dict: {language: [(idx, weight), ...]}
    """
    features_dict = {}
    
    # Buscar todos los probes para este concepto
    pattern = os.path.join(probe_dir, f"*_{concept_key}_{concept_value}.joblib")
    probe_files = glob.glob(pattern)
    
    if not probe_files:
        logging.warning(f"No probes found for {concept_key}:{concept_value}")
        return features_dict
    
    logging.info(f"Found {len(probe_files)} probes for {concept_key}:{concept_value}")
    
    for probe_file in tqdm(probe_files, desc=f"Processing {concept_key}:{concept_value}"):
        parsed = parse_probe_filename(probe_file)
        if not parsed:
            logging.warning(f"Could not parse filename: {probe_file}")
            continue
        
        language, _, _ = parsed
        
        try:
            # Load probe
            probe = load_probe(probe_file)
            
            # Find top features
            top_features = find_top_correlating_features(probe, k=k)
            
            features_dict[language] = top_features
            
            logging.info(f"{language}: Top feature index = {top_features[0][0]}, weight = {top_features[0][1]:.4f}")
            
        except Exception as e:
            logging.error(f"Error processing {probe_file}: {str(e)}")
    
    return features_dict


def main(args):
    """Main feature finding loop"""
    
    # Setup logging
    setup_logging(LOGS_DIR, 'find_features.log')
    
    logging.info("Starting feature analysis")
    logging.info(f"Top K features: {args.k}")
    
    # Directories
    probe_dir = PROBES_DIR
    output_dir = FEATURES_DIR
    ensure_dir(output_dir)
    
    if not os.path.exists(probe_dir):
        logging.error(f"Probe directory not found: {probe_dir}")
        return
    
    # Get all probe files
    all_probe_files = glob.glob(os.path.join(probe_dir, '*.joblib'))
    logging.info(f"Found {len(all_probe_files)} total probes")
    
    # Organize by concept
    concepts_dict = defaultdict(set)
    for probe_file in all_probe_files:
        parsed = parse_probe_filename(probe_file)
        if parsed:
            _, concept_key, concept_value = parsed
            concepts_dict[(concept_key, concept_value)].add(probe_file)
    
    logging.info(f"Found {len(concepts_dict)} unique concepts")
    
    # If specific concept requested, filter
    if args.concept_key and args.concept_value:
        key = (args.concept_key, args.concept_value)
        if key not in concepts_dict:
            logging.error(f"Concept {args.concept_key}:{args.concept_value} not found")
            return
        concepts_to_process = [key]
    else:
        concepts_to_process = list(concepts_dict.keys())
    
    # Process each concept
    for concept_key, concept_value in tqdm(concepts_to_process, desc="Processing concepts"):
        logging.info(f"\nProcessing {concept_key}:{concept_value}")
        
        # Find features
        features_dict = find_features_for_concept(
            concept_key, concept_value, probe_dir, args.k
        )
        
        if not features_dict:
            continue
        
        # Save results
        output_file = os.path.join(output_dir, f"{concept_key}_{concept_value}.json")
        save_json(features_dict, output_file)
        logging.info(f"Saved features to {output_file}")
        
        # Analyze shared features if multiple languages
        if len(features_dict) > 1:
            shared_features = get_shared_features_across_languages(
                features_dict, min_languages=2, top_k=args.k
            )
            
            logging.info(f"Found {len(shared_features)} features shared across languages")
            
            if shared_features:
                # Save shared features separately
                shared_output = os.path.join(output_dir, f"{concept_key}_{concept_value}_shared.json")
                save_json(shared_features, shared_output)
                logging.info(f"Saved shared features to {shared_output}")
                
                # Show top 5 most shared
                sorted_shared = sorted(
                    shared_features.items(), 
                    key=lambda x: x[1]['count'], 
                    reverse=True
                )[:5]
                
                logging.info("Top 5 most shared features:")
                for feature_idx, info in sorted_shared:
                    logging.info(f"  Feature {feature_idx}: {info['count']} languages, avg weight = {info['avg_weight']:.4f}")
    
    logging.info("\nFeature analysis complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find top correlating SAE features")
    parser.add_argument("--k", type=int, default=TOP_K_FEATURES,
                       help="Number of top features to find")
    parser.add_argument("--concept_key", type=str, default=None,
                       help="Specific concept key to process")
    parser.add_argument("--concept_value", type=str, default=None,
                       help="Specific concept value to process")
    
    args = parser.parse_args()
    
    # Validate arguments
    if (args.concept_key is None) != (args.concept_value is None):
        parser.error("--concept_key and --concept_value must be provided together")
    
    main(args)

