"""
Script para generar steering vectors usando difference-in-means

Este script genera steering vectors para cada combinación de (idioma, concepto, valor_concepto, capa)
usando la metodología difference-in-means, donde cada steering vector se calcula como la diferencia
entre la media de activaciones para un valor específico y la media global de todas las activaciones.

Usage:
    python scripts/generate_steering_vectors.py [--languages LANGS] [--concepts CONCEPTS] [--layers LAYERS]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import logging
import pickle
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pyconll
import torch
from tqdm import tqdm

from src.config import (
    MODEL_ID, LANGUAGES, CONCEPTS, BATCH_SIZE, SEED, 
    TRACER_KWARGS, MIN_SAMPLES_FOR_STEERING, MAX_SAMPLES_FOR_STEERING, STEERING_VECTORS_DIR, LOGS_DIR
)
from src.utils import setup_model, ensure_dir, setup_logging
from src.data import get_training_files, get_available_concepts, concept_filter
from src.activations import extract_all_activations_for_steering


def filter_sentences_by_concept(conll_files, concept_key, concept_value):
    """
    Filtra oraciones que contengan un concepto específico.
    
    Args:
        conll_files: lista de paths a archivos .conllu
        concept_key: clave del concepto (e.g., 'Tense')
        concept_value: valor del concepto (e.g., 'Past')
        
    Returns:
        list: lista de oraciones que contienen el concepto
    """
    filtered_sentences = []
    
    for conll_file in conll_files:
        if not os.path.exists(conll_file):
            logging.warning(f"File not found: {conll_file}")
            continue
            
        data = pyconll.load_from_file(conll_file)
        
        for sentence in data:
            if concept_filter(sentence, concept_key, concept_value):
                filtered_sentences.append(sentence.text)
    
    return filtered_sentences


def extract_concept_activations(model, conll_files, concept_key, concept_value, layers, tracer_kwargs=None, batch_size=16, max_samples=MAX_SAMPLES_FOR_STEERING):
    """
    Extrae activaciones para oraciones que contienen un concepto específico.
    
    Args:
        model: LanguageModel (nnsight)
        conll_files: lista de paths a archivos .conllu
        concept_key: clave del concepto
        concept_value: valor del concepto
        layers: lista de números de capa
        tracer_kwargs: argumentos para nnsight tracer
        batch_size: tamaño de batch
        max_samples: máximo número de muestras a procesar

    Returns:
        dict: {layer_num: numpy array de shape (n_sentences, hidden_dim)}
    """
    if tracer_kwargs is None:
        tracer_kwargs = TRACER_KWARGS
    
    # Filtrar oraciones que contengan el concepto
    filtered_sentences = filter_sentences_by_concept(conll_files, concept_key, concept_value)
    
    if max_samples is not None and len(filtered_sentences) > max_samples:
        logging.info(f"Limiting to {max_samples} samples for training {concept_key}={concept_value} steering vector")
        filtered_sentences = filtered_sentences[:max_samples]

    if not filtered_sentences:
        logging.warning(f"No sentences found for {concept_key}={concept_value}")
        return {}
    
    logging.info(f"Found {len(filtered_sentences)} sentences with {concept_key}={concept_value}")
    
    # Inicializar diccionario de activaciones por capa
    activations_by_layer = {layer: [] for layer in layers}
    
    with torch.no_grad():
        # Procesar en batches
        for i in tqdm(range(0, len(filtered_sentences), batch_size), 
                      desc=f"Extracting {concept_key}={concept_value}"):
            batch_sentences = filtered_sentences[i:i + batch_size]
            
            # Extraer activaciones para todas las capas en un solo forward pass
            with model.trace(batch_sentences, **tracer_kwargs):
                input_data = model.inputs.save()
                
                # Guardar activaciones de todas las capas solicitadas
                layer_activations = {}
                for layer_num in layers:
                    acts = model.model.layers[layer_num].output[0].save()
                    layer_activations[layer_num] = acts
            
            # Procesar cada capa
            attn_mask = input_data[1]['attention_mask']
            
            for layer_num in layers:
                acts = layer_activations[layer_num]
                
                # Mask out padding tokens
                acts = acts * attn_mask.unsqueeze(-1)
                
                # Compute mean pooling (weighted by attention mask)
                seq_lengths = attn_mask.sum(dim=1, keepdim=True).float()
                pooled_acts = (acts * attn_mask.unsqueeze(-1)).sum(1) / seq_lengths
                
                # Store results
                activations_by_layer[layer_num].append(pooled_acts.float().cpu().numpy())
    
    # Concatenar todos los batches para cada capa
    final_activations = {}
    for layer_num in layers:
        if activations_by_layer[layer_num]:
            final_activations[layer_num] = np.vstack(activations_by_layer[layer_num])
            logging.info(f"Layer {layer_num}: {final_activations[layer_num].shape[0]} samples")
    
    return final_activations


def compute_steering_vector(concept_activations, global_activations, layer_num):
    """
    Calcula el steering vector usando difference-in-means.
    
    Args:
        concept_activations: numpy array de activaciones para el concepto específico
        global_activations: numpy array de activaciones globales
        layer_num: número de capa
        
    Returns:
        tuple: (steering_vector, stats)
    """
    concept_mean = concept_activations.mean(axis=0)
    global_mean = global_activations.mean(axis=0)
    
    steering_vector = concept_mean - global_mean
    
    stats = {
        'n_samples_concept': len(concept_activations),
        'n_samples_global': len(global_activations),
        'vector_norm': float(np.linalg.norm(steering_vector)),
    }
    
    return steering_vector, stats


def save_steering_vector(vector, metadata, output_dir):
    """
    Guarda steering vector con metadata completa.
    
    Args:
        vector: numpy array del steering vector
        metadata: dict con metadata
        output_dir: directorio de salida
    """
    # Crear directorio
    vectors_dir = os.path.join(output_dir, "vectors")
    ensure_dir(vectors_dir)
    
    # Construir filename
    filename = f"{metadata['language']}_{metadata['concept_key']}_{metadata['concept_value']}_layer{metadata['layer']}.pkl"
    filepath = os.path.join(vectors_dir, filename)
    
    # Preparar datos
    data = {
        'vector': vector,
        'metadata': metadata,
        'timestamp': datetime.now().isoformat()
    }
    
    # Guardar
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    logging.info(f"Saved steering vector to {filename}")


def steering_vector_exists(output_dir, language, concept_key, concept_value, layer_num):
    """
    Verifica si un steering vector ya existe.
    
    Args:
        output_dir: directorio de salida
        language: idioma
        concept_key: clave del concepto
        concept_value: valor del concepto
        layer_num: número de capa
        
    Returns:
        bool: True si el archivo existe, False en caso contrario
    """
    vectors_dir = os.path.join(output_dir, "vectors")
    filename = f"{language}_{concept_key}_{concept_value}_layer{layer_num}.pkl"
    filepath = os.path.join(vectors_dir, filename)
    return os.path.exists(filepath)


def main(args):
    """Main workflow para generar steering vectors."""
    
    # Setup logging
    setup_logging(LOGS_DIR, 'generate_steering_vectors.log')
    
    logging.info("Starting steering vectors generation")
    logging.info(f"Model: {MODEL_ID}")
    logging.info(f"Languages: {args.languages}")
    logging.info(f"Concepts: {args.concepts}")
    logging.info(f"Layers: {args.layers}")
    logging.info(f"Min samples: {MIN_SAMPLES_FOR_STEERING}")
    
    # Load model
    logging.info("Loading model...")
    model, _, _, tokenizer = setup_model(MODEL_ID, None)  # No SAE needed for MLP activations
    logging.info(f"Model loaded: {MODEL_ID}")
    
    # Parse layers
    if args.layers == "all":
        logging.warning("Untested code: Processing all layers")
        layers = list(range(model.model.config.num_hidden_layers))
    else:
        layers = [int(x) for x in args.layers.split(',')]
    logging.info(f"Processing layers: {layers}")
    
    # Output directory
    ensure_dir(args.output_dir)
    
    # Track results
    all_results = []
    
    # Loop over languages
    for language in args.languages:
        logging.info(f"\n{'='*60}\nProcessing language: {language}\n{'='*60}")
        
        # Get training files
        train_files = get_training_files(language)
        if not train_files:
            logging.warning(f"No training files found for {language}, skipping")
            continue
        
        logging.info(f"Found {len(train_files)} training files for {language}")
        
        # Get available concepts for this language
        available_concepts = get_available_concepts(train_files)
        logging.info(f"Available concepts: {list(available_concepts.keys())}")
        
        # Extract global activations for all layers (baseline)
        logging.info("Extracting global activations...")
        global_activations = extract_all_activations_for_steering(
            model, train_files, layers, TRACER_KWARGS, args.batch_size, args.max_samples
        )
        
        # Loop over concepts
        for concept_key in args.concepts:
            if concept_key not in available_concepts:
                logging.info(f"Concept {concept_key} not found in {language}, skipping")
                continue
            
            logging.info(f"\nProcessing concept: {concept_key}")
            
            # Get all values for this concept
            concept_values = available_concepts[concept_key]
            logging.info(f"Available values: {list(concept_values)}")
            
            # Filter values by minimum samples
            valid_values = []
            for concept_value in concept_values:
                # Count samples for this value
                filtered_sentences = filter_sentences_by_concept(train_files, concept_key, concept_value)
                n_samples = len(filtered_sentences)
                
                if n_samples >= MIN_SAMPLES_FOR_STEERING:
                    valid_values.append(concept_value)
                    logging.info(f"  {concept_value}: {n_samples} samples ✓")
                else:
                    logging.info(f"  {concept_value}: {n_samples} samples ✗ (less than {MIN_SAMPLES_FOR_STEERING} samples)")
            
            if not valid_values:
                logging.warning(f"No valid values found for {concept_key} in {language}")
                continue
            
            # Process each valid value
            for concept_value in valid_values:
                logging.info(f"\nProcessing {concept_key}={concept_value}")
                
                # Extract activations for this concept value
                concept_activations = extract_concept_activations(
                    model, train_files, concept_key, concept_value, layers, 
                    TRACER_KWARGS, args.batch_size, args.max_samples
                )
                
                if not concept_activations:
                    logging.warning(f"No activations extracted for {concept_key}={concept_value}")
                    continue
                
                # Compute steering vectors for each layer
                for layer_num in layers:
                    if layer_num not in concept_activations or layer_num not in global_activations:
                        logging.warning(f"Missing activations for layer {layer_num}")
                        continue

                    # Check if steering vector already exists
                    if steering_vector_exists(args.output_dir, language, concept_key, concept_value, layer_num):
                        logging.info(f"Steering vector already exists for {language}_{concept_key}_{concept_value}_layer{layer_num}, skipping")
                        continue
                    
                    # Compute steering vector
                    steering_vector, stats = compute_steering_vector(
                        concept_activations[layer_num], 
                        global_activations[layer_num], 
                        layer_num
                    )
                    
                    # Prepare metadata
                    metadata = {
                        'model_name': MODEL_ID,
                        'language': language,
                        'layer': layer_num,
                        'concept_key': concept_key,
                        'concept_value': concept_value,
                        'method': 'diff_in_means_vs_global',
                        **stats
                    }
                    
                    # Save steering vector
                    save_steering_vector(steering_vector, metadata, args.output_dir)
                    
                    # Track result
                    result = {
                        'language': language,
                        'concept_key': concept_key,
                        'concept_value': concept_value,
                        'layer': layer_num,
                        'vector_norm': stats['vector_norm'],
                        'n_samples_concept': stats['n_samples_concept'],
                        'n_samples_global': stats['n_samples_global']
                    }
                    all_results.append(result)
    
    # Save summary
    summary_file = os.path.join(args.output_dir, "steering_stats.json")
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Generation complete! Generated {len(all_results)} steering vectors")
    logging.info(f"Summary saved to: {summary_file}")
    
    # Print summary statistics
    if all_results:
        languages = set(r['language'] for r in all_results)
        concepts = set(r['concept_key'] for r in all_results)
        layers_used = set(r['layer'] for r in all_results)
        
        logging.info(f"Languages processed: {len(languages)}")
        logging.info(f"Concepts processed: {len(concepts)}")
        logging.info(f"Layers processed: {len(layers_used)}")
        
        avg_norm = sum(r['vector_norm'] for r in all_results) / len(all_results)
        logging.info(f"Average vector norm: {avg_norm:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate steering vectors using diff-in-means")
    parser.add_argument("--languages", nargs='+', default=LANGUAGES,
                       help="Languages to process")
    parser.add_argument("--concepts", nargs='+', default=CONCEPTS,
                       help="Concepts to process")
    parser.add_argument("--layers", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31",
                       help="Comma-separated list of layers")
    parser.add_argument("--output_dir", type=str, default=STEERING_VECTORS_DIR,
                       help="Output directory for steering vectors")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                       help="Batch size for processing")
    parser.add_argument("--max_samples", type=int, default=MAX_SAMPLES_FOR_STEERING,
                       help="Maximum number of samples to process")
    
    args = parser.parse_args()
    main(args)
