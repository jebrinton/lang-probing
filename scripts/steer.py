#!/usr/bin/env python3
"""
Script para aplicar steering vectors a prompts y generar texto

Este script carga un steering vector almacenado y lo aplica a prompts de entrada,
generando texto con y sin steering para comparar los efectos.

Usage:
    python scripts/steer.py --steering-dir PATH --prompts-file PATH [OPTIONS]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime
import torch
import nnsight

from src.config import STEERING_OUTPUT_DIR, MODEL_ID, TRACER_KWARGS
from src.utils import setup_model, ensure_dir, setup_logging, load_steering_metadata, load_steering_vector


def generate_with_steering(model, prompt, steering_vector, steering_layer, coefficient, max_new_tokens, apply_to_all):
    """
    Genera texto con y sin steering vector aplicado.
    
    Args:
        model: LanguageModel (nnsight)
        prompt: Texto de entrada
        steering_vector: Vector de steering a aplicar
        steering_layer: Número de capa donde aplicar el steering
        coefficient: Coeficiente multiplicador para el steering vector
        max_new_tokens: Número de tokens nuevos a generar
        apply_to_all: Si True, aplicar a todos los tokens generados; si False, solo al primero
        
    Returns:
        tuple: (unsteered_output, steered_output) como strings decodificados
    """
    # Generar texto SIN steering (baseline)
    with model.generate(prompt, max_new_tokens=max_new_tokens) as tracer:
        unsteered_output = model.generator.output.save()

    with model.generate(prompt, max_new_tokens=max_new_tokens) as tracer:
        hidden_states = model.model.layers[steering_layer].output[0].save()
    
    print(hidden_states.shape)
    print(hidden_states[0].shape)
    print(hidden_states[0][0].shape)
    print("Steering vector shape:")
    print(steering_vector.shape)
    # exit()

    layers = model.model.layers

    # maybe revert to using next()??
    # hidden_states = []
    # with model.generate(prompt, max_new_tokens=max_new_tokens) as tracer:
    #     for i in range(max_new_tokens):
    #         # Apply intervention - set first layer output to zero
    #         layers[steering_layer].output[0][:] += coefficient * steering_vector

    #         # Append desired hidden state post-intervention
    #         hidden_states.append(layers[-1].output.save())

    #         # Move to next generated token
    #         layers[0].next()

    # Generar texto CON steering

    # TODO: steer only at the last generation step
    with model.generate(prompt, max_new_tokens=max_new_tokens) as tracer:
        hidden_states = nnsight.list().save()

        layers.all()

        layers[steering_layer].output[0][-1, :] += coefficient * steering_vector

        hidden_states.append(layers[steering_layer].output)

        steered_output = model.generator.output.save()

        # if apply_to_all:
        #     # Aplicar steering a todos los tokens generados
        #     model.model.layers.all()  # Llamar .all() en el módulo, NO en tracer
        #     model.model.layers[layer].output[0][0, :] += coefficient * steering_vector # apply steering vector to all tokens
        #     # I'm pretty sure that layers[layer].output[0] is a 2D tensor of (batch_size x hidden_dim)
        # else:
        #     print("EROROR")
        #     exit()
        #     # Aplicar steering solo al último token generado
        #     model.model.layers[layer].output[0][-1, :] += coefficient * steering_vector
        
        

    print("Steered output:")
    print(steered_output.shape)
    print(steered_output[0].shape)

    print("Unsteered output:")
    print(unsteered_output.shape)
    print(unsteered_output[0].shape)

    # Decodificar los outputs
    unsteered_text = model.tokenizer.decode(unsteered_output[0][-max_new_tokens:].cpu())
    steered_text = model.tokenizer.decode(steered_output[0][-max_new_tokens:].cpu())
    
    return unsteered_text, steered_text


def main():
    """Función principal del script"""
    parser = argparse.ArgumentParser(description="Aplicar steering vectors a prompts y generar texto")
    
    parser.add_argument("--steering-dir", required=True, 
                       help="Path al directorio que contiene steering_stats.json y subdirectorio vectors/")
    parser.add_argument("--prompts-file", required=True,
                       help="Path al archivo .txt con prompts (uno por línea)")
    parser.add_argument("--coefficient", type=float, default=1.0,
                       help="Coeficiente multiplicador para el steering vector (default: 1.0)")
    parser.add_argument("--max-new-tokens", type=int, default=28,
                       help="Número de tokens nuevos a generar (default: 28)")
    parser.add_argument("--apply-to-all", action="store_true", default=True,
                       help="Aplicar steering a todos los tokens generados (default: True)")
    parser.add_argument("--output-file", 
                       help="Path opcional para el archivo de salida JSON (si no se especifica, se genera automáticamente)")
    
    args = parser.parse_args()
    
    # Configurar logging
    setup_logging()
    
    try:
        # Validar que el directorio de steering existe
        if not os.path.exists(args.steering_dir):
            raise FileNotFoundError(f"El directorio de steering no existe: {args.steering_dir}")
        
        # Validar que el archivo de prompts existe
        if not os.path.exists(args.prompts_file):
            raise FileNotFoundError(f"El archivo de prompts no existe: {args.prompts_file}")
        
        # Cargar metadata de steering
        logging.info(f"Cargando metadata desde: {args.steering_dir}")
        metadata = load_steering_metadata(args.steering_dir) # TODO: this is not really "metadata" but a list of steering vectors
        # the metadata is stored as 
        # data = {
        #     'vector': vector,  # El vector numpy
        #     'metadata': metadata,
        #     'timestamp': datetime.now().isoformat()
        # }
        
        if len(metadata) == 0:
            raise ValueError("No se encontraron vectores en el directorio de steering")
        
        # Si hay múltiples vectores, usar el primero (podría mejorarse para permitir selección)
        if len(metadata) > 1:
            logging.warning(f"Se encontraron {len(metadata)} vectores. Usando el primero: {metadata[0]}")
        
        vector_info = metadata[0]
        language = vector_info["language"]
        concept_key = vector_info["concept_key"]
        concept_value = vector_info["concept_value"]
        layer = vector_info["layer"]
        
        logging.info(f"Usando vector: {language} {concept_key} {concept_value} layer {layer}")
        
        # Cargar el steering vector
        logging.info("Cargando steering vector...")
        steering_vector, metadata = load_steering_vector(args.steering_dir, language, concept_key, concept_value, layer)
        logging.info(f"Steering vector cargado: shape {steering_vector.shape}")
        
        # Cargar modelo
        logging.info(f"Cargando modelo: {MODEL_ID}")
        model, _, _, tokenizer = setup_model(MODEL_ID, None)
        
        # Leer prompts del archivo
        logging.info(f"Leyendo prompts desde: {args.prompts_file}")
        with open(args.prompts_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        logging.info(f"Se encontraron {len(prompts)} prompts")
        
        # Generar resultados para cada prompt
        results = []
        for i, prompt in enumerate(prompts):
            logging.info(f"Procesando prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            try:
                unsteered_output, steered_output = generate_with_steering(
                    model, prompt, steering_vector, layer, args.coefficient, 
                    args.max_new_tokens, args.apply_to_all
                )
                
                result = {
                    "prompt": prompt,
                    "unsteered_output": unsteered_output,
                    "steered_output": steered_output,
                    "coefficient": args.coefficient,
                    "language": language,
                    "concept_key": concept_key,
                    "concept_value": concept_value,
                    "layer": layer,
                    "max_new_tokens": args.max_new_tokens,
                    "apply_to_all": args.apply_to_all
                }
                
                results.append(result)
                
            except Exception as e:
                logging.error(f"Error procesando prompt {i+1}: {str(e)}")
                continue
        
        # Determinar archivo de salida
        if args.output_file:
            output_file = args.output_file
        else:
            # Generar nombre automático
            ensure_dir(STEERING_OUTPUT_DIR)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{language}_{concept_key}_{concept_value}_coef{args.coefficient}_tokens{args.max_new_tokens}_{timestamp}.json"
            output_file = os.path.join(STEERING_OUTPUT_DIR, filename)
        
        # Guardar resultados
        logging.info(f"Guardando resultados en: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logging.info(f"✓ Proceso completado exitosamente!")
        logging.info(f"  - Prompts procesados: {len(results)}")
        logging.info(f"  - Archivo de salida: {output_file}")
        
    except Exception as e:
        logging.error(f"✗ Error durante la ejecución: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
