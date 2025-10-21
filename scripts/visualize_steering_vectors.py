#!/usr/bin/env python3
"""
Script para visualizar y verificar steering vectors generados

Este script carga steering vectors desde el directorio especificado y verifica
que tanto el vector como los metadatos tengan la estructura correcta.
También puede generar heatmaps de similitudes coseno entre idiomas.

Usage:
    # Visualizar un vector específico:
    python scripts/visualize_steering_vectors.py --language English --concept_key Tense --concept_value Past --layer 18
    
    # Generar heatmap de similitudes coseno:
    python scripts/visualize_steering_vectors.py --concept_key Tense --concept_value Past --layer 18 --generate_heatmap
    
    # Ver resumen de todos los vectores:
    python scripts/visualize_steering_vectors.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

from src.config import STEERING_VECTORS_DIR_NOVA, LOGS_DIR, OUTPUTS_DIR, LANGUAGES
from src.utils import setup_logging, load_steering_metadata, load_steering_vector


def verify_vector_structure(vector, metadata):
    """
    Verifica que el vector y los metadatos tengan la estructura correcta.
    
    Args:
        vector: El steering vector (torch.Tensor o numpy array)
        metadata: Diccionario con metadatos
        
    Returns:
        tuple: (is_valid, issues) - si es válido y lista de problemas encontrados
    """
    issues = []
    
    # Verificar que el vector sea válido
    if vector is None:
        issues.append("Vector es None")
        return False, issues
    
    # Verificar que sea un tensor o array numpy
    if not isinstance(vector, (torch.Tensor, np.ndarray)):
        issues.append(f"Vector no es tensor/array: {type(vector)}")
        return False, issues
    
    # Verificar dimensiones
    if len(vector.shape) != 1:
        issues.append(f"Vector debe ser 1D, pero tiene shape: {vector.shape}")
        return False, issues
    
    # Verificar que no esté vacío
    if vector.shape[0] == 0:
        issues.append("Vector está vacío")
        return False, issues
    
    # Verificar que no contenga NaN o Inf
    if isinstance(vector, torch.Tensor):
        if torch.isnan(vector).any():
            issues.append("Vector contiene valores NaN")
        if torch.isinf(vector).any():
            issues.append("Vector contiene valores Inf")
    else:  # numpy array
        if np.isnan(vector).any():
            issues.append("Vector contiene valores NaN")
        if np.isinf(vector).any():
            issues.append("Vector contiene valores Inf")
    
    # Verificar metadatos requeridos
    required_metadata_keys = [
        'model_name', 'language', 'layer', 'concept_key', 'concept_value', 'method'
    ]
    
    for key in required_metadata_keys:
        if key not in metadata:
            issues.append(f"Metadata falta clave requerida: {key}")
    
    # Verificar tipos de metadatos
    if 'layer' in metadata and not isinstance(metadata['layer'], int):
        issues.append(f"Metadata 'layer' debe ser int, pero es: {type(metadata['layer'])}")
    
    if 'language' in metadata and not isinstance(metadata['language'], str):
        issues.append(f"Metadata 'language' debe ser str, pero es: {type(metadata['language'])}")
    
    if 'concept_key' in metadata and not isinstance(metadata['concept_key'], str):
        issues.append(f"Metadata 'concept_key' debe ser str, pero es: {type(metadata['concept_key'])}")
    
    if 'concept_value' in metadata and not isinstance(metadata['concept_value'], str):
        issues.append(f"Metadata 'concept_value' debe ser str, pero es: {type(metadata['concept_value'])}")
    
    return len(issues) == 0, issues


def compute_pairwise_cosine_similarities(steering_dir, layer, concept_key, concept_value, languages):
    """
    Calcula similitudes coseno pairwise entre steering vectors de diferentes idiomas
    para una capa, concepto y valor específicos.
    
    Args:
        steering_dir: Directorio que contiene los steering vectors
        layer: Número de capa
        concept_key: Clave del concepto (e.g., 'Tense')
        concept_value: Valor del concepto (e.g., 'Past')
        languages: Lista de idiomas a comparar
        
    Returns:
        tuple: (similarity_matrix, available_languages) - matriz de similitudes y idiomas disponibles
    """
    logging.info(f"Calculando similitudes coseno para layer {layer}, {concept_key}={concept_value}")
    
    # Cargar vectores para cada idioma disponible
    vectors = {}
    available_languages = []
    
    for language in languages:
        try:
            vector, metadata = load_steering_vector(steering_dir, language, concept_key, concept_value, layer)
            
            # Convertir a numpy array si es necesario
            if isinstance(vector, torch.Tensor):
                vector = vector.cpu().numpy()
            
            vectors[language] = vector
            available_languages.append(language)
            logging.info(f"✓ Cargado vector para {language}: shape {vector.shape}")
            
        except FileNotFoundError:
            logging.warning(f"✗ No se encontró vector para {language} {concept_key} {concept_value} layer {layer}")
        except Exception as e:
            logging.error(f"✗ Error cargando vector para {language}: {e}")
    
    if len(available_languages) < 2:
        raise ValueError(f"Se necesitan al menos 2 idiomas para calcular similitudes. Disponibles: {available_languages}")
    
    # Crear matriz de vectores
    vector_matrix = np.array([vectors[lang] for lang in available_languages])
    
    # Calcular similitudes coseno
    similarity_matrix = cosine_similarity(vector_matrix)
    
    logging.info(f"Matriz de similitudes calculada: {similarity_matrix.shape}")
    logging.info(f"Idiomas disponibles: {available_languages}")
    
    return similarity_matrix, available_languages


def save_cosine_similarity_heatmap(similarity_matrix, languages, layer, concept_key, concept_value, output_dir):
    """
    Genera y guarda un heatmap de similitudes coseno.
    
    Args:
        similarity_matrix: Matriz de similitudes coseno (numpy array)
        languages: Lista de idiomas (etiquetas para el heatmap)
        layer: Número de capa
        concept_key: Clave del concepto
        concept_value: Valor del concepto
        output_dir: Directorio donde guardar el heatmap
        
    Returns:
        str: Ruta del archivo guardado
    """
    from src.utils import ensure_dir
    
    # Crear directorio de salida
    ensure_dir(output_dir)
    
    # Configurar estilo del plot
    plt.style.use('default')
    sns.set_palette("viridis")
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Generar heatmap
    im = ax.imshow(similarity_matrix, cmap='viridis', vmin=0.7, vmax=1)
    
    # Configurar etiquetas
    ax.set_xticks(range(len(languages)))
    ax.set_yticks(range(len(languages)))
    ax.set_xticklabels(languages, rotation=45, ha='right')
    ax.set_yticklabels(languages)
    
    # Agregar valores en cada celda
    for i in range(len(languages)):
        for j in range(len(languages)):
            text = ax.text(j, i, f'{similarity_matrix[i, j]:.3f}',
                         ha="center", va="center", color="white" if similarity_matrix[i, j] < 0.5 else "black")
    
    # Configurar título y etiquetas
    title = f'Cosine Similarity Heatmap\nLayer {layer}, {concept_key}={concept_value}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Languages', fontsize=12)
    ax.set_ylabel('Languages', fontsize=12)
    
    # Agregar colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity', fontsize=12)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Generar nombre de archivo
    filename = f'cs_layer{layer}_{concept_key}_{concept_value}.png'
    filepath = os.path.join(output_dir, filename)
    
    # Guardar
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Heatmap guardado en: {filepath}")
    
    return filepath


def generate_cosine_similarity_heatmap(steering_dir, layer, concept_key, concept_value, languages, output_dir):
    """
    Función principal para generar heatmap de similitudes coseno.
    
    Args:
        steering_dir: Directorio que contiene los steering vectors
        layer: Número de capa
        concept_key: Clave del concepto
        concept_value: Valor del concepto
        languages: Lista de idiomas a comparar
        output_dir: Directorio donde guardar el heatmap
        
    Returns:
        str: Ruta del archivo guardado
    """
    try:
        # Calcular similitudes coseno
        similarity_matrix, available_languages = compute_pairwise_cosine_similarities(
            steering_dir, layer, concept_key, concept_value, languages
        )
        
        # Generar y guardar heatmap
        filepath = save_cosine_similarity_heatmap(
            similarity_matrix, available_languages, layer, concept_key, concept_value, output_dir
        )
        
        # Imprimir estadísticas
        print("="*60)
        print("HEATMAP DE SIMILITUDES COSENO GENERADO")
        print("="*60)
        print(f"Layer: {layer}")
        print(f"Concepto: {concept_key}={concept_value}")
        print(f"Idiomas: {available_languages}")
        print(f"Archivo guardado: {filepath}")
        
        # Mostrar algunas estadísticas de la matriz
        print(f"\nEstadísticas de similitud:")
        print(f"  Similitud promedio: {np.mean(similarity_matrix):.4f}")
        print(f"  Similitud máxima: {np.max(similarity_matrix):.4f}")
        print(f"  Similitud mínima: {np.min(similarity_matrix):.4f}")
        print(f"  Desviación estándar: {np.std(similarity_matrix):.4f}")
        
        # Mostrar similitudes más altas y más bajas (excluyendo diagonal)
        mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
        off_diagonal = similarity_matrix[mask]
        
        if len(off_diagonal) > 0:
            max_idx = np.argmax(off_diagonal)
            min_idx = np.argmin(off_diagonal)
            
            # Encontrar índices en la matriz original
            i_max, j_max = np.where(similarity_matrix == np.max(off_diagonal))
            i_min, j_min = np.where(similarity_matrix == np.min(off_diagonal))
            
            print(f"\nSimilitudes más altas:")
            for i, j in zip(i_max, j_max):
                if i != j:  # No diagonal
                    print(f"  {available_languages[i]} - {available_languages[j]}: {similarity_matrix[i, j]:.4f}")
            
            print(f"\nSimilitudes más bajas:")
            for i, j in zip(i_min, j_min):
                if i != j:  # No diagonal
                    print(f"  {available_languages[i]} - {available_languages[j]}: {similarity_matrix[i, j]:.4f}")
        
        return filepath
        
    except Exception as e:
        logging.error(f"Error generando heatmap: {e}")
        print(f"✗ Error generando heatmap: {e}")
        return None


def print_vector_info(vector, metadata):
    """
    Imprime información detallada sobre el vector y sus metadatos.
    
    Args:
        vector: El steering vector
        metadata: Diccionario con metadatos
    """
    print("="*60)
    print("INFORMACIÓN DEL STEERING VECTOR")
    print("="*60)
    
    # Información del vector
    print(f"Shape: {vector.shape}")
    print(f"Tipo: {type(vector)}")
    print(f"Dimensión: {vector.shape[0]}")
    
    if isinstance(vector, torch.Tensor):
        print("TORCH TENSOR")
        print(f"Device: {vector.device}")
        print(f"Dtype: {vector.dtype}")
        print(f"Norma L2: {torch.norm(vector).item():.4f}")
        print(f"Norma L1: {torch.norm(vector, p=1).item():.4f}")
        print(f"Min: {torch.min(vector).item():.4f}")
        print(f"Max: {torch.max(vector).item():.4f}")
        print(f"Media: {torch.mean(vector).item():.4f}")
        print(f"Std: {torch.std(vector).item():.4f}")
    else:
        logging.error(f"Vector is not a torch tensor: {type(vector)}")
    # else:  # numpy array
    #     print("NUMPY ARRAY")
    #     print(f"Dtype: {vector.dtype}")
    #     print(f"Norma L2: {np.linalg.norm(vector):.4f}")
    #     print(f"Norma L1: {np.linalg.norm(vector, ord=1):.4f}")
    #     print(f"Min: {np.min(vector):.4f}")
    #     print(f"Max: {np.max(vector):.4f}")
    #     print(f"Media: {np.mean(vector):.4f}")
    #     print(f"Std: {np.std(vector):.4f}")
    
    print("\n" + "-"*40)
    print("METADATOS")
    print("-"*40)
    
    # Mostrar metadatos de forma organizada
    metadata_order = [
        'model_name', 'language', 'layer', 'concept_key', 'concept_value', 'method',
        'n_samples_concept', 'n_samples_global', 'vector_norm'
    ]
    
    for key in metadata_order:
        if key in metadata:
            print(f"{key}: {metadata[key]}")
    
    # Mostrar cualquier metadato adicional
    additional_keys = set(metadata.keys()) - set(metadata_order)
    if additional_keys:
        print("\nMetadatos adicionales:")
        for key in sorted(additional_keys):
            print(f"{key}: {metadata[key]}")


def visualize_specific_vector(steering_dir, language, concept_key, concept_value, layer):
    """
    Visualiza un steering vector específico.
    
    Args:
        steering_dir: Directorio que contiene los steering vectors
        language: Idioma del vector
        concept_key: Clave del concepto
        concept_value: Valor del concepto
        layer: Número de capa
    """
    try:
        logging.info(f"Cargando vector: {language} {concept_key} {concept_value} layer {layer}")
        
        # Cargar el vector y metadatos
        vector, metadata = load_steering_vector(steering_dir, language, concept_key, concept_value, layer)
        
        # Verificar estructura
        is_valid, issues = verify_vector_structure(vector, metadata)
        
        if is_valid:
            print(f"✓ Vector válido!")
            print_vector_info(vector, metadata)
        else:
            print(f"✗ Vector inválido!")
            print("Problemas encontrados:")
            for issue in issues:
                print(f"  - {issue}")
        
        return is_valid
        
    except FileNotFoundError as e:
        logging.error(f"Archivo no encontrado: {e}")
        print(f"✗ Error: {e}")
        return False
    except Exception as e:
        logging.error(f"Error cargando vector: {e}")
        print(f"✗ Error: {e}")
        return False


def visualize_all_vectors(steering_dir):
    """
    Visualiza información general sobre todos los steering vectors disponibles.
    
    Args:
        steering_dir: Directorio que contiene los steering vectors
    """
    try:
        # Cargar metadatos de todos los vectores
        metadata_list = load_steering_metadata(steering_dir)
        
        print("="*60)
        print("RESUMEN DE STEERING VECTORS DISPONIBLES")
        print("="*60)
        
        print(f"Total de vectores: {len(metadata_list)}")
        
        # Agrupar por idioma
        languages = set(item['language'] for item in metadata_list)
        print(f"Idiomas: {sorted(languages)}")
        
        # Agrupar por concepto
        concepts = set(item['concept_key'] for item in metadata_list)
        print(f"Conceptos: {sorted(concepts)}")
        
        # Agrupar por capa
        layers = sorted(set(item['layer'] for item in metadata_list))
        print(f"Capas: {layers}")
        
        print("\n" + "-"*40)
        print("ESTADÍSTICAS POR CONCEPTO")
        print("-"*40)
        
        # Estadísticas por concepto
        concept_stats = {}
        for item in metadata_list:
            concept = item['concept_key']
            if concept not in concept_stats:
                concept_stats[concept] = {'count': 0, 'values': set(), 'languages': set()}
            concept_stats[concept]['count'] += 1
            concept_stats[concept]['values'].add(item['concept_value'])
            concept_stats[concept]['languages'].add(item['language'])
        
        for concept, stats in concept_stats.items():
            print(f"{concept}:")
            print(f"  - Total vectores: {stats['count']}")
            print(f"  - Valores: {sorted(stats['values'])}")
            print(f"  - Idiomas: {sorted(stats['languages'])}")
        
        print("\n" + "-"*40)
        print("PRIMEROS 10 VECTORES")
        print("-"*40)
        
        # Mostrar información de los primeros 10 vectores
        for i, item in enumerate(metadata_list[:10]):
            print(f"{i+1}. {item['language']} {item['concept_key']} {item['concept_value']} layer {item['layer']}")
            print(f"   Norma: {item.get('vector_norm', 'N/A'):.4f}")
            print(f"   Muestras concepto: {item.get('n_samples_concept', 'N/A')}")
            print(f"   Muestras global: {item.get('n_samples_global', 'N/A')}")
        
        if len(metadata_list) > 10:
            print(f"\n... y {len(metadata_list) - 10} vectores más")
        
        return True
        
    except Exception as e:
        logging.error(f"Error cargando metadatos: {e}")
        print(f"✗ Error: {e}")
        return False


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Visualizar steering vectors")
    parser.add_argument("--steering_dir", type=str, default=STEERING_VECTORS_DIR_NOVA,
                       help="Directorio que contiene los steering vectors")
    parser.add_argument("--language", type=str,
                       help="Idioma específico a visualizar")
    parser.add_argument("--concept_key", type=str,
                       help="Clave del concepto específico")
    parser.add_argument("--concept_value", type=str,
                       help="Valor del concepto específico")
    parser.add_argument("--layer", type=int,
                       help="Número de capa específico")
    parser.add_argument("--output_dir", type=str, default="outputs/visualization_steering_vectors",
                       help="Directorio de salida para visualizaciones")
    parser.add_argument("--generate_heatmap", action="store_true",
                       help="Generar heatmap de similitudes coseno")
    parser.add_argument("--languages", nargs='+', 
                       default=LANGUAGES,
                       help="Lista de idiomas para el heatmap")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(LOGS_DIR, 'visualize_steering_vectors.log')
    
    logging.info("Starting steering vectors visualization")
    logging.info(f"Steering directory: {args.steering_dir}")
    
    # Verificar que el directorio existe
    if not os.path.exists(args.steering_dir):
        logging.error(f"Error: Steering directory not found: {args.steering_dir}")

    if args.generate_heatmap:
        # Generar heatmap
        logging.info("Generando heatmap de similitudes coseno")
        filepath = generate_cosine_similarity_heatmap(
            args.steering_dir, args.layer, args.concept_key, args.concept_value, 
            args.languages, args.output_dir
        )
        logging.info(f"Heatmap de similitudes coseno generado en {filepath}")



if __name__ == "__main__":
    main()
