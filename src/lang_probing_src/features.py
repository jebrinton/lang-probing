"""
Análisis de features SAE correlacionadas con conceptos gramaticales
"""

import numpy as np
from .utils import save_json, load_json


def find_top_correlating_features(probe, k=100):
    """
    Encuentra las top k features SAE más correlacionadas con el concepto.
    
    Usa los pesos absolutos del probe de regresión logística como medida
    de importancia de cada feature.
    
    Args:
        probe: LogisticRegression probe entrenado
        k: Número de top features a retornar
        
    Returns:
        list: Lista de tuplas (feature_idx, weight) ordenadas por peso absoluto
    """
    # Obtener coeficientes del probe
    coef = probe.coef_.ravel()  # shape: (dict_size,)
    
    # Ordenar por magnitud absoluta
    abs_coef = np.abs(coef)
    top_indices = np.argsort(abs_coef)[-k:][::-1]
    
    # Obtener pesos originales (con signo)
    top_weights = coef[top_indices]
    
    # Retornar como lista de tuplas
    return list(zip(top_indices.tolist(), top_weights.tolist()))


def find_top_positive_negative_features(probe, k=50):
    """
    Encuentra las top k features positivas y negativas.
    
    Args:
        probe: LogisticRegression probe entrenado
        k: Número de features por categoría
        
    Returns:
        dict: {'positive': [(idx, weight), ...], 'negative': [(idx, weight), ...]}
    """
    coef = probe.coef_.ravel()
    
    # Top k positivas
    pos_indices = np.argsort(coef)[-k:][::-1]
    pos_weights = coef[pos_indices]
    
    # Top k negativas
    neg_indices = np.argsort(coef)[:k]
    neg_weights = coef[neg_indices]
    
    return {
        'positive': list(zip(pos_indices.tolist(), pos_weights.tolist())),
        'negative': list(zip(neg_indices.tolist(), neg_weights.tolist()))
    }


def save_features(features_dict, output_path):
    """
    Guarda un diccionario de features a JSON.
    
    Args:
        features_dict: Dict con estructura {language: [(idx, weight), ...]}
        output_path: Path del archivo de salida
    """
    save_json(features_dict, output_path)


def load_features(filepath):
    """
    Carga features desde un archivo JSON.
    
    Args:
        filepath: Path al archivo
        
    Returns:
        dict: Features cargadas
    """
    return load_json(filepath)


def get_feature_indices(features_list, top_k=None):
    """
    Extrae solo los índices de features de una lista.
    
    Args:
        features_list: Lista de tuplas (idx, weight)
        top_k: Número de features a retornar (None = todas)
        
    Returns:
        list: Lista de índices
    """
    if top_k is not None:
        features_list = features_list[:top_k]
    
    return [idx for idx, _ in features_list]


def get_feature_weights(features_list, top_k=None):
    """
    Extrae solo los pesos de features de una lista.
    
    Args:
        features_list: Lista de tuplas (idx, weight)
        top_k: Número de features a retornar (None = todas)
        
    Returns:
        list: Lista de pesos
    """
    if top_k is not None:
        features_list = features_list[:top_k]
    
    return [weight for _, weight in features_list]


def analyze_feature_overlap(features_dict_1, features_dict_2, top_k=50):
    """
    Analiza el overlap entre dos conjuntos de features.
    
    Args:
        features_dict_1: Primer dict de features {language: [(idx, weight), ...]}
        features_dict_2: Segundo dict de features
        top_k: Número de top features a comparar
        
    Returns:
        dict: Estadísticas de overlap
    """
    results = {}
    
    for lang in features_dict_1.keys():
        if lang in features_dict_2:
            indices_1 = set(get_feature_indices(features_dict_1[lang], top_k))
            indices_2 = set(get_feature_indices(features_dict_2[lang], top_k))
            
            overlap = indices_1.intersection(indices_2)
            jaccard = len(overlap) / len(indices_1.union(indices_2))
            
            results[lang] = {
                'overlap_count': len(overlap),
                'jaccard_similarity': jaccard,
                'overlap_indices': list(overlap)
            }
    
    return results


def get_shared_features_across_languages(features_dict, min_languages=2, top_k=100):
    """
    Encuentra features compartidas entre múltiples idiomas.
    
    Args:
        features_dict: Dict {language: [(idx, weight), ...]}
        min_languages: Mínimo número de idiomas en los que debe aparecer
        top_k: Considerar solo top k features de cada idioma
        
    Returns:
        dict: {feature_idx: {'count': int, 'languages': [str], 'avg_weight': float}}
    """
    from collections import defaultdict
    
    # Contar apariciones de cada feature
    feature_counts = defaultdict(list)
    
    for language, features in features_dict.items():
        top_features = features[:top_k]
        for idx, weight in top_features:
            feature_counts[idx].append({'language': language, 'weight': weight})
    
    # Filtrar por mínimo número de idiomas
    shared_features = {}
    for idx, occurrences in feature_counts.items():
        if len(occurrences) >= min_languages:
            languages = [occ['language'] for occ in occurrences]
            weights = [occ['weight'] for occ in occurrences]
            shared_features[int(idx)] = {
                'count': len(occurrences),
                'languages': languages,
                'avg_weight': float(np.mean(weights)),
                'std_weight': float(np.std(weights))
            }
    
    return shared_features

