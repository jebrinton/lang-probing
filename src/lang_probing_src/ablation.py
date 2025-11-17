"""
Sistema de ablación de features SAE
"""

import torch
import numpy as np
from .activations import extract_single_sentence_sae_activations, get_mean_sae_activation


def ablate_features(model, submodule, autoencoder, probe, tokenizer, sentence, 
                   feature_indices, tracer_kwargs=None):
    """
    Ablata features específicas del SAE y mide el efecto en el probe.
    
    Args:
        model: LanguageModel (nnsight)
        submodule: Submodule del modelo
        autoencoder: SAE autoencoder
        probe: Probe entrenado
        tokenizer: Tokenizer del modelo
        sentence: Texto de la sentence
        feature_indices: Lista o array de índices de features a ablatar
        tracer_kwargs: Argumentos para nnsight tracer
        
    Returns:
        dict: {
            'original_logit': float,
            'ablated_logit': float,
            'logit_change': float,
            'original_prob': float,
            'ablated_prob': float
        }
    """
    if tracer_kwargs is None:
        from .config import TRACER_KWARGS
        tracer_kwargs = TRACER_KWARGS
    
    # Obtener activación original
    original_activation = get_mean_sae_activation(
        model, submodule, autoencoder, tokenizer, sentence, tracer_kwargs
    )
    
    # Crear copia para ablación
    ablated_activation = original_activation.clone()
    
    # Ablatar las features
    ablated_activation[feature_indices] = 0
    
    # Evaluar probe en ambas activaciones
    original_activation_np = original_activation.cpu().numpy().reshape(1, -1)
    ablated_activation_np = ablated_activation.cpu().numpy().reshape(1, -1)
    
    # Obtener logits y probabilidades
    original_logit = probe.decision_function(original_activation_np)[0]
    ablated_logit = probe.decision_function(ablated_activation_np)[0]
    
    original_prob = probe.predict_proba(original_activation_np)[0, 1]
    ablated_prob = probe.predict_proba(ablated_activation_np)[0, 1]
    
    logit_change = ablated_logit - original_logit
    
    return {
        'original_logit': float(original_logit),
        'ablated_logit': float(ablated_logit),
        'logit_change': float(logit_change),
        'original_prob': float(original_prob),
        'ablated_prob': float(ablated_prob),
        'prob_change': float(ablated_prob - original_prob)
    }


def activate_features(model, submodule, autoencoder, probe, tokenizer, sentence,
                     feature_indices, activation_value=1.0, tracer_kwargs=None):
    """
    Activa (incrementa) features específicas del SAE y mide el efecto.
    
    Args:
        model: LanguageModel (nnsight)
        submodule: Submodule del modelo
        autoencoder: SAE autoencoder
        probe: Probe entrenado
        tokenizer: Tokenizer del modelo
        sentence: Texto de la sentence
        feature_indices: Lista o array de índices de features a activar
        activation_value: Valor para añadir a las features
        tracer_kwargs: Argumentos para nnsight tracer
        
    Returns:
        dict: Similar a ablate_features
    """
    if tracer_kwargs is None:
        from .config import TRACER_KWARGS
        tracer_kwargs = TRACER_KWARGS
    
    # Obtener activación original
    original_activation = get_mean_sae_activation(
        model, submodule, autoencoder, tokenizer, sentence, tracer_kwargs
    )
    
    # Crear copia para activación
    activated_activation = original_activation.clone()
    
    # Activar las features
    activated_activation[feature_indices] += activation_value
    
    # Evaluar probe
    original_activation_np = original_activation.cpu().numpy().reshape(1, -1)
    activated_activation_np = activated_activation.cpu().numpy().reshape(1, -1)
    
    original_logit = probe.decision_function(original_activation_np)[0]
    activated_logit = probe.decision_function(activated_activation_np)[0]
    
    original_prob = probe.predict_proba(original_activation_np)[0, 1]
    activated_prob = probe.predict_proba(activated_activation_np)[0, 1]
    
    logit_change = activated_logit - original_logit
    
    return {
        'original_logit': float(original_logit),
        'activated_logit': float(activated_logit),
        'logit_change': float(logit_change),
        'original_prob': float(original_prob),
        'activated_prob': float(activated_prob),
        'prob_change': float(activated_prob - original_prob)
    }


def progressive_ablation(model, submodule, autoencoder, probe, tokenizer, sentence,
                        feature_indices, tracer_kwargs=None):
    """
    Ablata features progresivamente y mide el efecto acumulativo.
    
    Args:
        model: LanguageModel
        submodule: Submodule del modelo
        autoencoder: SAE autoencoder
        probe: Probe entrenado
        tokenizer: Tokenizer del modelo
        sentence: Texto de la sentence
        feature_indices: Lista de índices ordenados por importancia
        tracer_kwargs: Argumentos para nnsight tracer
        
    Returns:
        list: Lista de dicts con resultados para cada paso
    """
    results = []
    
    for i in range(1, len(feature_indices) + 1):
        features_to_ablate = feature_indices[:i]
        result = ablate_features(
            model, submodule, autoencoder, probe, tokenizer, sentence,
            features_to_ablate, tracer_kwargs
        )
        result['n_features_ablated'] = i
        results.append(result)
    
    return results


def test_feature_necessity(model, submodule, autoencoder, probe, tokenizer,
                          positive_examples, negative_examples, feature_indices,
                          tracer_kwargs=None):
    """
    Prueba si las features son necesarias para la clasificación.
    
    Args:
        model: LanguageModel
        submodule: Submodule del modelo
        autoencoder: SAE autoencoder
        probe: Probe entrenado
        tokenizer: Tokenizer del modelo
        positive_examples: Lista de sentences con el concepto
        negative_examples: Lista de sentences sin el concepto
        feature_indices: Features a ablatar
        tracer_kwargs: Argumentos para nnsight tracer
        
    Returns:
        dict: Estadísticas de necesidad
    """
    results = {
        'positive': [],
        'negative': []
    }
    
    # Test en ejemplos positivos
    for sentence in positive_examples:
        result = ablate_features(
            model, submodule, autoencoder, probe, tokenizer, sentence,
            feature_indices, tracer_kwargs
        )
        results['positive'].append(result)
    
    # Test en ejemplos negativos
    for sentence in negative_examples:
        result = ablate_features(
            model, submodule, autoencoder, probe, tokenizer, sentence,
            feature_indices, tracer_kwargs
        )
        results['negative'].append(result)
    
    # Calcular estadísticas
    pos_logit_changes = [r['logit_change'] for r in results['positive']]
    neg_logit_changes = [r['logit_change'] for r in results['negative']]
    
    stats = {
        'positive_mean_change': float(np.mean(pos_logit_changes)),
        'positive_std_change': float(np.std(pos_logit_changes)),
        'negative_mean_change': float(np.mean(neg_logit_changes)),
        'negative_std_change': float(np.std(neg_logit_changes)),
        'effect_size': float(np.mean(pos_logit_changes) - np.mean(neg_logit_changes))
    }
    
    results['statistics'] = stats
    
    return results

