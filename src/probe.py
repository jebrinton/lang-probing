"""
Entrenamiento y evaluación de probes lineales
"""

import os
import joblib
from sklearn.linear_model import LogisticRegression


def train_probe(train_activations, train_labels, seed=42, max_iter=5000, C=0.1):
    """
    Entrena un probe de regresión logística.
    
    Args:
        train_activations: np.array de shape (n_samples, n_features)
        train_labels: np.array de shape (n_samples,)
        seed: Random seed para reproducibilidad
        max_iter: Máximo número de iteraciones
        C: Parámetro de regularización inverso
        
    Returns:
        LogisticRegression: Probe entrenado
    """
    classifier = LogisticRegression(
        random_state=seed,
        max_iter=max_iter,
        class_weight="balanced",
        solver="liblinear",
        C=C
    )
    
    classifier.fit(train_activations, train_labels)
    
    return classifier


def evaluate_probe(probe, test_activations, test_labels):
    """
    Evalúa un probe en un conjunto de test.
    
    Args:
        probe: LogisticRegression probe entrenado
        test_activations: np.array de shape (n_samples, n_features)
        test_labels: np.array de shape (n_samples,)
        
    Returns:
        float: Accuracy en el conjunto de test
    """
    accuracy = probe.score(test_activations, test_labels)
    return accuracy


def save_probe(probe, output_path):
    """
    Guarda un probe entrenado.
    
    Args:
        probe: LogisticRegression probe
        output_path: Path donde guardar el probe
    """
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Guardar con joblib
    joblib.dump(probe, output_path)


def load_probe(probe_path):
    """
    Carga un probe guardado.
    
    Args:
        probe_path: Path al archivo del probe
        
    Returns:
        LogisticRegression: Probe cargado
    """
    return joblib.load(probe_path)


def get_probe_predictions(probe, activations):
    """
    Obtiene predicciones del probe.
    
    Args:
        probe: LogisticRegression probe
        activations: np.array de shape (n_samples, n_features)
        
    Returns:
        dict: {'predictions': np.array, 'probabilities': np.array, 'logits': np.array}
    """
    predictions = probe.predict(activations)
    probabilities = probe.predict_proba(activations)
    
    # Logits = log odds
    decision_function = probe.decision_function(activations)
    
    return {
        'predictions': predictions,
        'probabilities': probabilities,
        'logits': decision_function
    }


def get_probe_info(probe):
    """
    Obtiene información sobre un probe entrenado.
    
    Args:
        probe: LogisticRegression probe
        
    Returns:
        dict: Información del probe
    """
    return {
        'n_features': probe.coef_.shape[1],
        'intercept': float(probe.intercept_[0]),
        'classes': probe.classes_.tolist(),
        'n_iter': probe.n_iter_[0] if hasattr(probe, 'n_iter_') else None,
    }

