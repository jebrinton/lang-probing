"""
Utilidades generales para el sistema de probing
"""

import os
import json
import pickle
import logging
import torch
import pandas as pd
from nnsight import LanguageModel
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

# Import setup_autoencoder from the existing codebase
import sys
sys.path.append('/projectnb/mcnet/jbrin/lang-similarity/src')
from sae.utils import setup_autoencoder


def get_device_info():
    """
    Determina el mejor dispositivo disponible para PyTorch.
    
    Returns:
        tuple: (device, dtype) - dispositivo y tipo de datos recomendado
    """
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        if n_gpus == 1:
            device = torch.device("cuda")
            dtype = torch.bfloat16
        else:
            task_id = int(os.environ.get("SGE_TASK_ID", 1))
            gpu_id = (task_id - 1) % n_gpus
            torch.cuda.set_device(gpu_id)
            device = torch.device(f"cuda:{gpu_id}")
            dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    return device, dtype


def setup_model(model_id, sae_id=None):
    """
    Configura el modelo de lenguaje, tokenizer y opcionalmente autoencoder SAE.

    Args:
        model_id: ID del modelo en HuggingFace
        sae_id: ID del SAE en HuggingFace (opcional)

    Returns:
        tuple: (model, submodule, autoencoder, tokenizer)
    """
    device, dtype = get_device_info()
    
    # Load language model
    model = LanguageModel(model_id, torch_dtype=dtype, device_map=device)
    submodule = model.model.layers[16]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.tokenizer = tokenizer

    # Load SAE from HuggingFace Hub (opcional)
    if sae_id is not None:
        sae_filename = "aya-23-8b-layer16.pt" if "aya" in model_id.lower() else "llama-3-8b-layer16.pt"
        sae_path = hf_hub_download(repo_id=sae_id, filename=sae_filename)
        autoencoder = setup_autoencoder(sae_path)
    else:
        autoencoder = None

    return model, submodule, autoencoder, tokenizer


def print_gpu_memory_usage():
    """
    Imprime el uso actual de memoria GPU.
    """
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(i) / 1024**3    # GB
            available = torch.cuda.get_device_properties(i).total_memory / 1024**3    # GB
            print(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {available:.2f}GB available")
    else:
        print("No CUDA GPUs available")


def save_json(data, filepath):
    """
    Guarda datos en formato JSON.
    
    Args:
        data: Datos a guardar
        filepath: Ruta del archivo
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath):
    """
    Carga datos desde un archivo JSON.
    
    Args:
        filepath: Ruta del archivo
        
    Returns:
        dict: Datos cargados
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def ensure_dir(directory):
    """
    Asegura que un directorio existe.
    
    Args:
        directory: Ruta del directorio
    """
    os.makedirs(directory, exist_ok=True)


def setup_logging(log_dir, log_filename="unnamed.log"):
    """
    Configura el sistema de logging.
    
    Args:
        log_dir: Directorio donde guardar el archivo de log
        log_filename: Nombre del archivo de log
    """
    ensure_dir(log_dir)
    log_file = os.path.join(log_dir, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def load_steering_metadata(steering_dir):
    """
    Carga y parsea el archivo steering_stats.json del directorio especificado.
    
    Args:
        steering_dir: Path al directorio que contiene steering_stats.json
        
    Returns:
        list: Lista de metadatos de vectores disponibles
    """
    stats_file = os.path.join(steering_dir, "steering_stats.json")
    
    if not os.path.exists(stats_file):
        raise FileNotFoundError(f"No se encontró el archivo de estadísticas: {stats_file}")
    
    with open(stats_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    if not isinstance(metadata, list):
        raise ValueError("El archivo steering_stats.json debe contener una lista de objetos")
    
    return metadata


def load_steering_vector(steering_dir, language, concept_key, concept_value, layer):
    """
    Carga un steering vector específico desde el directorio.
    
    Args:
        steering_dir: Path al directorio que contiene el subdirectorio vectors/
        language: Idioma del vector
        concept_key: Clave del concepto (e.g., 'Tense')
        concept_value: Valor del concepto (e.g., 'Past')
        layer: Número de capa
        
    Returns:
        tuple: (vector, metadata) - El steering vector y sus metadatos
    """
    vector_filename = f"{language}_{concept_key}_{concept_value}_layer{layer}.pkl"
    vector_path = os.path.join(steering_dir, "vectors", vector_filename)
    
    if not os.path.exists(vector_path):
        raise FileNotFoundError(f"No se encontró el vector: {vector_path}")
    
    with open(vector_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extraer el vector y metadata del diccionario guardado
    if isinstance(data, dict) and 'vector' in data:
        vector = data['vector']
        metadata = data.get('metadata', {})
    else:
        # Fallback: asumir que el archivo contiene directamente el vector
        logging.warning(f"File {vector_path} does not contain a dictionary with 'vector' key. Assuming it contains the vector directly.")
        vector = data
        metadata = {}
    
    # Asegurar que es un tensor de PyTorch
    if not isinstance(vector, torch.Tensor):
        vector = torch.tensor(vector)
    
    return vector, metadata

def load_parquet_steering_vector(steering_dir, concept_key, concept_value, language, layer):
    """
    Carga un steering vector específico desde el directorio.
    
    Args:
        steering_dir: Path al directorio que contiene el subdirectorio vectors/
        language: Idioma del vector
        concept_key: Clave del concepto (e.g., 'Tense')
        concept_value: Valor del concepto (e.g., 'Past')
        layer: Número de capa
    """
    sv_path = os.path.join(steering_dir, f"concept={concept_key}/value={concept_value}/language={language}/layer={layer}/data.parquet")
    if not os.path.exists(sv_path):
        raise FileNotFoundError(f"No se encontró el vector: {sv_path}")
    return pd.read_parquet(sv_path)
