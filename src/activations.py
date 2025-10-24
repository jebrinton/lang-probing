"""
Extracción de activaciones desde modelos de lenguaje
"""

import torch
import numpy as np
import warnings
import logging
import os
from tqdm import tqdm


def extract_mlp_activations(model, dataloader, layer_num, tracer_kwargs=None):
    """
    Extrae activaciones MLP para un dataset completo desde cualquier capa.
    
    Args:
        model: LanguageModel (nnsight)
        dataloader: DataLoader con batches de sentences
        layer_num: Número de capa a extraer
        tracer_kwargs: Argumentos para nnsight tracer
        
    Returns:
        tuple: (activations, labels)
            - activations: np.array de shape (n_samples, hidden_dim)
            - labels: np.array de shape (n_samples,)
    """
    if tracer_kwargs is None:
        from .config import TRACER_KWARGS
        tracer_kwargs = TRACER_KWARGS
    
    all_activations = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting MLP activations"):
            text_batch = batch["sentence"]
            labels = batch["label"]
            
            # Extract activations from model.model.layers[layer_num].output[0]
            with model.trace(text_batch, **tracer_kwargs):
                input_data = model.inputs.save()
                acts = model.model.layers[layer_num].output[0].save()
            
            # Get attention mask to handle padding
            attn_mask = input_data[1]['attention_mask']
            
            # Mask out padding tokens
            acts = acts * attn_mask.unsqueeze(-1)
            
            # Compute mean pooling (weighted by attention mask)
            seq_lengths = attn_mask.sum(dim=1, keepdim=True).float()
            pooled_acts = (acts * attn_mask.unsqueeze(-1)).sum(1) / seq_lengths
            
            # Store results
            all_activations.append(pooled_acts.float().cpu().numpy())
            all_labels.append(labels.numpy())
    
    # Concatenate all batches
    activations = np.vstack(all_activations)
    labels = np.concatenate(all_labels)
    
    return activations, labels


def extract_sae_activations(model, submodule, autoencoder, dataloader, layer_num=16, tracer_kwargs=None):
    """
    DEPRECATED: Use extract_mlp_activations() instead.
    
    Extrae activaciones SAE para un dataset completo.
    
    Args:
        model: LanguageModel (nnsight)
        submodule: Submodule del modelo (e.g., model.model.layers[16])
        autoencoder: SAE autoencoder
        dataloader: DataLoader con batches de sentences
        layer_num: Número de capa a extraer
        tracer_kwargs: Argumentos para nnsight tracer
        
    Returns:
        tuple: (activations, labels)
            - activations: np.array de shape (n_samples, dict_size)
            - labels: np.array de shape (n_samples,)
    """
    warnings.warn(
        "extract_sae_activations is deprecated. Use extract_mlp_activations() for general MLP activations.",
        DeprecationWarning,
        stacklevel=2
    )
    if tracer_kwargs is None:
        from .config import TRACER_KWARGS
        tracer_kwargs = TRACER_KWARGS
    
    all_activations = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting SAE activations"):
            text_batch = batch["sentence"]
            labels = batch["label"]
            
            # Tokenize and get activations from the model
            with model.trace(text_batch, **tracer_kwargs):
                input_data = model.inputs.save()
                acts = submodule.output[0].save()
            
            # Get attention mask to handle padding
            attn_mask = input_data[1]['attention_mask']
            
            # Mask out padding tokens
            acts = acts * attn_mask.unsqueeze(-1)
            
            # Compute mean pooling (weighted by attention mask)
            seq_lengths = attn_mask.sum(dim=1, keepdim=True).float()
            pooled_acts = (acts * attn_mask.unsqueeze(-1)).sum(1) / seq_lengths
            
            # Encode through SAE
            # Reshape to (batch_size, hidden_dim)
            batch_size = pooled_acts.shape[0]
            pooled_acts_2d = pooled_acts.view(batch_size, -1)
            
            # Encode to SAE feature space
            sae_activations = autoencoder.encode(pooled_acts_2d)
            
            # Store results
            all_activations.append(sae_activations.float().cpu().numpy())
            all_labels.append(labels.numpy())
    
    # Concatenate all batches
    activations = np.vstack(all_activations)
    labels = np.concatenate(all_labels)
    
    return activations, labels


def extract_single_sentence_sae_activations(model, submodule, autoencoder, tokenizer, sentence, tracer_kwargs=None):
    """
    Extrae activaciones SAE para una sola sentence.
    
    Args:
        model: LanguageModel (nnsight)
        submodule: Submodule del modelo
        autoencoder: SAE autoencoder
        tokenizer: Tokenizer del modelo
        sentence: Texto de la sentence
        tracer_kwargs: Argumentos para nnsight tracer
        
    Returns:
        torch.Tensor: Activaciones SAE de shape (seq_len, dict_size)
    """
    if tracer_kwargs is None:
        from .config import TRACER_KWARGS
        tracer_kwargs = TRACER_KWARGS
    
    with torch.no_grad():
        # Tokenize
        tokens = tokenizer(sentence, return_tensors="pt", padding=False)
        input_ids = tokens['input_ids']
        
        # Get activations
        with model.trace(input_ids, **tracer_kwargs):
            acts = submodule.output[0].save()
        
        # acts shape: (batch=1, seq_len, hidden_dim)
        acts = acts.squeeze(0)  # (seq_len, hidden_dim)
        
        # Encode through SAE
        sae_activations = autoencoder.encode(acts)
        
        return sae_activations


def get_mean_sae_activation(model, submodule, autoencoder, tokenizer, sentence, tracer_kwargs=None):
    """
    Obtiene la activación SAE promedio para una sentence.
    
    Args:
        model: LanguageModel (nnsight)
        submodule: Submodule del modelo
        autoencoder: SAE autoencoder
        tokenizer: Tokenizer del modelo
        sentence: Texto de la sentence
        tracer_kwargs: Argumentos para nnsight tracer
        
    Returns:
        torch.Tensor: Activación SAE promedio de shape (dict_size,)
    """
    sae_acts = extract_single_sentence_sae_activations(
        model, submodule, autoencoder, tokenizer, sentence, tracer_kwargs
    )
    
    # Mean pooling over sequence length
    mean_acts = sae_acts.mean(dim=0)
    
    return mean_acts


def extract_all_activations_for_steering(model, conll_files, layers, tracer_kwargs=None, batch_size=16, max_sentences=None, seed=42):
    """
    Extrae activaciones para todas las oraciones en archivos UD, para múltiples capas.
    Esta función se usa para obtener la baseline global de activaciones.
    
    Args:
        model: LanguageModel (nnsight)
        conll_files: lista de paths a archivos .conllu
        layers: lista de números de capa a extraer (e.g., [0,1,2,...,31])
        tracer_kwargs: argumentos para nnsight tracer
        batch_size: tamaño de batch para procesamiento
        
    Returns:
        dict: {layer_num: numpy array de shape (n_sentences, hidden_dim)}
    """
    if tracer_kwargs is None:
        from .config import TRACER_KWARGS
        tracer_kwargs = TRACER_KWARGS
    
    import pyconll
    
    # Cargar todas las oraciones de todos los archivos
    all_sentences = []
    for conll_file in conll_files:
        if not os.path.exists(conll_file):
            logging.warning(f"File not found: {conll_file}")
            continue
            
        data = pyconll.load_from_file(conll_file)
        for sentence in data:
            all_sentences.append(sentence.text)
    
    # Limitar el número de oraciones
    if max_sentences is not None and len(all_sentences) > max_sentences:
        logging.info(f"Limiting to {max_sentences} sentences for testing")
        step = len(all_sentences)//max_sentences
        all_sentences = all_sentences[::step]
    
    logging.info(f"Loaded {len(all_sentences)} sentences from {len(conll_files)} files")
    
    # Inicializar diccionario de activaciones por capa
    activations_by_layer = {layer: [] for layer in layers}
    
    with torch.no_grad():
        # Procesar en batches
        for i in tqdm(range(0, len(all_sentences), batch_size), desc="Extracting global activations"):
            batch_sentences = all_sentences[i:i + batch_size]
            
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
        final_activations[layer_num] = np.vstack(activations_by_layer[layer_num])
        logging.info(f"Layer {layer_num}: {final_activations[layer_num].shape[0]} samples, {final_activations[layer_num].shape[1]} dimensions")
    
    return final_activations

