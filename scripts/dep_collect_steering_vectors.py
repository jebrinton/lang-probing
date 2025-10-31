"""
Demo script showing basic operations with the activations_all.parquet file.
This demonstrates common analysis tasks you can perform with the activation data.
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from collections import defaultdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import ACTIVATIONS_DIR


def load_activations(parquet_path):
    """Load the activations parquet file."""
    print(f"Loading activations from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # Deserialize tags back to dict
    df['tags'] = df['tags'].apply(json.loads)
    
    print(f"✓ Loaded {len(df)} activation records")
    return df


def print_basic_stats(df):
    """Print basic statistics about the dataset."""
    print("\n" + "="*60)
    print("BASIC STATISTICS")
    print("="*60)
    
    print(f"Total records: {len(df)}")
    print(f"Unique languages: {df['language'].nunique()}")
    print(f"Unique layers: {df['layer'].nunique()}")
    print(f"Unique sentences: {df['sentence_id'].nunique()}")
    
    print("\nLanguages:")
    for lang, count in df['language'].value_counts().items():
        print(f"  {lang}: {count} records")
    
    print("\nLayers:")
    for layer, count in df['layer'].value_counts().sort_index().items():
        print(f"  Layer {layer}: {count} records")
    
    # Activation vector dimensionality
    sample_activation = df.iloc[0]['activation']
    print(f"\nActivation vector dimension: {len(sample_activation)}")


def get_language_activations(df, language, layer=None):
    """
    Get all activations for a specific language, optionally filtered by layer.
    
    Returns:
        DataFrame with filtered activations
    """
    filtered = df[df['language'] == language]
    if layer is not None:
        filtered = filtered[filtered['layer'] == layer]
    return filtered


def get_layer_activations(df, layer):
    """Get all activations for a specific layer across all languages."""
    return df[df['layer'] == layer]


def compute_mean_activation_by_language(df, layer=None):
    """
    Compute mean activation vector for each language.
    
    Args:
        df: DataFrame with activations
        layer: Optional layer to filter by
    
    Returns:
        dict mapping language -> mean activation vector
    """
    if layer is not None:
        df = df[df['layer'] == layer]
    
    mean_activations = {}
    for language in df['language'].unique():
        lang_df = df[df['language'] == language]
        # Stack all activation vectors and compute mean
        activations_matrix = np.stack(lang_df['activation'].values)
        mean_activations[language] = activations_matrix.mean(axis=0)
    
    return mean_activations


def compute_steering_vectors(df, source_lang, target_lang, layer=None):
    """
    Compute steering vector as the difference between mean activations.
    
    Steering vector = mean(target_lang) - mean(source_lang)
    
    Args:
        df: DataFrame with activations
        source_lang: Source language code
        target_lang: Target language code
        layer: Optional layer to filter by
    
    Returns:
        numpy array: steering vector
    """
    mean_activations = compute_mean_activation_by_language(df, layer=layer)
    
    if source_lang not in mean_activations:
        raise ValueError(f"Source language '{source_lang}' not found")
    if target_lang not in mean_activations:
        raise ValueError(f"Target language '{target_lang}' not found")
    
    steering_vector = mean_activations[target_lang] - mean_activations[source_lang]
    return steering_vector


def get_sentences_by_tag(df, tag_key, tag_value, language=None):
    """
    Filter sentences by a specific tag.
    
    Args:
        df: DataFrame with activations
        tag_key: Tag key to filter by
        tag_value: Tag value to match
        language: Optional language filter
    
    Returns:
        DataFrame with filtered sentences
    """
    # Filter by tag
    mask = df['tags'].apply(lambda x: x.get(tag_key) == tag_value)
    filtered = df[mask]
    
    if language is not None:
        filtered = filtered[filtered['language'] == language]
    
    return filtered


def compute_cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)


def compute_language_similarity_matrix(df, layer=None):
    """
    Compute pairwise cosine similarity between language mean activations.
    
    Returns:
        tuple: (similarity_matrix, language_list)
    """
    mean_activations = compute_mean_activation_by_language(df, layer=layer)
    languages = sorted(mean_activations.keys())
    n = len(languages)
    
    similarity_matrix = np.zeros((n, n))
    
    for i, lang1 in enumerate(languages):
        for j, lang2 in enumerate(languages):
            similarity_matrix[i, j] = compute_cosine_similarity(
                mean_activations[lang1],
                mean_activations[lang2]
            )
    
    return similarity_matrix, languages


def demo_basic_operations():
    """Run a demo of basic operations."""
    # Load data
    parquet_path = os.path.join(ACTIVATIONS_DIR, "activations_all.parquet")
    df = load_activations(parquet_path)
    
    # Basic stats
    print_basic_stats(df)
    
    # Example 1: Get activations for a specific language
    print("\n" + "="*60)
    print("EXAMPLE 1: Get Spanish activations at layer 16")
    print("="*60)
    spanish_layer16 = get_language_activations(df, 'es', layer=16)
    print(f"Found {len(spanish_layer16)} Spanish sentences at layer 16")
    print(f"Sample sentence: {spanish_layer16.iloc[0]['sentence_text']}")
    
    # Example 2: Compute mean activation for each language
    print("\n" + "="*60)
    print("EXAMPLE 2: Compute mean activations per language (layer 16)")
    print("="*60)
    mean_acts = compute_mean_activation_by_language(df, layer=16)
    for lang, mean_vec in mean_acts.items():
        print(f"{lang}: shape={mean_vec.shape}, norm={np.linalg.norm(mean_vec):.2f}")
    
    # Example 3: Compute steering vector
    print("\n" + "="*60)
    print("EXAMPLE 3: Compute English->Spanish steering vector (layer 16)")
    print("="*60)
    if 'en' in df['language'].values and 'es' in df['language'].values:
        steering_vec = compute_steering_vectors(df, 'en', 'es', layer=16)
        print(f"Steering vector shape: {steering_vec.shape}")
        print(f"Steering vector norm: {np.linalg.norm(steering_vec):.2f}")
    else:
        print("English or Spanish not found in dataset")
    
    # Example 4: Language similarity matrix
    print("\n" + "="*60)
    print("EXAMPLE 4: Language similarity matrix (layer 16)")
    print("="*60)
    similarity_matrix, languages = compute_language_similarity_matrix(df, layer=16)
    print(f"Languages: {languages}")
    print("Similarity matrix (first 5x5):")
    print(similarity_matrix[:5, :5])
    
    # Example 5: Filter by tags
    print("\n" + "="*60)
    print("EXAMPLE 5: Filter sentences by tags")
    print("="*60)
    # Get all unique tags
    all_tags = defaultdict(set)
    for tags_dict in df['tags']:
        for key, value in tags_dict.items():
            all_tags[key].add(value)
    
    print("Available tags:")
    for key, values in all_tags.items():
        print(f"  {key}: {sorted(values)}")
    
    # Example tag filter (adjust based on your actual tags)
    if all_tags:
        sample_key = list(all_tags.keys())[0]
        sample_value = list(all_tags[sample_key])[0]
        filtered = get_sentences_by_tag(df, sample_key, sample_value)
        print(f"\nFiltered by {sample_key}={sample_value}: {len(filtered)} records")


def load_activations_chunked(parquet_path, layer=None, languages=None, max_rows=None):
    """
    Load activations in chunks to avoid memory issues with large files.
    
    Args:
        parquet_path: Path to parquet file
        layer: Optional layer to filter by
        languages: Optional list of languages to include
        max_rows: Optional maximum rows to load
    
    Returns:
        DataFrame with activations
    """
    print(f"Loading activations from {parquet_path}...")
    
    # Use PyArrow dataset for more efficient filtering
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    
    # Open parquet dataset
    # parquet_file = pq.ParquetFile(parquet_path)
    
    # Build filter conditions
    filters = []
    if layer is not None:
        filters.append(('layer', '=', layer))
    if languages is not None:
        filters.append(('language', 'in', languages))
    
    # Read with filters
    # print(f"Reading {parquet_file.metadata.num_rows} total rows...")
    df = pd.read_parquet(parquet_path, columns=None, filters=filters)
    
    # Deserialize tags
    print("Deserializing tags...")
    df['tags'] = df['tags'].apply(json.loads)
    
    # Limit rows if specified
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)
    
    print(f"✓ Loaded {len(df)} activation records")
    return df


def load_activations_smart(parquet_path, layer=None, languages=None):
    """
    Smart loading with automatic chunking for very large files.
    """
    try:
        # Try direct load first
        df = pd.read_parquet(parquet_path)
        
        # Apply filters
        if layer is not None:
            df = df[df['layer'] == layer]
        if languages is not None:
            df = df[df['language'].isin(languages)]
        
        # Deserialize tags
        df['tags'] = df['tags'].apply(json.loads)
        print(f"✓ Loaded {len(df)} activation records")
        return df
        
    except (OSError, MemoryError) as e:
        print(f"Direct load failed ({e}), trying chunked approach...")
        return load_activations_chunked(parquet_path, layer=layer, languages=languages)

import pyarrow.parquet as pq
import pandas as pd
import json

def load_activations_chunked_by_row_group(parquet_path, layer=None, languages=None, max_rows=None):
    print(f"Loading activations from {parquet_path}...")

    parquet_file = pq.ParquetFile(parquet_path)
    # Build filter conditions
    filters = []
    if layer is not None:
        filters.append(('layer', '=', layer))
    if languages is not None:
        filters.append(('language', 'in', set(languages))) # Use a set for faster 'in' check

    df_chunks = []
    total_rows = 0

    for i in range(parquet_file.num_row_groups):
        row_group = parquet_file.read_row_group(i, columns=["language", "layer", "tags", "activation"])
        # Convert to pandas DataFrame for easier filtering
        df_chunk = row_group.to_pandas()

        # Apply filters
        if layer is not None:
            df_chunk = df_chunk[df_chunk['layer'] == layer]
        if languages is not None:
            df_chunk = df_chunk[df_chunk['language'].isin(languages)]

        if not df_chunk.empty:
            df_chunks.append(df_chunk)
            total_rows += len(df_chunk)

        if max_rows is not None and total_rows >= max_rows:
            break
            
    if not df_chunks:
        return pd.DataFrame()

    df = pd.concat(df_chunks, ignore_index=True)

    # Deserialize tags
    print("Deserializing tags...")
    df['tags'] = df['tags'].apply(json.loads)

    # Limit rows if specified
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)

    print(f"✓ Loaded {len(df)} activation records")
    return df

def main():
    parquet_path = os.path.join(ACTIVATIONS_DIR, "test","activations_test.parquet")

    df = load_activations_chunked_by_row_group(parquet_path, layer=16, languages=['English', 'Spanish'])
    print(df.head())

    # """Main function to collect steering vectors."""
    # df = load_activations_chunked(parquet_path, layer=16, languages=['Greek'])
    # print(df.head())

if __name__ == "__main__":
    # main()
    demo_basic_operations()