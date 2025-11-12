import os
import sys
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import ACTIVATIONS_DIR, LANGUAGES, CONCEPTS_VALUES, BASE_DIR, LAYERS, CONCEPTS_VALUES, COLLECTION_LAYERS

import json
import numpy as np
import logging
from src.config import OUTPUTS_DIR, STEERING_VECTORS_DIR
from src.utils import ensure_dir
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def generate_steering_vector(df, concept_key, concept_value) -> tuple[np.ndarray, int, int]:
    """
    Generate a steering vector for a given concept. 
    Difference in means of mean activations accross all token positions for sentences with and without concept_key=concept_value.
    
    Args:
        df: DataFrame with 'tags' and 'activation' columns
        concept_key: The concept to filter on (e.g., 'Gender', 'Number')
        concept_value: The specific value to filter for (e.g., 'Fem', 'Sing')
    
    Returns:
        steering_vector: numpy array containing the difference in means
        pos_count: number of positive examples
        neg_count: number of negative examples
    """
    if df['tags'].iloc[0] is None:
        logging.warning(f"No tags found for {concept_key}={concept_value}")
        return None, 0, 0
    
    # Parse JSON tags if they're strings
    if isinstance(df['tags'].iloc[0], str):
        df['tags_parsed'] = df['tags'].apply(json.loads)
    else:
        df['tags_parsed'] = df['tags']
    
    # Filter rows where concept_value is in the concept_key list
    def has_concept(tags_dict, key, value):
        """Check if concept_value is in the concept_key list."""
        return key in tags_dict and value in tags_dict[key]
    
    mask = df['tags_parsed'].apply(lambda x: has_concept(x, concept_key, concept_value))
    df_filtered = df[mask].copy()
    df_not_filtered = df[~mask].copy()
    
    pos_count = len(df_filtered)
    neg_count = len(df_not_filtered)
    
    logging.info(f"Positive examples (has {concept_key}={concept_value}): {pos_count}")
    logging.info(f"Negative examples (doesn't have {concept_key}={concept_value}): {neg_count}")
    
    if pos_count == 0:
        return None, pos_count, neg_count
    
    # Convert activations to numpy arrays if they're not already
    pos_activations = np.array([np.array(act) for act in df_filtered['activation'].values])
    neg_activations = np.array([np.array(act) for act in df_not_filtered['activation'].values])
    
    # Calculate means
    pos_mean = np.mean(pos_activations, axis=0)
    neg_mean = np.mean(neg_activations, axis=0)
    
    # Calculate steering vector as difference in means
    steering_vector = pos_mean - neg_mean
    
    # Clean up temporary column
    df.drop(columns=['tags_parsed'], inplace=True, errors='ignore')
    
    return steering_vector, pos_count, neg_count


def load_parquet(language, layer):
    """Load a parquet file and display its head."""
    parquet_path = os.path.join(
        ACTIVATIONS_DIR,
        f"language={language}",
        f"layer={str(layer)}",
        "data.parquet"
    )
    
    if not os.path.exists(parquet_path):
        print(f"⚠ File not found: {parquet_path}")
        return None

    df = pd.read_parquet(parquet_path)
    
    return df


def generate_cosine_similarity_heatmap(steering_vectors, save_path=None, title=None, vmax=1.0, vmin=-1.0):
    """
    Generate a cosine similarity heatmap of all steering vectors.
    
    Args:
        steering_vectors: List of dictionaries with keys 'language', 'layer', 'steering_vector', etc.
        save_path: Path to save the heatmap (optional)
        title: Title for the heatmap (optional)
    """
    # Filter out None steering vectors (when no examples found)
    valid_vectors = [sv for sv in steering_vectors if sv['steering_vector'] is not None]
    
    if len(valid_vectors) == 0:
        logging.error("No valid steering vectors found!")
        return
    
    logging.info(f"Generating heatmap for {len(valid_vectors)} steering vectors")
    
    # Create labels for each steering vector
    labels = []
    vectors = []
    
    for sv in valid_vectors:
        label = f"{sv['language']}\nLayer {sv['layer']}"
        labels.append(label)
        vectors.append(sv['steering_vector'])
    
    # Stack vectors into a matrix
    vectors_matrix = np.vstack(vectors)

    print(vectors_matrix.shape)
    
    # Compute pairwise cosine similarities
    # cosine_similarity expects shape (n_samples, n_features)
    similarity_matrix = cosine_similarity(vectors_matrix)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, len(labels)*0.8), max(10, len(labels)*0.8)))
    
    # Create heatmap
    sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt='.3f',
        cmap='RdYlBu_r',
        center=0,
        vmax=vmax,
        vmin=vmin,
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Cosine Similarity'},
        ax=ax
    )
    
    # Set title
    if title is None:
        title = 'Cosine Similarity of Steering Vectors'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        # Ensure directory exists before saving
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"✓ Heatmap saved to {save_path}")
    
    # Show plot
    plt.show()
    
    return similarity_matrix, labels


def load_steering_vector_parquet(base_dir, concept, value, language, layer) -> pd.DataFrame:
    """
    Loads a single steering vector and its metadata from the partitioned
    Parquet dataset.

    Returns steering vector DataFrame
    """
    try:
        df = pd.read_parquet(
            base_dir,
            filters=[
                ('concept', '==', concept),
                ('value', '==', value),
                ('language', '==', language),
                ('layer', '==', layer)
            ]
        )

        # The filters should match exactly one row/file
        return df.iloc[0]

    except Exception as e:
        if "single positional indexer is out-of-bounds" in str(e):
            logging.debug(f"No vector found for {concept}/{value}/{language}/{layer}")
        else:
            logging.warning(f"Error loading vector {concept}/{value}/{language}/{layer}: {e}")
        return None


def main():
    logging.basicConfig(level=logging.WARNING)

    CONCEPTS_VALUES = {
        "Tense": ["Past", "Pres", "Fut"],
        "Case": ["Nom", "Acc", "Gen", "Dat", "Loc"],
        "Polarity": ["Pos", "Neg"],
        "Aspect": ["Prog", "Imp", "Perf"],
        "Mood": ["Ind", "Imp", "Cnd", "Sub"],
        "Polite": ["Infm", "Form"],
        "Person": ["1", "2", "3"],
        "Degree": ["Pos", "Cmp", "Sup"],
        "Animacy": ["Anim", "Inan"],
    }
    for concept_key in tqdm(CONCEPTS_VALUES.keys(), desc="All concepts", position=0, leave=True, colour='blue'):
        for concept_value in tqdm(CONCEPTS_VALUES[concept_key], desc=f"Concept values of {concept_key}", position=1, leave=False, colour='green'):
            for language in tqdm(LANGUAGES, desc=f"Languages for {concept_key}={concept_value}", position=2, leave=False, colour='red'):
                for layer in tqdm(COLLECTION_LAYERS, desc=f"Layers for {concept_key}={concept_value} in {language}", position=3, leave=False, colour='black'):
                    
                    df = load_parquet(language, layer)
                    steering_vector, pos_count, neg_count = generate_steering_vector(df, concept_key, concept_value)

                    if steering_vector is None:
                        logging.info(f"No examples found for {concept_key}={concept_value} in {language} at layer {layer}, skipping...")
                        continue
                    
                    data = {
                        'steering_vector': [steering_vector],
                        'pos_count': [pos_count],
                        'neg_count': [neg_count],
                    }
                    df_sv = pd.DataFrame(data)

                    partition_dir = os.path.join(
                        STEERING_VECTORS_DIR,
                        f"concept={concept_key}",
                        f"value={concept_value}",
                        f"language={language}",
                        f"layer={layer}"
                    )
                    ensure_dir(partition_dir)
                    
                    df_sv.to_parquet(os.path.join(partition_dir, "data.parquet"), compression='snappy', index=False)

    return

    for concept_key in CONCEPTS_VALUES.keys():
        for concept_value in CONCEPTS_VALUES[concept_key]:
            for layer in layers:
                steering_vectors = {}
                for language in LANGUAGES:
                    steering_vectors[language] = load_steering_vector(STEERING_VECTORS_DIR, concept_key, concept_value, language, layer)

                home = Path.home()
                cos_img_dir = home / "projectnb" / "mcnet" / "jbrin" / "lang-probing" / "cos_img"
                ensure_dir(cos_img_dir)
                save_path = cos_img_dir / f"sv_{concept_key}_{concept_value}_layer{layer}.png"
                
                generate_cosine_similarity_heatmap(steering_vectors, save_path=save_path, title=f"Cosine similarity {concept_key}={concept_value} at layer {layer}")

if __name__ == "__main__":
    main()