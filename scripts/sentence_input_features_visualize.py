import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import logging
from pathlib import Path

from input_output_features_visualize import load_diff_vector, sort_diff_vector


def plot_feature_language_distribution(output_dir, languages, concept, value, k=100, save_dir=None):
    """
    Plot a histogram showing the distribution of how many languages each SAE feature
    appears in the top k for.
    
    Args:
        output_dir: Directory containing the diff vectors
        languages: List of language names
        concept: Concept name (e.g., "Tense", "Number")
        value: Value name (e.g., "Past", "Plur")
        k: Number of top features to consider
        save_dir: Optional directory to save the plot. If None, displays the plot.
    """
    # Load all diff vectors and get top k features for each language
    top_k_features = {}
    
    for language in languages:
        diff = load_diff_vector(output_dir, language, concept, value)
        if diff is not None:
            top_k_features[language] = get_top_k_features(diff, k)
        else:
            logging.warning(f"Skipping {language} for {concept}={value} (no diff vector found)")
    
    # Filter out languages that don't have data
    available_languages = list(top_k_features.keys())
    
    if len(available_languages) == 0:
        logging.error(f"No data available for {concept}={value}")
        return
    
    # Count how many languages each feature appears in
    # First, get all unique features that appear in at least one language's top k
    all_features = set()
    for feature_set in top_k_features.values():
        all_features.update(feature_set)
    
    # Count occurrences across languages for each feature
    feature_counts = {}
    for feature_idx in all_features:
        count = sum(1 for lang_features in top_k_features.values() if feature_idx in lang_features)
        feature_counts[feature_idx] = count
    
    # Create histogram data
    language_counts = list(feature_counts.values())
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    max_count = max(language_counts) if language_counts else len(available_languages)
    bins = range(1, max_count + 2)  # +2 to include the max value in the last bin
    
    plt.hist(language_counts, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel('Number of languages')
    plt.ylabel('Number of features')
    plt.title(f'Number of languages across which a feature is present\n(concept={concept}, value={value}, k={k})')
    plt.xticks(range(1, max_count + 1))
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_dir is not None:
        save_path = Path(save_dir) / f"feature_language_distribution_{concept}_{value}_top{k}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved feature language distribution histogram to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_diff_vector_magnitudes(sorted_diff_vector, language, concept, value, save_dir=None):
    """
    Plot the magnitudes of features from a sorted diff_vector.
    
    Args:
        sorted_diff_vector: Tuple of (values, indices) from torch.sort()
        language: Language name for the title
        concept: Concept name for the title
        value: Value name for the title
        save_dir: Optional directory to save the plot. If None, displays the plot.
    """
    values, indices = sorted_diff_vector
    magnitudes = torch.abs(values).cpu().numpy()
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(magnitudes)), magnitudes, width=1.0, edgecolor='none')
    plt.xlabel('Feature Index (sorted by diff value)')
    plt.ylabel('Magnitude')
    plt.title(f'Feature Magnitudes: {language} - {concept} - {value}')
    plt.tight_layout()
    
    if save_dir is not None:
        save_path = Path(save_dir) / f"{language}_{concept}_{value}_magnitudes.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved plot to {save_path}")
        plt.close()
    else:
        plt.show()


def get_top_k_features(diff_vector, k, descending=True):
    """
    Get the indices of the top k features by absolute value.
    
    Args:
        diff_vector: Tensor of difference values
        k: Number of top features to return
        descending: If True, sort by highest absolute values first
        
    Returns:
        Set of feature indices
    """
    abs_values = torch.abs(diff_vector)
    sorted_values, sorted_indices = abs_values.sort(descending=descending)
    top_k_indices = sorted_indices[:k].cpu().tolist()
    return set(top_k_indices)


def jaccard_similarity(set1, set2):
    """
    Compute Jaccard similarity between two sets.
    
    Args:
        set1: First set
        set2: Second set
        
    Returns:
        Jaccard similarity (intersection / union)
    """
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0


def plot_jaccard_similarity_heatmap(output_dir, languages, concept, value, k=100, save_dir=None):
    """
    Create a heatmap showing Jaccard similarity of top k features across languages.
    
    Args:
        output_dir: Directory containing the diff vectors
        languages: List of language names
        concept: Concept name (e.g., "Tense", "Number")
        value: Value name (e.g., "Past", "Plur")
        k: Number of top features to consider
        save_dir: Optional directory to save the plot. If None, displays the plot.
    """
    # Load all diff vectors and get top k features for each language
    top_k_features = {}
    
    for language in languages:
        diff = load_diff_vector(output_dir, language, concept, value)
        if diff is not None:
            top_k_features[language] = get_top_k_features(diff, k)
        else:
            logging.warning(f"Skipping {language} for {concept}={value} (no diff vector found)")
    
    # Filter out languages that don't have data
    available_languages = list(top_k_features.keys())
    
    if len(available_languages) == 0:
        logging.error(f"No data available for {concept}={value}")
        return
    
    # Compute Jaccard similarity matrix
    n_langs = len(available_languages)
    similarity_matrix = np.zeros((n_langs, n_langs))
    
    for i, lang1 in enumerate(available_languages):
        for j, lang2 in enumerate(available_languages):
            similarity_matrix[i, j] = jaccard_similarity(
                top_k_features[lang1], 
                top_k_features[lang2]
            )
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        similarity_matrix,
        xticklabels=available_languages,
        yticklabels=available_languages,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={'label': 'Jaccard Similarity'}
    )
    plt.title(f'Top-{k} Feature Jaccard Similarity: {concept}={value}')
    plt.xlabel('Language')
    plt.ylabel('Language')
    plt.tight_layout()
    
    if save_dir is not None:
        save_path = Path(save_dir) / f"jaccard_similarity_{concept}_{value}_top{k}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved Jaccard similarity heatmap to {save_path}")
        plt.close()
    else:
        plt.show()


def main():
    logging.basicConfig(level=logging.INFO)

    output_dir = Path("/projectnb/mcnet/jbrin/lang-probing/outputs/sentence_input_features")
    output_dir.mkdir(exist_ok=True)

    languages = ["English", "French", "German", "Spanish", "Turkish", "Arabic", "Hindi", "Chinese", "Indonesian"]
    concepts = {"Tense": ["Past", "Pres", "Fut"], "Number": ["Sing", "Dual", "Plur"]}

    # Create plots directory
    plots_dir = Path("/projectnb/mcnet/jbrin/lang-probing/img/sentence_input_features")
    plots_dir.mkdir(exist_ok=True)
    
    # Generate magnitude plots for each language/concept/value
    # for language in languages:
    #     for concept in concepts.keys():
    #         for value in concepts[concept]:
    #             diff = load_diff_vector(output_dir, language, concept, value)
    #             if diff is not None:
    #                 sorted_diff = sort_diff_vector(diff)
    #                 plot_diff_vector_magnitudes(sorted_diff, language, concept, value, save_dir=plots_dir)
    
    # Generate Jaccard similarity heatmaps for each concept/value
    k_values = [30, 50, 100, 200]  # Different k values to try
    for concept in concepts.keys():
        for value in concepts[concept]:
            for k in k_values:
                plot_feature_language_distribution(
                    output_dir,
                    languages,
                    concept,
                    value,
                    k=k,
                    save_dir=plots_dir
                )
                # plot_jaccard_similarity_heatmap(
                #     output_dir, 
                #     languages, 
                #     concept, 
                #     value, 
                #     k=k, 
                #     save_dir=plots_dir
                # )


if __name__ == "__main__":
    main()