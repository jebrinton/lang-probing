import os
import glob
import logging
from tqdm import tqdm
from pathlib import Path

import torch
import numpy as np

import matplotlib.pyplot as plt

from lang_probing_src.utils_input_output import get_output_features_vector, get_input_features_vector, load_effects_files, get_language_pairs_and_concepts

def plot_jaccard_topk_similarity(
    input_features: np.ndarray,
    output_features: np.ndarray,
    max_k: int | None = None,
    step_k: int = 1,
    title: str | None = None,
    save_path: Path | str | None = None,
):
    """
    Plot Jaccard similarity between the top-K features of two feature vectors.

    This function treats the feature vectors as scores over the same feature
    indices. For each K, it takes the indices of the K largest scores in
    input_features and output_features, computes the Jaccard similarity
    between these two index sets, and plots Jaccard(K) vs K.

    Args:
        input_features: 1D numpy array of input feature scores.
        output_features: 1D numpy array of output feature scores.
        max_k: Maximum K to consider. If None, defaults to the length
            of the feature vectors.
        step_k: Step size for K (e.g., 10 plots K = 1, 11, 21, ...).
        title: Optional plot title. If None, a default title is used.
        save_path: Optional path to save the figure. If None, the plot
            is shown instead of saved.
    """

    # 1. Input Validation
    if input_features.shape != output_features.shape:
        raise ValueError(
            f"Shape mismatch: input {input_features.shape} vs output {output_features.shape}"
        )
    
    n_features = len(input_features)
    if max_k is None:
        max_k = n_features
    
    max_k = min(max_k, n_features) # Ensure we don't exceed array bounds

    # 2. Get indices of features sorted by score (descending)
    # np.argsort returns ascending, so we flip it [::-1]
    input_sorted_indices = np.argsort(input_features)[::-1]
    output_sorted_indices = np.argsort(output_features)[::-1]

    k_values = []
    jaccard_scores = []

    # 3. Iterate through K values
    # We start range at step_k to avoid k=0, or 1 if step_k is large
    start_k = 1
    
    for k in range(start_k, max_k + 1, step_k):
        # Slice the top k indices
        top_k_input = set(input_sorted_indices[:k])
        top_k_output = set(output_sorted_indices[:k])

        # Calculate Jaccard Similarity
        # J(A, B) = |A ∩ B| / |A ∪ B|
        intersection = len(top_k_input.intersection(top_k_output))
        union = len(top_k_input.union(top_k_output))
        
        jaccard = intersection / union if union > 0 else 0.0
        
        k_values.append(k)
        jaccard_scores.append(jaccard)

    # 4. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, jaccard_scores, label='Jaccard Similarity', linewidth=2)
    
    # Styling
    final_title = title if title else f"Top-K Jaccard Similarity (Max K={max_k})"
    plt.title(final_title, fontsize=14)
    plt.xlabel("K (Number of Top Features)", fontsize=12)
    plt.ylabel("Jaccard Similarity", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(-0.05, 1.05)  # Jaccard is always between 0 and 1
    
    # 5. Save or Show
    if save_path:
        # Convert string to Path object if necessary and ensure parent dir exists
        path_obj = Path(save_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_obj, dpi=300, bbox_inches='tight')
        plt.close() # Close to free memory
    else:
        plt.show()


def signal_plot(signal, title, xlabel, ylabel, save_path=None):
    plt.figure(figsize=(15, 6))

    # Use a thin line (linewidth) and transparency (alpha) to handle the density of 32k points
    plt.plot(signal, linewidth=0.5, alpha=0.8)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3) # Adds a subtle grid for readability

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved plot to {save_path}")
    else:
        plt.show()


def main():
    input_features_dir = Path("/projectnb/mcnet/jbrin/lang-probing/outputs/sentence_input_features")
    effects_files = load_effects_files()
    language_pairs, concepts = get_language_pairs_and_concepts(effects_files)

    plots_dir = Path("/projectnb/mcnet/jbrin/lang-probing/img/input_output")
    plots_dir.mkdir(exist_ok=True)

    for (_, target_lang) in language_pairs:
        for concept in concepts:
            for value in concepts[concept]:
                # if target_lang != "English" or concept != "Tense" or value != "Past":
                #     continue
                
                print(f"Getting input/output features for tg:{target_lang} {concept}={value}")
                try:
                    input_features = get_input_features_vector(input_features_dir, target_lang, concept, value)
                    output_features_list = []

                    # print(input_features[10:])
                    # input_features = np.sort(input_features)
                    # print(input_features[10:])

                    source_langs = [src for src, tgt in language_pairs if tgt == target_lang]
                    for source_lang in source_langs:
                        output_features = get_output_features_vector(effects_files, (source_lang, target_lang), concept, value)
                        output_features_list.append(output_features)
                    output_features = np.mean(output_features_list, axis=0)

                    # print(output_features[10:])
                    # output_features = np.sort(output_features)
                    # print(output_features[10:])

                    print("in/out nonzero counts:")
                    print(np.count_nonzero(input_features))
                    print(np.count_nonzero(output_features))

                    save_dir = plots_dir / f"{target_lang}_{concept}_{value}"
                    save_dir.mkdir(exist_ok=True)

                    # input_features = np.sort(input_features)
                    # output_features = np.sort(output_features)

                    # input_order = np.argsort(input_features)[::-1]
                    # output_order = np.argsort(output_features)[::-1]

                    # print("First 10 indices input_order:", input_order[:10])
                    # print("First 10 indices output_order:", output_order[:10])

                    plot_jaccard_topk_similarity(
                        input_features,
                        output_features,
                        max_k=200,  # o lo que tenga sentido en tu setup
                        step_k=1,
                        title=f"Jaccard top-K for {target_lang} {concept}={value}",
                        save_path=save_dir / "jaccard_topk.png",
                    )

                    # signal_plot(input_features, f"Input features for {target_lang} {concept}={value}", "Feature Index", "Value", save_path=save_dir / "signal_input.png")
                    # signal_plot(output_features, f"Output features for {target_lang} {concept}={value}", "Feature Index", "Value", save_path=save_dir / "signal_output.png")

                except Exception as e:
                    print(f"Error getting input/output features for {target_lang} {concept} {value}: {e}")


if __name__ == "__main__":
    main()
