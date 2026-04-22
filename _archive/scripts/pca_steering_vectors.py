import sys
import os
import logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

from scripts.collect_steering_vectors import load_steering_vector_parquet
from lang_probing_src.config import STEERING_VECTORS_DIR, LANGUAGES, CONCEPTS_VALUES, COLLECTION_LAYERS, IMG_DIR

def load_all_vectors(concept_values, languages, layers, base_dir):
    """
    Loads all steering vectors into a single Pandas DataFrame for easy querying.
    """
    all_data = []
    for concept, values in tqdm(concept_values.items(), desc="All concepts", position=0, leave=True, colour='blue'):
        for value in tqdm(values, desc=f"Values for {concept[:10]}", position=1, leave=False, colour='green'):
            for language in tqdm(languages, desc=f"Langs for {value[:10]}", position=2, leave=False, colour='red'):
                for layer in tqdm(layers, desc=f"Layers for {language}", position=3, leave=False):
                    sv_package = load_steering_vector_parquet(base_dir, concept, value, language, layer)
                    
                    if sv_package is None:
                        continue
                        
                    sv = sv_package["steering_vector"]
                    
                    # Append metadata and the vector itself
                    all_data.append({
                        "concept": concept,
                        "value": value,
                        "language": language,
                        "layer": layer,
                        "vector": sv  # Store the full np.array object
                    })
                    
    if not all_data:
        logging.warning("No steering vectors were loaded.")
        return pd.DataFrame(columns=["concept", "value", "language", "layer", "vector"])
        
    return pd.DataFrame(all_data)


def compute_pca(vectors_series, n_components=2):
    """
    Takes a Pandas Series of vectors, stacks them, and runs PCA.
    
    Args:
        vectors_series (pd.Series): A Series where each element is a 1D numpy array.
        n_components (int): Number of principal components.
        
    Returns:
        tuple: (pca_results, pca_model)
               pca_results is the transformed data (n_samples, n_components)
               pca_model is the fitted PCA object.
    """
    if vectors_series.empty or len(vectors_series) < n_components:
        logging.warning("Not enough vectors for PCA.")
        return None, None
        
    # Stack the 1D vectors into a 2D matrix (n_samples, n_features)
    data_matrix = np.stack(vectors_series.values)
    
    pca = PCA(n_components=n_components)
    pca_results = pca.fit_transform(data_matrix)
    
    return pca_results, pca


def create_viz_labels(df, metadata_cols=['concept', 'value', 'language', 'layer'], sep="_"):
    """
    Automatically creates a label Series for visualization by finding
    which columns in the DataFrame *actually vary* (have >1 unique value).
    """
    if df.empty:
        return pd.Series(dtype=str)

    # Find columns (from our list) that are not constant
    varied_cols = []
    for col in metadata_cols:
        if col in df.columns and df[col].nunique() > 1:
            varied_cols.append(col)
    
    # If no columns vary (e.g., all data is for "Fem_Spanish_L16"),
    # we'll just return the single value that *was* in the query.
    # This is an edge case, but good to handle.
    if not varied_cols:
        # Just return the 'value' or first metadata col as a label
        if 'value' in df.columns and df['value'].nunique() == 1:
             return df['value'].astype(str)
        # Fallback to a generic label if 'value' isn't available
        return pd.Series("data_point", index=df.index, dtype=str)

    # Combine the varied columns into a single label
    # Start with the first varied column
    labels = df[varied_cols[0]].astype(str)
    
    # Add any subsequent varied columns with a separator
    for col in varied_cols[1:]:
        labels += sep + df[col].astype(str)
            
    return labels


def visualize_pca(pca_results, pca_model, labels, title="PCA Plot", save_path="my_pca_plot.png"):
    """
    Visualizes 2D PCA results using Seaborn.
    
    Args:
        pca_results (np.ndarray): The (n_samples, 2) array from PCA.
        pca_model (PCA): The fitted PCA object (for variance).
        labels (pd.Series): A Series of labels for coloring the points.
        title (str): The plot title.
    """
    if pca_results is None:
        print("Visualization skipped: No PCA results to plot.")
        return

    # Create a DataFrame for easy plotting
    plot_df = pd.DataFrame(pca_results, columns=['PC1', 'PC2'])
    # Reset index if labels series index doesn't match pca_results (0 to n-1)
    plot_df['label'] = labels.values 
    
    var_pc1 = pca_model.explained_variance_ratio_[0] * 100
    var_pc2 = pca_model.explained_variance_ratio_[1] * 100
    
    plt.figure(figsize=(10, 7))
    ax = sns.scatterplot(
        data=plot_df,
        x='PC1',
        y='PC2',
        hue='label',
        s=100,
        alpha=0.8
    )

    plt.title(title)
    plt.xlabel(f"PC1 ({var_pc1:.1f}% variance)")
    plt.ylabel(f"PC2 ({var_pc2:.1f}% variance)")

    ax.legend (
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.15), 
        borderaxespad=0.,
        ncol=3  # Adjust this number based on label length
    )
    # Give the plot room at the bottom
    plt.subplots_adjust(bottom=0.25)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    plt.savefig(save_path)
    plt.close()


def visualize_pca_style_hue(plot_df, pca_model, style_col='value', hue_col='layer', title="PCA Plot", save_path="my_pca_style_hue.png"):
    """
    Visualizes 2D PCA results using Seaborn.
    
    - 'value' (e.g., Sing, Dual, Plur) is mapped to marker style.
    - 'layer' (numeric) is mapped to a continuous color hue.
    
    Args:
        plot_df (pd.DataFrame): DataFrame with 'PC1', 'PC2', 'value', 'layer'.
        pca_model (PCA): The fitted PCA object (for variance).
        title (str): The plot title.
        save_path (str): Path to save the image.
    """
    if plot_df.empty:
        print("Visualization skipped: No data to plot.")
        return

    var_pc1 = pca_model.explained_variance_ratio_[0] * 100
    var_pc2 = pca_model.explained_variance_ratio_[1] * 100
    
    # Set a default figure size, e.g., (12, 8) to give the legend space
    plt.figure(figsize=(12, 8)) 
    
    # Use 'layer' for hue (continuous color) and 'value' for style (shape)
    ax = sns.scatterplot(
        data=plot_df,
        x='PC1',
        y='PC2',
        style=style_col,      # Map 'layer' to a continuous color
        hue=hue_col,    # Map 'value' (Sing/Dual/Plur) to shape
        palette='viridis', # A good sequential palette for numbers
        s=150,            # Increase size for styled markers
        alpha=0.8
    )
    
    plt.title(title)
    plt.xlabel(f"PC1 ({var_pc1:.1f}% variance)")
    plt.ylabel(f"PC2 ({var_pc2:.1f}% variance)")
    
    # This legend will now be split and much cleaner.
    # It will automatically create a style legend and a hue (colorbar) legend.
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save with bbox_inches='tight' to ensure the legend is included
    plt.savefig(save_path, bbox_inches='tight') 
    plt.show()
    plt.close()


def get_vectors(df, concept=None, value=None, language=None, layer=None):
    """
    Selects vectors from the DataFrame using flexible criteria (Method 1).
    - None (default) means "all" (no filter on this column).
    - A single value (e.g., "Fem") filters for that value.
    - A list (e.g., ["Fem", "Masc"]) filters for any value in the list.
    """
    mask = pd.Series(True, index=df.index)
    criteria = {'concept': concept, 'value': value, 'language': language, 'layer': layer}

    for col, crit in criteria.items():
        if crit is None:
            continue
        if not isinstance(crit, (list, set, tuple)):
            crit = [crit]
        mask &= df[col].isin(crit)

    return df[mask]


def create_plot_filename(concept=None, value=None, language=None, layer=None, sep="_"):
    """
    Creates a standardized string (for titles or filenames) 
    from query parameters.
    """
    params = {'concept': concept, 'value': value, 'language': language, 'layer': layer}
    parts = []
    
    for key, val in params.items():
        if val is None:
            val_str = "all"
        elif isinstance(val, (list, set, tuple)):
            val_str = "-".join(map(str, sorted(list(val)))) # Sort for consistency
        else:
            val_str = str(val)
        parts.append(f"{key}={val_str}")
        
    return sep.join(parts) + ".png"


def create_plot_title(params: dict) -> str:
    """
    Creates a formatted title string from a dictionary,
    skipping any entries where the value is None.
    """
    
    parts = [
        f"{k.capitalize()}: {str(v).replace('_', ' ')}"
        for k, v in params.items()
        if v is not None
    ]
    
    return " | ".join(parts) if parts else "PCA Plot"


def main():
    logging.basicConfig(level=logging.INFO)
    # Step 0: Load all vectors once
    # all_vectors_df = load_all_vectors(
    #     CONCEPTS_VALUES, 
    #     LANGUAGES, 
    #     COLLECTION_LAYERS, 
    #     STEERING_VECTORS_DIR
    # )

    save_dir = os.path.join(IMG_DIR, "pca_steering_vectors")

    my_concepts_values = {
        # "Number": ["Sing", "Dual", "Plur"], 
        # "Gender": ["Fem", "Masc"], 
        # "Gender": ["Fem", "Masc", "Neut"], 
        "Tense": ["Past", "Pres", "Fut"], 
        # "Polite": ["Infm", "Form"]
        # "Case": ["Nom", "Acc", "Gen", "Dat", "Loc"],
        # "Polarity": ["Pos", "Neg"],
        # "Mood": ["Ind", "Imp", "Cnd", "Sub"],
        # "Aspect": ["Prog", "Imp", "Perf"],
        # "Person": ["1", "2", "3"],
        # "Degree": ["Pos", "Cmp", "Sup"],
        # "Animacy": ["Anim", "Inan"],
    }
    my_languages = ["German", "English", "Russian", "Spanish", "Arabic"]
    my_layers = [0, 4, 8, 12, 16, 20, 24, 28, 31]

    all_vectors_df = load_all_vectors(
        my_concepts_values,
        my_languages,
        my_layers,
        STEERING_VECTORS_DIR
    )
    
    # None means do not filter on this column
    queries = [
        {
            "concept": concept,
            "value": None,
            "language": language,
            "layer": my_layers
        } 
        for concept in my_concepts_values.keys() for language in my_languages
    ]

    for query in queries:
        # Step 2: Get the subset of data
        logging.info(f"Querying with params: {query}")
        subset_df = get_vectors(all_vectors_df, **query)
        
        if subset_df.empty:
            logging.warning("Query returned no data.")
        else:
            # Step 3: Generate the title/filename automatically
            plot_filename = os.path.join(save_dir, create_plot_filename(**query))
            plot_title = create_plot_title(query)
            
            # Step 4: Compute PCA
            pca_results, pca_model = compute_pca(subset_df['vector'])
            
            # Step 5: Visualize
            # viz_labels = create_viz_labels(subset_df)
            # visualize_pca(
            #     pca_results,
            #     pca_model,
            #     labels=viz_labels,
            #     title=plot_title,
            #     save_path=plot_filename
            # )

            style_col = 'value'
            hue_col = 'layer'
            plot_df = pd.DataFrame(pca_results, columns=['PC1', 'PC2'])
            plot_df[style_col] = subset_df[style_col].values
            plot_df[hue_col] = subset_df[hue_col].values
            visualize_pca_style_hue(
                plot_df,
                pca_model,
                style_col,
                hue_col,
                title=plot_title,
                save_path=plot_filename
            )

            logging.info(f"Saved plot to {plot_filename}")

if __name__ == "__main__":
    main()
