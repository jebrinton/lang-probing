import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast
import logging

# --- Configuration ---

# Default location for the probe-results CSVs. Override at the CLI with --input-csv.
DEFAULT_PROBES_DIR = "/projectnb/mcnet/jbrin/lang-probing/outputs/probes"
DEFAULT_INPUT_CSV_GLOB = os.path.join(DEFAULT_PROBES_DIR, "all_probe_results_*.csv")
INPUT_CSV_FILE = None  # populated in main() from CLI args

# The output directory you specified
output_dir = "/projectnb/mcnet/jbrin/lang-probing/img/probes/"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Function ---

def parse_c_value(params_str):
    """
    Safely parses the string representation of a dictionary 
    to extract the 'model__C' value.
    """
    try:
        # ast.literal_eval is a safe way to evaluate a string as a Python literal
        params_dict = ast.literal_eval(params_str)
        if isinstance(params_dict, dict) and 'model__C' in params_dict:
            return float(params_dict['model__C'])
    except (ValueError, SyntaxError, TypeError) as e:
        logging.warning(f"Could not parse 'best_params': {params_str}. Error: {e}")
    return None

# --- Plotting Functions ---

def plot_accuracy_vs_layer(data, concept, value, output_dir):
    """
    Plots training and test accuracy vs. layer for all languages.
    """
    filename = os.path.join(output_dir, f"{concept}_{value}_accuracy_vs_layer.png")
    
    # Melt the dataframe to plot train and test accuracy as categories
    df_melted = data.melt(
        id_vars=['language', 'layer'],
        value_vars=['train_accuracy', 'test_accuracy'],
        var_name='Accuracy Type',
        value_name='Accuracy'
    )
    
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    
    plot = sns.lineplot(
        data=df_melted,
        x='layer',
        y='Accuracy',
        hue='language',
        style='Accuracy Type',
        markers=True,
        markersize=8,
        linewidth=2.5
    )
    
    plot.set_title(f"Probe Accuracy vs. Layer for {concept}: {value}", fontsize=16, weight='bold')
    plot.set_xlabel("Model Layer", fontsize=12)
    plot.set_ylabel("Accuracy", fontsize=12)
    plot.set_ylim(0.895, 1.005) # Adjust this if your accuracy varies more
    plot.legend(title='Legend', loc='best')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    logging.info(f"Saved plot: {filename}")

def plot_c_value_vs_layer(data, output_dir):
    """
    Plots the 'C' hyperparameter value vs. layer for all languages
    and concept/value pairs on a single chart.
    Uses a log scale for the y-axis.
    """
    filename = os.path.join(output_dir, "all_concepts_c_value_vs_layer.png")
    
    # Create a combined column for styling
    data_copy = data.copy()
    data_copy['concept_value'] = data_copy['concept'] + '_' + data_copy['value']
    
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    
    plot = sns.lineplot(
        data=data_copy,
        x='layer',
        y='C_value',
        hue='language',
        style='concept_value', # Use concept_value for style
        marker='o',
        markersize=8,
        linewidth=2.5
    )
    
    # Use a log scale for 'C' as it spans orders of magnitude
    plot.set_yscale('log')
    
    plot.set_title("Hyperparameter 'C' vs. Layer (All Concepts)", fontsize=16, weight='bold')
    plot.set_xlabel("Model Layer", fontsize=12)
    plot.set_ylabel("C Value (Log Scale)", fontsize=12)
    plot.legend(title='Legend', loc='best', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    logging.info(f"Saved plot: {filename}")

def plot_accuracy_distribution(data, concept, value, output_dir):
    """
    Plots the distribution of test accuracy per language using a box plot.
    This shows the median, quartiles, and range of performance across all layers.
    """
    filename = os.path.join(output_dir, f"{concept}_{value}_test_accuracy_distribution.png")
    
    plt.figure(figsize=(10, 7))
    sns.set_theme(style="whitegrid")
    
    plot = sns.boxplot(
        data=data,
        x='language',
        y='test_accuracy',
        palette='muted'
    )
    
    # Add swarmplot for individual data points
    sns.stripplot(
        data=data,
        x='language',
        y='test_accuracy',
        color='black',
        alpha=0.5,
        jitter=0.1
    )
    
    plot.set_title(f"Test Accuracy Distribution (across layers) for {concept}: {value}", fontsize=16, weight='bold')
    plot.set_xlabel("Language", fontsize=12)
    plot.set_ylabel("Test Accuracy", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    logging.info(f"Saved plot: {filename}")

# --- Main Execution ---

def _resolve_input_csv(cli_path):
    """Pick a probe-results CSV, falling back to the most recent match if none is given."""
    if cli_path:
        return cli_path
    matches = sorted(glob.glob(DEFAULT_INPUT_CSV_GLOB), key=os.path.getmtime, reverse=True)
    if not matches:
        return None
    return matches[0]


def main():
    """
    Main function to load data, preprocess, and generate all plots.
    """
    parser = argparse.ArgumentParser(description="Visualize probe results CSVs.")
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help=f"Path to a probe-results CSV. Defaults to most recent match of {DEFAULT_INPUT_CSV_GLOB}.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=output_dir,
        help="Directory to write plots into.",
    )
    args = parser.parse_args()

    input_csv = _resolve_input_csv(args.input_csv)
    if input_csv is None:
        logging.error(f"No input CSV found matching {DEFAULT_INPUT_CSV_GLOB} and --input-csv not set.")
        return
    output_dir = args.output_dir

    # --- 1. Load Data ---
    logging.info(f"Loading data from {input_csv}...")
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        logging.error(f"Error: Input file not found at {input_csv}")
        return
    except Exception as e:
        logging.error(f"Error loading CSV: {e}")
        return
        
    logging.info(f"Loaded {len(df)} rows.")

    # --- 2. Preprocess Data ---
    logging.info("Preprocessing data and parsing 'C' values...")
    df['C_value'] = df['best_params'].apply(parse_c_value)
    
    # Check for parsing errors
    parsed_count = df['C_value'].notna().sum()
    if parsed_count == 0:
        logging.error("Could not parse any 'C' values. Check 'best_params' column format.")
        return
    logging.info(f"Successfully parsed {parsed_count} 'C' values.")

    # Drop rows where parsing failed, as they can't be plotted
    df = df.dropna(subset=['C_value'])

    # --- 3. Create Output Directory ---
    try:
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Ensured output directory exists: {output_dir}")
    except OSError as e:
        logging.error(f"Error: Could not create directory {output_dir}. {e}")
        logging.error("Please check permissions and the path.")
        return

    # --- 4. Generate Plots for each Concept/Value ---
    
    # Group by concept and value to generate separate plots for each
    grouped = df.groupby(['concept', 'value'])
    
    if not any(grouped):
        logging.warning("No data to plot. Check if CSV is empty or parsing failed.")
        return

    for (concept, value), group_df in grouped:
        logging.info(f"--- Generating plots for Concept: {concept}, Value: {value} ---")
        
        if len(group_df) < 2:
            logging.warning(f"Skipping {concept}/{value} - not enough data to plot.")
            continue
            
        # Plot 1: Accuracy (Train/Test) vs. Layer
        plot_accuracy_vs_layer(group_df, concept, value, output_dir)
        
        # Plot 3: Test Accuracy Distribution by Language
        plot_accuracy_distribution(group_df, concept, value, output_dir)

        # Plot 4: C Value vs. Layer
        plot_c_value_vs_layer(group_df, output_dir)


    # --- 5. Generate Global Plots ---
    logging.info("--- Generating global plots ---")
    
    # Plot 2: 'C' Value vs. Layer (All concepts)
    # We pass the full dataframe 'df' here
    plot_c_value_vs_layer(df, output_dir)

    logging.info("--- Visualization script finished ---")

if __name__ == "__main__":
    main()