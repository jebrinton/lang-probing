import argparse
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# Import your language mappings
try:
    from lang_probing_src.config import LANG_CODE_TO_NAME
except ImportError:
    print("Warning: Could not import LANG_CODE_TO_NAME. Using raw language codes.")
    LANG_CODE_TO_NAME = {}

def get_lang_name(code):
    """Helper to get full language name, falling back to code if missing."""
    return LANG_CODE_TO_NAME.get(code, code)

def plot_source_competence(df, save_dir, model_name, model_save_name):
    """Plots Source PER vs Average Outbound BLEU."""
    plt.figure(figsize=(10, 6))
    
    # Calculate correlations
    r, _ = pearsonr(df['src_per'], df['outbound_bleu'])
    rho, _ = spearmanr(df['src_per'], df['outbound_bleu'])
    
    sns.scatterplot(data=df, x='src_per', y='outbound_bleu', s=100, color='blue', alpha=0.7)
    
    # Annotate points with language names
    for i, row in df.iterrows():
        plt.text(row['src_per'], row['outbound_bleu'] + 0.5, row['lang_name'], 
                 fontsize=9, ha='center')

    plt.title(f"Source Competence vs. Outbound Quality ({model_name})\nPearson: {r:.3f} | Spearman: {rho:.3f}")
    plt.xlabel("Source Perplexity Error Rate")
    plt.ylabel("Average Source Language BLEU")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_save_name}_source_competence.png"), dpi=300)
    plt.close()

def plot_target_competence(df, save_dir, model_name, model_save_name):
    """Plots Target PER vs Average Inbound BLEU."""
    plt.figure(figsize=(10, 6))
    
    # Calculate correlations
    r, _ = pearsonr(df['tgt_per'], df['inbound_bleu'])
    rho, _ = spearmanr(df['tgt_per'], df['inbound_bleu'])
    
    sns.scatterplot(data=df, x='tgt_per', y='inbound_bleu', s=100, color='green', alpha=0.7)
    
    # Annotate points
    for i, row in df.iterrows():
        plt.text(row['tgt_per'], row['inbound_bleu'] + 0.5, row['lang_name'], 
                 fontsize=9, ha='center')

    plt.title(f"Target Competence vs. Inbound Quality ({model_name})\nPearson: {r:.3f} | Spearman: {rho:.3f}")
    plt.xlabel("Source Perplexity Error Rate")
    plt.ylabel("Average Target Language BLEU")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_save_name}_target_competence.png"), dpi=300)
    plt.close()

def plot_joint_competence(df_bleu, per_dict, save_dir, model_name, model_save_name, sort_by_per=False):
    """Plots a 2D Scatter plot with colored points for BLEU, and a sorted categorical heatmap."""
    # 1. Bivariate Scatter Plot (Numeric: Src PER x Tgt PER -> Color=BLEU)
    plt.figure(figsize=(9, 7))
    
    # Filter out identical src-tgt pairs if they exist
    df_plot = df_bleu[df_bleu['src'] != df_bleu['tgt']].copy()
    
    # Map PERs
    df_plot['src_per'] = df_plot['src'].map(per_dict)
    df_plot['tgt_per'] = df_plot['tgt'].map(per_dict)
    
    # Drop NaNs
    df_plot = df_plot.dropna(subset=['src_per', 'tgt_per', 'bleu'])

    # Replaced contour plot with scatter plot with color mapping
    sc = plt.scatter(df_plot['src_per'], df_plot['tgt_per'], c=df_plot['bleu'], 
                     cmap='viridis', s=60, alpha=0.8, edgecolor='k', linewidth=0.5)
    
    plt.colorbar(sc, label='Translation BLEU Score')
    
    plt.title(f"Joint Competence Scatter ({model_name})\nBLEU as a function of Source and Target PER")
    plt.xlabel("Source PER")
    plt.ylabel("Target PER")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    # Renamed file to reflect scatter plot type
    plt.savefig(os.path.join(save_dir, f"{model_save_name}_joint_competence_scatter.png"), dpi=300)
    plt.close()

    plt.tricontourf(df_plot['src_per'], df_plot['tgt_per'], df_plot['bleu'], 
                    levels=15, cmap='viridis')
    plt.colorbar(label='Translation BLEU Score')
    plt.scatter(df_plot['src_per'], df_plot['tgt_per'], c='black', s=10, alpha=0.5)
    
    plt.title(f"Joint Competence Contour ({model_name})\nBLEU as a function of Source and Target PER")
    plt.xlabel("Source PER")
    plt.ylabel("Target PER")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_save_name}_joint_competence_contour.png"), dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze PER vs BLEU correlations.")
    parser.add_argument("--per_json", type=str, required=True, help="Path to JSON containing language PERs.")
    parser.add_argument("--bleu_csv", type=str, required=True, help="Path to CSV containing src, tgt, bleu.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the plots.")
    parser.add_argument("--sorted", action="store_true", help="Sort language axes by PER instead of alphabetically.")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Determine model name
    model_name = "Llama-3.1-8B" if "llama" in args.bleu_csv.lower() else "Aya-23-8B" if "aya" in args.bleu_csv.lower() else "unknown"
    model_save_name = "llama" if "llama" in args.bleu_csv.lower() else "aya" if "aya" in args.bleu_csv.lower() else "unknown"

    # Load data
    with open(args.per_json, 'r') as f:
        per_dict = json.load(f)
    
    df_bleu = pd.read_csv(args.bleu_csv)

    # Calculate aggregations
    outbound = df_bleu.groupby('src')['bleu'].mean().reset_index()
    outbound.columns = ['lang', 'outbound_bleu']
    
    inbound = df_bleu.groupby('tgt')['bleu'].mean().reset_index()
    inbound.columns = ['lang', 'inbound_bleu']

    # Combine with PER data and language names
    df_agg = pd.DataFrame({'lang': list(per_dict.keys())})
    df_agg['lang_name'] = df_agg['lang'].apply(get_lang_name)
    df_agg['per'] = df_agg['lang'].map(per_dict)
    
    df_source = pd.merge(df_agg, outbound, on='lang', how='inner')
    df_source.rename(columns={'per': 'src_per'}, inplace=True)
    
    df_target = pd.merge(df_agg, inbound, on='lang', how='inner')
    df_target.rename(columns={'per': 'tgt_per'}, inplace=True)

    # Generate Plots
    print(f"Generating plots for model: {model_name.upper()}")
    plot_source_competence(df_source, args.save_dir, model_name, model_save_name)
    plot_target_competence(df_target, args.save_dir, model_name, model_save_name)
    plot_joint_competence(df_bleu, per_dict, args.save_dir, model_name, model_save_name, args.sorted)
    
    print(f"Success! Visualizations saved to {args.save_dir}/")

if __name__ == "__main__":
    main()

# python scripts/visualize_perplexity_bleu_correlation.py --per_json outputs/perplexity_comparison/error_rates_by_language_meta-llama_Meta-Llama-3_1-8B_jumelet_multiblimp.json --bleu_csv outputs/perplexity_bleu/bleu_results_llama.csv --save_dir img/perplexity_bleu --sorted
# python scripts/visualize_perplexity_bleu_correlation.py --per_json outputs/perplexity_comparison/error_rates_by_language_CohereLabs_aya-23-8B_jumelet_multiblimp.json --bleu_csv outputs/perplexity_bleu/bleu_results_aya.csv --save_dir img/perplexity_bleu --sorted

# python scripts/visualize_perplexity_bleu_correlation.py --per_json outputs/perplexity_comparison/error_rates_by_language_meta-llama_Meta-Llama-3_1-8B_jumelet_multiblimp.json --bleu_csv outputs/perplexity_bleu/bleu_results_llama.csv --save_dir img/perplexity_bleu
# python scripts/visualize_perplexity_bleu_correlation.py --per_json outputs/perplexity_comparison/error_rates_by_language_CohereLabs_aya-23-8B_jumelet_multiblimp.json --bleu_csv outputs/perplexity_bleu/bleu_results_aya.csv --save_dir img/perplexity_bleu