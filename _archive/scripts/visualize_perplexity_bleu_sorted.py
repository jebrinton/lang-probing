"""
Visualize the perplexity vs BLEU sorted by difficulty. Same thing as visualize_perplexity_bleu.py, but x-axis is labeled with language codes.
"""
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def main(args):
    # --- 1. Load Data ---
    print(f"Loading BLEU from: {args.bleu_csv}")
    print(f"Loading Perplexity from: {args.perplexity_csv}")
    
    df_bleu = pd.read_csv(args.bleu_csv)
    df_ppl = pd.read_csv(args.perplexity_csv)

    # --- 2. Filter for the specific model ---
    # Determine short name ('lla' or 'aya') based on the input arg
    if "llama" in args.model_id.lower():
        short_model_name = "lla" 
        save_name = "llama_sorted"
    else:
        short_model_name = "aya"
        save_name = "aya_sorted"

    print(f"Filtering for model short-code: {short_model_name}")
    df_bleu = df_bleu[df_bleu['model'] == short_model_name].copy()

    # --- 3. Merge Perplexity Data ---
    # Merge perplexity into the Target Language column
    df_merged = df_bleu.merge(df_ppl, left_on='tgt', right_on='Language', how='inner')
    df_merged.rename(columns={'Perplexity': 'tgt_ppl'}, inplace=True)

    # --- 4. Fit the Linear Mixed-Effects Model ---
    print("Fitting Linear Mixed-Effects Model...")
    model = smf.mixedlm("bleu ~ tgt_ppl", df_merged, groups=df_merged["src"])
    result = model.fit()
    print(result.summary())

    # Ensure output directory exists
    output_dir = "/projectnb/mcnet/jbrin/lang-probing/img/perplexity_bleu"
    os.makedirs(output_dir, exist_ok=True)

    # ==========================================
    # PLOT 1: Scatter (Perplexity Value vs BLEU)
    # ==========================================
    plt.figure(figsize=(10, 6))

    groups = df_merged['src'].unique()
    # Create a colormap
    colors = plt.cm.jet(np.linspace(0, 1, len(groups)))
    
    # Scatter plot loop
    for group, color in zip(groups, colors):
        subset = df_merged[df_merged['src'] == group]
        plt.scatter(subset['tgt_ppl'], subset['bleu'], label=group, color=color, s=100, alpha=0.7)

    # Global trend line
    x_vals = np.linspace(df_merged['tgt_ppl'].min(), df_merged['tgt_ppl'].max(), 100)
    params = result.params
    y_vals = params['Intercept'] + params['tgt_ppl'] * x_vals
    plt.plot(x_vals, y_vals, 'k--', linewidth=2, label='Global Trend (Fixed Effect)')

    plt.xlabel('Target Language Perplexity')
    plt.ylabel('BLEU Score')
    plt.title(f'Effect of Target Perplexity on BLEU ({save_name})\n(Grouped by Source Language)')
    plt.legend(title='Source Language', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path_1 = f"{output_dir}/perplexity_vs_bleu_{save_name}.png"
    plt.savefig(save_path_1)
    print(f"Saved scatter plot to: {save_path_1}")
    plt.close() # Close figure to free memory

    # ==========================================
    # PLOT 2: Sorted by Difficulty (Language Name vs BLEU)
    # ==========================================
    # 1. Sort the dataframe by target perplexity (Low -> High)
    df_sorted = df_merged.sort_values(by='tgt_ppl', ascending=True)

    # 2. Extract unique targets in sorted order for the X-axis
    sorted_targets = df_sorted['tgt'].unique()
    # We need to sort these unique targets by their perplexity to ensure the axis is correct
    # (The dataframe sort above sorts the rows, but we need the unique list for the X-axis labels)
    target_ppl_map = df_sorted[['tgt', 'tgt_ppl']].drop_duplicates().set_index('tgt')['tgt_ppl']
    sorted_targets = sorted(sorted_targets, key=lambda x: target_ppl_map[x])

    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()

    # 3. Scatter Plot (BLEU) on primary axis
    # We map languages to integer indices for plotting, then swap labels back
    x_indices = range(len(sorted_targets))
    lang_to_idx = {lang: i for i, lang in enumerate(sorted_targets)}
    
    for group, color in zip(groups, colors):
        subset = df_sorted[df_sorted['src'] == group]
        # Map the 'tgt' strings to their sorted x-index
        subset_x = subset['tgt'].map(lang_to_idx)
        ax1.scatter(subset_x, subset['bleu'], label=group, color=color, s=100, alpha=0.8)

    ax1.set_ylabel('BLEU Score', fontweight='bold')
    ax1.set_xlabel('Target Language (Sorted by Perplexity)', fontweight='bold')
    
    # Set X-ticks to be the language names
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(sorted_targets, rotation=45)

    # 4. Line Plot (Perplexity) on secondary axis
    ax2 = ax1.twinx()
    # Get the perplexity value for each sorted language
    sorted_ppl_values = [target_ppl_map[lang] for lang in sorted_targets]
    
    ax2.plot(x_indices, sorted_ppl_values, color='gray', linestyle='--', alpha=0.5, linewidth=2, label='Perplexity Score')
    ax2.set_ylabel('Perplexity (Lower is Better)', color='gray', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='gray')

    plt.title(f'BLEU Scores by Target Language Difficulty ({save_name})\n(Sorted: Easy -> Hard)', fontsize=14)
    
    # Combine legends? Or just keep Source Legend
    # Placing the legend outside
    ax1.legend(title='Source Language', bbox_to_anchor=(1.10, 1), loc='upper left')

    plt.tight_layout()
    
    save_path_2 = f"{output_dir}/perplexity_sorted_vs_bleu_{save_name}.png"
    plt.savefig(save_path_2)
    print(f"Saved sorted plot to: {save_path_2}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--bleu_csv", type=str, default="/projectnb/mcnet/jbrin/lang-probing/outputs/perplexity_bleu/bleu_results.csv")
    parser.add_argument("--perplexity_csv", type=str, default="/projectnb/mcnet/jbrin/lang-probing/outputs/perplexity_bleu/perplexity_results_llama.csv")
    args = parser.parse_args()
    main(args)