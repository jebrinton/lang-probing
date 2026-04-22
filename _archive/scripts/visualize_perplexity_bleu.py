import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


def main(args):
    # 1. Load your data
    # Replace 'bleu_scores.csv' and 'perplexity.csv' with your actual file paths
    df_bleu = pd.read_csv(args.bleu_csv)
    df_ppl = pd.read_csv(args.perplexity_csv)

    # 2. Filter for the model you are interested in (e.g., "lla")
    df_bleu = df_bleu[df_bleu['model'] == 'lla'].copy()

    # 3. Merge Perplexity Data
    # We merge perplexity into the Target Language column to see if Target difficulty affects BLEU
    df_merged = df_bleu.merge(df_ppl, left_on='tgt', right_on='Language', how='inner')
    df_merged.rename(columns={'Perplexity': 'tgt_ppl'}, inplace=True)

    # Optional: You could also merge on 'src' if you wanted to test Source Perplexity
    # df_merged = df_merged.merge(df_ppl, left_on='src', right_on='Language', how='inner')

    # 4. Fit the Linear Mixed-Effects Model
    # Formula: BLEU is predicted by Target Perplexity, grouping by Source Language
    model = smf.mixedlm("bleu ~ tgt_ppl", df_merged, groups=df_merged["src"])
    result = model.fit()

    print(result.summary())

    # 5. Plotting Perplexity vs BLEU
    plt.figure(figsize=(10, 6))

    # Plot scatter points colored by the Group (Source Language)
    groups = df_merged['src'].unique()
    colors = plt.cm.jet(np.linspace(0, 1, len(groups)))

    for group, color in zip(groups, colors):
        subset = df_merged[df_merged['src'] == group]
        plt.scatter(subset['tgt_ppl'], subset['bleu'], label=group, color=color, s=100, alpha=0.7)

    # Plot the global trend line (Fixed Effect)
    x_vals = np.linspace(df_merged['tgt_ppl'].min(), df_merged['tgt_ppl'].max(), 100)
    params = result.params
    y_vals = params['Intercept'] + params['tgt_ppl'] * x_vals
    plt.plot(x_vals, y_vals, 'k--', linewidth=2, label='Global Trend (Fixed Effect)')

    plt.xlabel('Target Language Perplexity')
    plt.ylabel('BLEU Score')
    plt.title('Effect of Target Perplexity on BLEU\n(Grouped by Source Language)')
    plt.legend(title='Source Language', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    os.makedirs("/projectnb/mcnet/jbrin/lang-probing/img/perplexity_bleu", exist_ok=True)
    model_name = "llama" if "llama" in args.model_id else "aya"
    plt.savefig(f"/projectnb/mcnet/jbrin/lang-probing/img/perplexity_bleu/perplexity_vs_bleu_{model_name}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--bleu_csv", type=str, default="/projectnb/mcnet/jbrin/lang-probing/outputs/perplexity_bleu/bleu_results.csv")
    parser.add_argument("--perplexity_csv", type=str, default="/projectnb/mcnet/jbrin/lang-probing/outputs/perplexity_bleu/perplexity_results_llama.csv")
    args = parser.parse_args()
    main(args)