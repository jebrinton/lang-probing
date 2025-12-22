import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os

def main(args):
    # Load your data
    # Assuming the data is in a file named 'results.jsonl'
    df = pd.read_json(args.input_file, lines=True)

    # 1. Filter by num_samples == 256
    df = df[df['num_samples'] == 256]

    # 2. Exclude null values (specifically where target_lang is missing)
    df = df.dropna(subset=['target_lang'])

    # 3. Identify unique experimental configurations to create separate plots
    #    (Since you have different 'k' and 'value' settings, mixing them in one heatmap 
    #     would cause overwrites or aggregation errors).
    groups = df.groupby(['experiment', 'concept', 'value', 'k'])

    for name, group in groups:
        experiment, concept, value, k = name
        
        # 4. Pivot the data: Rows=source_lang, Cols=target_lang, Vals=mean_delta
        heatmap_data = group.pivot(index='source_lang', columns='target_lang', values='mean_delta')
        
        # 5. Plotting
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', center=0)
        
        # Set the title dynamically
        plt.title(f"{experiment} on {concept}={value}; ablating {k} features")
        plt.xlabel("Target Language")
        plt.ylabel("Source Language")
        
        plt.tight_layout()
        
        # Save or show
        filename = f"heatmap_{experiment}_{concept}_{value}_{k}.png"
        plt.savefig(os.path.join(args.output_dir, filename))
        plt.close() # Close figure to free memory

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize ablation results")
    parser.add_argument("--input_file", type=str, required=True, help="Input file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()
    main(args)