import argparse
import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main(args):
    # 1. Collect all data into one DataFrame
    all_files = glob.glob(os.path.join(args.input_dir, "results_*.jsonl"))
    
    df_list = []
    
    print(f"Found {len(all_files)} files in {args.input_dir}")
    
    for filename in all_files:
        try:
            # Read each file
            temp_df = pd.read_json(filename, lines=True)
            
            # If the 'experiment' column is missing in the json, infer it from filename
            # filename format: results_{experiment}.jsonl
            if 'experiment' not in temp_df.columns:
                basename = os.path.basename(filename)
                exp_name = basename.replace("results_", "").replace(".jsonl", "")
                temp_df['experiment'] = exp_name
            
            df_list.append(temp_df)
        except ValueError as e:
            print(f"Skipping {filename}: {e}")

    if not df_list:
        print("No data found. Exiting.")
        return

    df = pd.concat(df_list, ignore_index=True)

    # 2. Filter: only include num_samples == 256
    df = df[df['num_samples'] == 128]
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # 3. Process each group (k / concept / value)
    # We group by these columns to create isolated plots for each configuration
    groups = df.groupby(['concept', 'value', 'k'])

    for name, group_df in groups:
        concept, value, k = name
        print(f"Processing group: Concept={concept}, Value={value}, k={k}")

        # --- Plot 1: Source Language Analysis ---
        # Group by Source Lang + Experiment
        plt.figure(figsize=(12, 6))
        
        # seaborn's barplot automatically handles the "group by X, hue by Experiment" logic
        # resulting in "Turkish + multi_output" bars next to "Turkish + mono_input"
        sns.barplot(
            data=group_df,
            x='source_lang',
            y='mean_delta',
            hue='experiment',
            errorbar=None  # Remove CI bars if simple mean is preferred
        )
        
        plt.title(f"Source Language Ablation Impact\n(Concept: {concept}={value}, k={k})")
        plt.xlabel("Source Language")
        plt.ylabel("Mean Delta")
        plt.legend(title="Experiment", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save Source Plot
        src_filename = f"barplot_source_{concept}_{value}_{k}.png"
        plt.savefig(os.path.join(args.output_dir, src_filename))
        plt.close()

        # --- Plot 2: Target Language Analysis ---
        # Only relevant if target_lang is NOT null
        target_df = group_df.dropna(subset=['target_lang'])
        
        if not target_df.empty:
            plt.figure(figsize=(12, 6))
            sns.barplot(
                data=target_df,
                x='target_lang',
                y='mean_delta',
                hue='experiment',
                errorbar=None
            )
            
            plt.title(f"Target Language Ablation Impact\n(Concept: {concept}={value}, k={k})")
            plt.xlabel("Target Language")
            plt.ylabel("Mean Delta")
            plt.legend(title="Experiment", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            # Save Target Plot
            tgt_filename = f"barplot_target_{concept}_{value}_{k}.png"
            plt.savefig(os.path.join(args.output_dir, tgt_filename))
            plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize ablation results")
    parser.add_argument("--input_dir", type=str, required=True, help="Input dir containing results_*.jsonl files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for plots")
    args = parser.parse_args()
    main(args)
