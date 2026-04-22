import argparse
import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def _ylabel_for_column(y_col: str) -> str:
    if y_col == "mean_logprob_delta":
        return "Mean Δ log p (reference tokens)"
    if y_col == "mean_delta":
        return "Mean (exp(Δ log p) − 1)  [≈ rel. prob change]"
    return y_col


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

    y_col = args.y_column
    if y_col not in df.columns:
        fallback = "mean_delta" if "mean_delta" in df.columns else None
        if fallback is None:
            print(f"Column {y_col!r} not found and no mean_delta; exiting.")
            return
        print(f"Column {y_col!r} not found; using {fallback!r} instead.")
        y_col = fallback

    # Language of the ablated segment: for mono_output source_lang is null, we ablate target-language text
    df["ablate_lang"] = df["source_lang"].fillna(df["target_lang"])

    # 2. Optional filters (only apply when provided)
    if args.num_samples is not None:
        df = df[df["num_samples"] == args.num_samples]
    if args.probe_layer is not None and "probe_layer" in df.columns:
        df = df[df["probe_layer"] == args.probe_layer]

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # 3. Process each group (k / concept / value)
    # We group by these columns to create isolated plots for each configuration
    groups = df.groupby(['concept', 'value', 'k'])

    for name, group_df in groups:
        concept, value, k = name
        print(f"Processing group: Concept={concept}, Value={value}, k={k}")

        # --- Plot 1: Ablation Language Analysis ---
        # Use ablate_lang so mono_output (source_lang=null) appears by target_lang
        plot_df = group_df.dropna(subset=["ablate_lang"])
        if plot_df.empty:
            continue
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=plot_df,
            x="ablate_lang",
            y=y_col,
            hue="experiment",
            errorbar=None,
        )
        plt.title(f"Ablation Language Impact\n(Concept: {concept}={value}, k={k})")
        plt.xlabel("Ablation Language")
        plt.ylabel(_ylabel_for_column(y_col))
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
                y=y_col,
                hue='experiment',
                errorbar=None
            )
            
            plt.title(f"Target Language Ablation Impact\n(Concept: {concept}={value}, k={k})")
            plt.xlabel("Target Language")
            plt.ylabel(_ylabel_for_column(y_col))
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
    parser.add_argument("--num_samples", type=int, default=None, help="Filter to this num_samples (default: no filter)")
    parser.add_argument("--probe_layer", type=int, default=None, help="Filter to this probe_layer (default: no filter; omit for non-probe runs)")
    parser.add_argument(
        "--y_column",
        type=str,
        default="mean_logprob_delta",
        choices=("mean_logprob_delta", "mean_delta"),
        help="Metric to plot: mean_logprob_delta (Δ log p) or mean_delta (exp(Δ log p)-1). Falls back if column missing.",
    )
    args = parser.parse_args()
    main(args)
