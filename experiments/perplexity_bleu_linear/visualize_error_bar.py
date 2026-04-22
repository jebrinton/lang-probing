import json
import argparse
import matplotlib.pyplot as plt
from lang_probing_src.config import LANG_CODE_TO_NAME
import os

def plot_perplexity_error_rates(json_path, output_dir, sort_by_rate=False):
    model_name = "Llama-3.1-8B" if "llama" in json_path else "Aya-23-8B" if "aya" in json_path else "unknown"

    # Load the JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Map language codes to names and pair with their error rates
    mapped_data = []
    for code, rate in data.items():
        # Use the imported mapping, fallback to the code itself if missing
        lang_name = LANG_CODE_TO_NAME.get(code, code)
        mapped_data.append((lang_name, rate))

    # Sort the data based on the flag
    if sort_by_rate:
        # Sort by error rate (ascending)
        mapped_data.sort(key=lambda x: x[1])
    else:
        # Sort alphabetically by language name
        mapped_data.sort(key=lambda x: x[0])

    # Unpack the sorted data into separate lists for plotting
    languages = [x[0] for x in mapped_data]
    rates = [x[1] for x in mapped_data]

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.bar(languages, rates, color='skyblue', edgecolor='black', zorder=2)

    # Add labels and title
    plt.xlabel('Language', fontsize=12, fontweight='bold')
    plt.ylabel('Perplexity Error Rate', fontsize=12, fontweight='bold')
    plt.title(f'Perplexity Error Rate by Language: {model_name}', fontsize=14)

    # Rotate x-axis labels to prevent overlap and align them nicely
    plt.xticks(rotation=45, ha='right')
    
    # Add a horizontal grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=1)

    # Scale the y-axis to utilize the whole plot space tightly around the min/max values
    min_rate = min(rates)
    max_rate = max(rates)
    y_padding = (max_rate - min_rate) * 0.05  # Add a 5% margin to top and bottom
    
    # Ensure the minimum limit doesn't drop below 0
    plt.ylim(max(0, min_rate - y_padding), max_rate + y_padding)

    # Automatically adjust layout so labels are not truncated
    plt.tight_layout()

    # Determine model name and adjust filename based on sorting flag
    suffix = "_sorted" if sort_by_rate else ""
    save_path = os.path.join(output_dir, f"perplexity_plot_{model_name}{suffix}.png")
    
    # Save the plot to the given file path
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved successfully to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot language perplexity error rates from a JSON file.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the JSON file containing the error rates.")
    parser.add_argument("--output_dir", type=str, default="img/perplexity_bleu_linear", help="Output directory for the saved plot image.")
    parser.add_argument("--sort_by_rate", action="store_true", help="Flag to sort the chart by perplexity error rate. Defaults to alphabetical sort by language if omitted.")
    args = parser.parse_args()

    # Ensure output directory exists to prevent FileNotFoundError
    os.makedirs(args.output_dir, exist_ok=True)

    plot_perplexity_error_rates(args.json_path, args.output_dir, args.sort_by_rate)