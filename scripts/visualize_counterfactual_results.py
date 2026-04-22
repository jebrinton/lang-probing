"""
Visualize counterfactual attribution results.

Loads per-pair and aggregated results produced by scripts/counterfactual_attribution.py
and produces publication-quality matplotlib/seaborn plots.

Plots produced:
    1. Per-concept top-N feature importance bar charts (one figure per concept)
       — by mean |grad| and by mean |grad*act|, saved as two PNGs per concept.
    2. Cross-concept feature overlap heatmap — Jaccard similarity of each concept's
       top-K sets (one figure for grad ranking, one for grad*act ranking).
    3. Distribution of metric_value (logP_orig - logP_cf) across pairs, grouped by
       concept — a violin+strip plot.
    4. Per-pair top features scatter — activation (x) vs |grad| (y) across all pairs,
       faceted by concept.

Usage:
    python scripts/visualize_counterfactual_results.py \
        --results_dir outputs/counterfactual_attribution \
        --output_dir outputs/counterfactual_attribution/plots
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lang_probing_src.config import OUTPUTS_DIR


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_results(results_dir):
    """Load per-pair and aggregated results. Returns (per_pair, aggregated).

    Either may be None if the corresponding file is missing.
    """
    per_pair_path = os.path.join(results_dir, "per_pair_results.json")
    agg_path = os.path.join(results_dir, "aggregated_by_concept.json")

    per_pair = None
    aggregated = None

    if os.path.exists(per_pair_path):
        with open(per_pair_path) as f:
            per_pair = json.load(f)
    else:
        print(f"WARNING: {per_pair_path} not found; per-pair plots will be skipped.")

    if os.path.exists(agg_path):
        with open(agg_path) as f:
            aggregated = json.load(f)
    else:
        print(f"WARNING: {agg_path} not found; aggregated plots will be skipped.")

    return per_pair, aggregated


# ---------------------------------------------------------------------------
# Plot 1: per-concept top-N feature bar charts
# ---------------------------------------------------------------------------

def plot_per_concept_bars(aggregated, output_dir, top_n=20):
    """One bar chart per (concept, ranking). X = feature_idx, Y = metric value."""
    if not aggregated:
        print("Skipping per-concept bar charts: no aggregated data.")
        return []

    written = []
    rankings = [
        ("top_50_by_mean_grad", "mean_abs_grad", "mean |gradient|"),
        ("top_50_by_mean_grad_x_act", "mean_abs_grad_x_act", "mean |gradient * activation|"),
    ]

    for concept, data in sorted(aggregated.items()):
        n_pairs = data.get("num_pairs", 0)
        for ranking_key, value_key, pretty in rankings:
            features = data.get(ranking_key, [])[:top_n]
            if not features:
                print(
                    f"Skipping bar chart for concept={concept}, ranking={ranking_key}: no features."
                )
                continue

            df = pd.DataFrame(features)
            # Build x labels as strings so matplotlib treats them as categorical
            df["feature_label"] = df["feature_idx"].astype(str)

            fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(df) + 4), 5))
            sns.barplot(
                data=df,
                x="feature_label",
                y=value_key,
                color=sns.color_palette("viridis", as_cmap=False)[3],
                ax=ax,
            )
            ax.set_title(
                f"Top {len(df)} features for {concept} by {pretty}\n"
                f"({n_pairs} pairs)"
            )
            ax.set_xlabel("SAE feature index")
            ax.set_ylabel(pretty)
            plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
            plt.tight_layout()

            short = "grad" if ranking_key == "top_50_by_mean_grad" else "grad_x_act"
            fname = f"bar_top{len(df)}_{concept}_{short}.png"
            fpath = os.path.join(output_dir, fname)
            fig.savefig(fpath, dpi=150)
            plt.close(fig)
            written.append(fpath)

    return written


# ---------------------------------------------------------------------------
# Plot 2: cross-concept Jaccard heatmap
# ---------------------------------------------------------------------------

def _jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def plot_jaccard_heatmap(aggregated, output_dir, top_k=50, ranking="top_50_by_mean_grad"):
    """Heatmap of pairwise Jaccard similarity between concept top-K feature sets."""
    if not aggregated:
        print(f"Skipping Jaccard heatmap ({ranking}): no aggregated data.")
        return None

    concept_features = {}
    for concept, data in aggregated.items():
        feats = data.get(ranking, [])[:top_k]
        if feats:
            concept_features[concept] = set(f["feature_idx"] for f in feats)

    if len(concept_features) < 2:
        print(
            f"Skipping Jaccard heatmap ({ranking}): need >=2 concepts with features, "
            f"got {len(concept_features)}."
        )
        return None

    concepts = sorted(concept_features.keys())
    mat = np.zeros((len(concepts), len(concepts)), dtype=float)
    for i, c1 in enumerate(concepts):
        for j, c2 in enumerate(concepts):
            mat[i, j] = _jaccard(concept_features[c1], concept_features[c2])

    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(concepts) + 3),
                                    max(5, 0.7 * len(concepts) + 2)))
    sns.heatmap(
        mat,
        xticklabels=concepts,
        yticklabels=concepts,
        annot=True,
        fmt=".2f",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        square=True,
        cbar_kws={"label": "Jaccard similarity"},
        ax=ax,
    )
    pretty = "mean |grad|" if ranking == "top_50_by_mean_grad" else "mean |grad * activation|"
    ax.set_title(f"Cross-concept top-{top_k} feature Jaccard overlap\n(ranking: {pretty})")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()

    short = "grad" if ranking == "top_50_by_mean_grad" else "grad_x_act"
    fname = f"jaccard_top{top_k}_{short}.png"
    fpath = os.path.join(output_dir, fname)
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    return fpath


# ---------------------------------------------------------------------------
# Plot 3: metric_value distribution by concept
# ---------------------------------------------------------------------------

def plot_metric_distribution(per_pair, output_dir):
    """Violin + strip of logP_orig - logP_cf per concept."""
    if not per_pair:
        print("Skipping metric distribution plot: no per-pair data.")
        return None

    df = pd.DataFrame(
        [
            {"concept": r["concept"], "metric_value": r["metric_value"]}
            for r in per_pair
        ]
    )
    if df.empty:
        print("Skipping metric distribution plot: empty dataframe.")
        return None

    concepts = sorted(df["concept"].unique())

    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(concepts) + 3), 5))
    sns.violinplot(
        data=df,
        x="concept",
        y="metric_value",
        order=concepts,
        hue="concept",
        hue_order=concepts,
        inner=None,
        cut=0,
        palette="pastel",
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x="concept",
        y="metric_value",
        order=concepts,
        color="black",
        size=4,
        alpha=0.7,
        jitter=0.15,
        ax=ax,
    )
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1,
               label="indifferent (0)")
    ax.set_title(
        "Distribution of metric value (logP_orig - logP_cf) by concept\n"
        "positive -> model prefers the original token"
    )
    ax.set_xlabel("Concept")
    ax.set_ylabel("logP(orig) - logP(cf)")
    ax.legend(loc="best")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()

    fpath = os.path.join(output_dir, "metric_distribution_by_concept.png")
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    return fpath


# ---------------------------------------------------------------------------
# Plot 4: per-pair top features scatter (activation vs |grad|), faceted by concept
# ---------------------------------------------------------------------------

def plot_feature_scatter(per_pair, output_dir, top_n=20):
    """Scatter of activation (x) vs |grad| (y) for top-N features per pair, faceted by concept."""
    if not per_pair:
        print("Skipping feature scatter plot: no per-pair data.")
        return []

    rows = []
    for r in per_pair:
        concept = r["concept"]
        for feat in r.get("top_k_by_grad_last_pos", [])[:top_n]:
            rows.append(
                {
                    "concept": concept,
                    "pair_id": r["pair_id"],
                    "feature_idx": feat["feature_idx"],
                    "activation": feat["activation"],
                    "abs_grad": abs(feat["grad"]),
                    "abs_grad_x_act": abs(feat["grad_x_act"]),
                }
            )

    if not rows:
        print("Skipping feature scatter plot: no rows built.")
        return []

    df = pd.DataFrame(rows)
    concepts = sorted(df["concept"].unique())

    written = []

    # Faceted scatter (one subplot per concept) + one combined version.
    # --- Combined scatter (all concepts, colored) ---
    fig, ax = plt.subplots(figsize=(9, 6))
    palette = sns.color_palette("tab10", n_colors=len(concepts))
    for i, concept in enumerate(concepts):
        sub = df[df["concept"] == concept]
        ax.scatter(
            sub["activation"],
            sub["abs_grad"],
            label=f"{concept} (n={len(sub)})",
            alpha=0.6,
            s=30,
            color=palette[i],
            edgecolors="none",
        )
    ax.set_xlabel("SAE feature activation (last token position)")
    ax.set_ylabel("|gradient|")
    ax.set_title(
        f"Per-pair top-{top_n} features: activation vs |gradient|\n"
        f"(shared high-attribution features stand out; {len(df)} points)"
    )
    ax.legend(title="Concept", bbox_to_anchor=(1.02, 1), loc="upper left",
              frameon=False, fontsize=9)
    plt.tight_layout()
    fpath = os.path.join(output_dir, "feature_scatter_activation_vs_grad.png")
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    written.append(fpath)

    # --- Faceted (one panel per concept) ---
    try:
        g = sns.FacetGrid(
            df,
            col="concept",
            col_wrap=min(4, len(concepts)),
            height=3.3,
            aspect=1.2,
            sharex=True,
            sharey=True,
        )
        g.map_dataframe(
            sns.scatterplot,
            x="activation",
            y="abs_grad",
            alpha=0.7,
            s=25,
            color=sns.color_palette("deep")[0],
        )
        g.set_axis_labels("activation", "|gradient|")
        g.set_titles(col_template="{col_name}")
        g.figure.suptitle(
            f"Per-pair top-{top_n} features by concept (activation vs |gradient|)",
            y=1.02,
        )
        g.figure.tight_layout()
        fpath2 = os.path.join(output_dir, "feature_scatter_activation_vs_grad_facet.png")
        g.figure.savefig(fpath2, dpi=150, bbox_inches="tight")
        plt.close(g.figure)
        written.append(fpath2)
    except Exception as e:
        print(f"WARNING: faceted scatter failed: {e}")

    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize counterfactual attribution results."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join(OUTPUTS_DIR, "counterfactual_attribution"),
        help="Directory containing per_pair_results.json and aggregated_by_concept.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(OUTPUTS_DIR, "counterfactual_attribution", "plots"),
        help="Directory to save PNG plots into",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=20,
        help="Number of top features to show in per-concept bar charts and per-pair scatter.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-K feature set size for the Jaccard heatmap.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Seaborn global style
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.0)

    per_pair, aggregated = load_results(args.results_dir)

    written = []

    print("\n[1/4] Per-concept top-N feature bar charts...")
    written.extend(plot_per_concept_bars(aggregated, args.output_dir, top_n=args.top_n))

    print("\n[2/4] Cross-concept Jaccard heatmaps...")
    for ranking in ("top_50_by_mean_grad", "top_50_by_mean_grad_x_act"):
        fp = plot_jaccard_heatmap(aggregated, args.output_dir, top_k=args.top_k, ranking=ranking)
        if fp:
            written.append(fp)

    print("\n[3/4] Metric value distribution by concept...")
    fp = plot_metric_distribution(per_pair, args.output_dir)
    if fp:
        written.append(fp)

    print("\n[4/4] Per-pair top-features scatter (activation vs |grad|)...")
    written.extend(plot_feature_scatter(per_pair, args.output_dir, top_n=args.top_n))

    print("\n" + "=" * 70)
    print(f"Wrote {len(written)} PNG(s) to {args.output_dir}:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
