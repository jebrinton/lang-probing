"""
Improved plot suite (post-review):
- Cross-concept histograms: # features at each cell-count tier (hundred-fold
  clearer than the previous sorted-bar plot for understanding the distribution)
- Tok-strategy: raw counts AND fraction-per-cell for cross-language comparison
- §4.4 per-cell ablation bar chart: visualizes the top-K vs random comparison
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_cross_concept_histograms(attr_dir, out_dir, topk=50, sae_dim=32768):
    """One histogram panel per language: x = # cells feature landed top-K in,
    y = # features at that count."""
    langs = ["eng", "fra", "spa", "tur", "ara"]
    lang_data = {}
    for lang in langs:
        cells = []
        for cell_dir in sorted((Path(attr_dir) / lang).iterdir()):
            if not cell_dir.is_dir() or "_" not in cell_dir.name:
                continue
            sp = cell_dir / "summary.json"
            if not sp.exists():
                continue
            with open(sp) as f:
                s = json.load(f)
            cells.append(s.get("top_50_by_abs_gxa", []))
        if not cells:
            continue
        n_cells = len(cells)
        feat_counts = defaultdict(int)
        for cell in cells:
            for f in cell[:topk]:
                feat_counts[f["feature_idx"]] += 1
        # Distribution: how many features at each count
        dist = defaultdict(int)
        for feat, cnt in feat_counts.items():
            dist[cnt] += 1
        lang_data[lang] = (n_cells, dist)

    # One figure, one panel per language
    n = len(lang_data)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, (lang, (n_cells, dist)) in zip(axes, lang_data.items()):
        xs = np.arange(1, n_cells + 1)
        ys = [dist.get(i, 0) for i in xs]
        # Expected under null: E[X] = total_topk_slots_covered * binomial pmf
        n_feats_total = sum(dist.values())
        p_null = topk / sae_dim
        # Each feature: P(count = k) = Binomial(n_cells, k, p_null)
        # Total across all features that hit at least once is not quite
        # Binomial-distributed (conditioned on ≥1), so draw the unconditional
        # expected-count line for reference
        from scipy import stats
        expected = n_feats_total * np.array(
            [stats.binom.pmf(k, n_cells, p_null) /
             (1 - stats.binom.pmf(0, n_cells, p_null))
             for k in xs]
        )
        ax.bar(xs, ys, color="steelblue", alpha=0.85, label="observed")
        ax.plot(xs, expected, "r--", label="null (i.i.d. top-50)", alpha=0.6)
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xlabel("# cells in top-50")
        ax.set_ylabel("# features" if lang == list(lang_data.keys())[0] else "")
        ax.set_title(f"{lang}  ({n_cells} cells)")
        if lang == list(lang_data.keys())[0]:
            ax.legend(fontsize=8)
    fig.suptitle("How many features land in top-50 of how many cells?")
    plt.tight_layout()
    fig.savefig(out_dir / "fig_cross_concept_histograms.png", dpi=110)
    plt.close(fig)
    print(f"  wrote fig_cross_concept_histograms.png")


def plot_tok_strategy_with_ratio(attr_dir, out_dir):
    """Two-panel: raw counts and fraction multi-token per cell, grouped by lang."""
    rows = []
    for lang_dir in sorted(Path(attr_dir).iterdir()):
        if not lang_dir.is_dir():
            continue
        for cell_dir in sorted(lang_dir.iterdir()):
            if not cell_dir.is_dir() or "_" not in cell_dir.name:
                continue
            sp = cell_dir / "summary.json"
            if not sp.exists():
                continue
            with open(sp) as f:
                s = json.load(f)
            rows.append({
                "lang": lang_dir.name,
                "cell": cell_dir.name,
                "n_pairs": s.get("n_pairs", 0),
                "n_single": s.get("n_single", 0),
                "n_last": s.get("n_last", 0),
            })
    df = pd.DataFrame(rows)
    df["frac_multi"] = df["n_last"] / df["n_pairs"].clip(lower=1)
    df = df.sort_values(["lang", "cell"]).reset_index(drop=True)

    lang_colors = {"eng": "tab:gray", "fra": "tab:blue",
                   "spa": "tab:orange", "tur": "tab:green", "ara": "tab:red"}

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    xs = np.arange(len(df))
    w = 0.4

    # Panel 1: stacked counts
    axes[0].bar(xs, df["n_single"], w, color="lightblue", label="single-BPE")
    axes[0].bar(xs, df["n_last"], w, bottom=df["n_single"],
                color="coral", label="multi (last-BPE)")
    axes[0].set_ylabel("# pairs")
    axes[0].set_title("Per-cell: absolute pair counts, split by tokenization strategy")
    axes[0].legend()

    # Panel 2: fraction multi
    bar_colors = [lang_colors.get(l, "gray") for l in df["lang"]]
    axes[1].bar(xs, df["frac_multi"], color=bar_colors)
    axes[1].axhline(0.5, color="k", linestyle=":", alpha=0.4)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("fraction multi-BPE")
    axes[1].set_title("Fraction of pairs needing last-BPE approximation "
                      "(cross-language comparable)")
    # Per-language bars for legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in lang_colors.values()]
    axes[1].legend(handles, lang_colors.keys(), loc="upper left", fontsize=8)

    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(
        [f"{r['lang']}/{r['cell']}" for _, r in df.iterrows()],
        rotation=75, fontsize=7, ha="right"
    )
    plt.tight_layout()
    fig.savefig(out_dir / "fig_tok_strategy_v2.png", dpi=110)
    plt.close(fig)
    print(f"  wrote fig_tok_strategy_v2.png")

    # Also a simple per-lang summary: aggregate fraction
    agg = df.groupby("lang").apply(
        lambda d: pd.Series({
            "total_pairs": d["n_pairs"].sum(),
            "total_single": d["n_single"].sum(),
            "total_multi": d["n_last"].sum(),
            "frac_multi": d["n_last"].sum() / max(1, d["n_pairs"].sum()),
        })
    ).reset_index()
    agg.to_csv(out_dir / "tok_strategy_per_lang.csv", index=False)
    print(f"  wrote tok_strategy_per_lang.csv")


def plot_ablation_bar_chart(abl_dir, out_dir):
    """§4.4 visualization: per-cell bars of top-K Δorig vs random-K baseline."""
    rows = []
    for lang_dir in sorted(Path(abl_dir).iterdir()):
        if not lang_dir.is_dir():
            continue
        for cell_dir in sorted(lang_dir.iterdir()):
            hp = cell_dir / "holdout_ablation.json"
            if not hp.exists():
                continue
            with open(hp) as f:
                h = json.load(f)
            concept, value = cell_dir.name.split("_", 1)
            rows.append({
                "lang": lang_dir.name,
                "cell": cell_dir.name,
                "concept": concept,
                "value": value,
                "delta_top": h["top_result"]["mean_delta_orig"],
                "delta_rand": h["random_result"]["mean_delta_orig"],
                "n_holdout": h["top_result"]["n"],
            })
    df = pd.DataFrame(rows)
    # Exclude tiny-n cells for plotting (< 5), keep all for CSV
    df.to_csv(out_dir / "ablation_effects_all.csv", index=False)
    df_plot = df[df["n_holdout"] >= 5].sort_values(["lang", "cell"]).reset_index(drop=True)

    lang_colors = {"eng": "tab:gray", "fra": "tab:blue",
                   "spa": "tab:orange", "tur": "tab:green", "ara": "tab:red"}

    fig, ax = plt.subplots(figsize=(14, 5))
    xs = np.arange(len(df_plot))
    w = 0.38

    top_colors = [lang_colors.get(l, "gray") for l in df_plot["lang"]]
    ax.bar(xs - w/2, df_plot["delta_top"], w, color=top_colors,
           label="top-20 features", edgecolor="black", linewidth=0.5)
    ax.bar(xs + w/2, df_plot["delta_rand"], w, color="lightgray",
           label="random-20 baseline", edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{r['lang']}/{r['cell']} (n={r['n_holdout']})"
         for _, r in df_plot.iterrows()],
        rotation=70, fontsize=8, ha="right"
    )
    ax.set_ylabel(r"mean $\Delta$ logP(orig)   after ablation")
    ax.set_title("Causal validation: top-20 attribution features vs random-20 baseline\n"
                 "Negative = ablation hurts the correct prediction (what we expect)")
    # Per-language legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in lang_colors.values()]
    handles.append(plt.Rectangle((0, 0), 1, 1, color="lightgray"))
    labels = list(lang_colors.keys()) + ["random-20"]
    ax.legend(handles, labels, loc="lower left", fontsize=8)

    # Annotate anomalies (positive delta on top)
    for i, r in df_plot.iterrows():
        if r["delta_top"] > 0.3:
            ax.annotate(f"↑{r['delta_top']:.2f}",
                        (i - w/2, r["delta_top"]),
                        fontsize=7, ha="center", va="bottom", color="red")

    plt.tight_layout()
    fig.savefig(out_dir / "fig_ablation_bar_chart.png", dpi=110)
    plt.close(fig)
    print(f"  wrote fig_ablation_bar_chart.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr_dir",
                    default="outputs/counterfactual_attribution/attribution")
    ap.add_argument("--abl_dir",
                    default="outputs/counterfactual_attribution/ablation")
    ap.add_argument("--analyses_dir",
                    default="outputs/counterfactual_attribution/analyses")
    ap.add_argument("--bug_audit_dir",
                    default="outputs/counterfactual_attribution/bug_audit")
    args = ap.parse_args()

    cc_out = Path(args.analyses_dir) / "cross_concept"
    io_out = Path(args.analyses_dir) / "input_vs_output"
    ba_out = Path(args.bug_audit_dir)
    for d in [cc_out, io_out, ba_out]:
        d.mkdir(parents=True, exist_ok=True)

    print("cross-concept histograms:")
    plot_cross_concept_histograms(args.attr_dir, cc_out)

    print("tok-strategy v2:")
    plot_tok_strategy_with_ratio(args.attr_dir, ba_out)

    print("ablation bar chart:")
    plot_ablation_bar_chart(args.abl_dir, io_out)


if __name__ == "__main__":
    main()
