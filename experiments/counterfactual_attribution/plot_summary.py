"""
Plot the per-cell ablation summary at
  outputs/counterfactual_attribution/analyses/input_vs_output/summary.json

Three views:
  1. fig_summary_heatmap.png   — language x (concept/value) heatmap of
                                  mean_delta_orig_top, annotated with n.
  2. fig_summary_scatter.png   — random-20 delta (x) vs top-20 delta (y),
                                  one point per cell, colored by language,
                                  sized by n_holdout. y=x diagonal for ref.
  3. fig_summary_jaccard.png   — Jaccard overlap of top-20 feature sets
                                  across all cells, ordered by language.

Usage:
    python experiments/counterfactual_attribution/plot_summary.py \
        [--summary <path>] [--out-dir <dir>]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LANG_COLORS = {
    "eng": "tab:gray",
    "fra": "tab:blue",
    "spa": "tab:orange",
    "tur": "tab:green",
    "ara": "tab:red",
}


def load_summary(path):
    with open(path) as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)
    df["cell"] = df["concept"] + "/" + df["value"].astype(str)
    return df


def plot_heatmap(df, out_path):
    """Lang x cell heatmap of mean_delta_orig_top."""
    pivot = df.pivot(index="lang", columns="cell", values="mean_delta_orig_top")
    ncounts = df.pivot(index="lang", columns="cell", values="n_holdout")

    lang_order = [l for l in ["eng", "fra", "spa", "tur", "ara"] if l in pivot.index]
    pivot = pivot.reindex(lang_order)
    ncounts = ncounts.reindex(lang_order)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    ncounts = ncounts.reindex(pivot.columns, axis=1)

    vmax = np.nanmax(np.abs(pivot.values))
    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(pivot.columns)),
                                    0.6 * len(pivot.index) + 2))
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=60, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=11)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            n = ncounts.values[i, j]
            if np.isnan(v):
                continue
            txt = f"{v:.2f}\nn={int(n)}"
            color = "white" if abs(v) > vmax * 0.5 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(r"mean $\Delta$ logP(orig) after top-20 ablation")
    ax.set_title(
        "Per-cell ablation effect (negative = ablation hurts correct prediction)"
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def plot_scatter(df, out_path):
    """Random-20 delta (x) vs top-20 delta (y)."""
    fig, ax = plt.subplots(figsize=(8, 7))

    for lang, sub in df.groupby("lang"):
        ax.scatter(
            sub["mean_delta_orig_random"],
            sub["mean_delta_orig_top"],
            s=np.clip(sub["n_holdout"], 2, 30) * 12,
            c=LANG_COLORS.get(lang, "black"),
            alpha=0.75,
            edgecolor="black",
            linewidth=0.5,
            label=lang,
        )

    # y = x diagonal
    lims_all = np.concatenate(
        [df["mean_delta_orig_random"].values, df["mean_delta_orig_top"].values]
    )
    lo, hi = np.nanmin(lims_all), np.nanmax(lims_all)
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            "k--", lw=0.8, label="y = x (no effect)")
    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(0, color="k", lw=0.4)

    # Label the most-negative top-20 cells
    df_sorted = df.sort_values("mean_delta_orig_top").head(5)
    for _, r in df_sorted.iterrows():
        ax.annotate(
            f"{r['lang']}/{r['concept']}/{r['value']}",
            (r["mean_delta_orig_random"], r["mean_delta_orig_top"]),
            xytext=(6, -2), textcoords="offset points",
            fontsize=7, color="black",
        )

    ax.set_xlabel(r"mean $\Delta$ logP(orig)  —  random-20 baseline")
    ax.set_ylabel(r"mean $\Delta$ logP(orig)  —  top-20 attributed features")
    ax.set_title("Top-20 ablation vs random-20 baseline, per cell\n"
                 "Below the diagonal = top-20 hurts more than random (as expected)")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def plot_jaccard(df, out_path):
    """Jaccard overlap of top-20 feature sets across cells, lang-grouped."""
    df = df.sort_values(["lang", "concept", "value"]).reset_index(drop=True)
    feats = [set(r) for r in df["top_features"]]
    n = len(feats)

    jac = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            a, b = feats[i], feats[j]
            u = len(a | b)
            jac[i, j] = (len(a & b) / u) if u else 0.0

    labels = [f"{r['lang']}/{r['concept']}/{r['value']}"
              for _, r in df.iterrows()]

    fig, ax = plt.subplots(figsize=(0.32 * n + 3, 0.32 * n + 3))
    im = ax.imshow(jac, cmap="viridis", vmin=0, vmax=1, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=7)

    # Language-block separators
    lang_vec = df["lang"].values
    boundaries = [i for i in range(1, n) if lang_vec[i] != lang_vec[i - 1]]
    for b in boundaries:
        ax.axhline(b - 0.5, color="white", lw=1.2)
        ax.axvline(b - 0.5, color="white", lw=1.2)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Jaccard overlap (top-20 features)")
    ax.set_title(
        "Top-20 feature overlap across cells\n"
        "White lines separate languages — off-diagonal blocks = cross-lingual sharing"
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def main():
    ap = argparse.ArgumentParser()
    default_summary = (
        "outputs/counterfactual_attribution/analyses/input_vs_output/summary.json"
    )
    ap.add_argument("--summary", default=default_summary)
    ap.add_argument("--out-dir", default=None,
                    help="Defaults to the dir containing --summary.")
    args = ap.parse_args()

    summary_path = Path(args.summary)
    out_dir = Path(args.out_dir) if args.out_dir else summary_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_summary(summary_path)
    print(f"loaded {len(df)} cells from {summary_path}")

    plot_heatmap(df, out_dir / "fig_summary_heatmap.png")
    plot_scatter(df, out_dir / "fig_summary_scatter.png")
    plot_jaccard(df, out_dir / "fig_summary_jaccard.png")


if __name__ == "__main__":
    main()
