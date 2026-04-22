"""
E1 / #3 — Cross-concept feature overlap analysis.

Per language, find SAE features that rank in the top-K of MULTIPLE
(concept, value) cells. A feature appearing in top-K across k cells has
exact Binomial tail probability under i.i.d. top-K sampling over SAE_DIM.

Bonferroni correction over SAE_DIM × (num_cells_in_lang choose 2).

Outputs per language:
  analyses/cross_concept/fig_cross_concept_<lang>.png
  analyses/cross_concept/cross_concept_hits_<lang>.csv
"""
import argparse
import json
from collections import defaultdict
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

SAE_DIM = 32768


def load_cells(attr_dir, lang):
    """Return dict (concept, value) -> top_abs_gxa list-of-dicts, plus n_cells."""
    lang_dir = Path(attr_dir) / lang
    cells = {}
    if not lang_dir.exists():
        return cells
    for cell_dir in sorted(lang_dir.iterdir()):
        if not cell_dir.is_dir():
            continue
        name = cell_dir.name  # e.g. Number_Plur
        if "_" not in name:
            continue
        concept, value = name.split("_", 1)
        summary_path = cell_dir / "summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        cells[(concept, value)] = summary.get("top_50_by_abs_gxa", [])
    return cells


def analyze_lang(attr_dir, lang, topk, out_dir, full_only=False):
    cells = load_cells(attr_dir, lang)
    if len(cells) < 2:
        return None

    # Filter to "full" view vs. single-token (not relevant for cross-concept;
    # already aggregated). Skip for now.

    # Build feature -> list of cells where it appears in top-K
    feat_cells = defaultdict(list)
    for cell, features in cells.items():
        for f in features[:topk]:
            feat_cells[f["feature_idx"]].append(cell)

    n_cells = len(cells)
    # Binomial: prob a specific feature is in top-K of a cell = topk / SAE_DIM
    p = topk / SAE_DIM
    # P(in >= k cells) for n=n_cells Bernoulli trials with success p
    rows = []
    for feat_idx, cells_in in feat_cells.items():
        k_obs = len(cells_in)
        # Survival of binomial: P(X >= k)
        pval = stats.binom.sf(k_obs - 1, n_cells, p)
        # Bonferroni: over SAE_DIM potential features
        pval_bonf = min(1.0, pval * SAE_DIM)
        rows.append({
            "feature_idx": feat_idx,
            "n_cells": k_obs,
            "cells": ";".join([f"{c[0]}/{c[1]}" for c in cells_in]),
            "pval": pval,
            "pval_bonferroni": pval_bonf,
        })

    df = pd.DataFrame(rows).sort_values(
        ["n_cells", "pval_bonferroni"],
        ascending=[False, True],
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"cross_concept_hits_{lang}.csv"
    df.to_csv(csv_path, index=False)

    # Plot: bar chart of top-40 features by n_cells, colored by significance
    hits = df.head(40)
    if len(hits) > 0:
        fig, ax = plt.subplots(figsize=(12, max(4, len(hits) * 0.2)))
        colors = [
            "tab:red" if p < 0.01 else ("tab:orange" if p < 0.05 else "tab:gray")
            for p in hits["pval_bonferroni"]
        ]
        ax.barh(range(len(hits)), hits["n_cells"], color=colors)
        ax.set_yticks(range(len(hits)))
        ax.set_yticklabels([f"f{int(i)}" for i in hits["feature_idx"]], fontsize=8)
        ax.set_xlabel(f"# cells in top-{topk}")
        ax.set_title(f"{lang}: features top-{topk} across multiple concept-cells "
                     f"({n_cells} cells total)\nred=p_bonf<0.01, orange=p_bonf<0.05")
        ax.invert_yaxis()
        ax.set_xlim(1, n_cells + 0.5)
        plt.tight_layout()
        fig.savefig(out_dir / f"fig_cross_concept_{lang}.png", dpi=110)
        plt.close(fig)

    return {
        "lang": lang,
        "n_cells": n_cells,
        "n_features_in_any_topk": len(df),
        "n_features_k_ge_2": int((df["n_cells"] >= 2).sum()),
        "n_features_k_ge_2_sig": int(((df["n_cells"] >= 2) & (df["pval_bonferroni"] < 0.01)).sum()),
        "n_features_k_ge_3": int((df["n_cells"] >= 3).sum()),
        "top_3_features": df.head(3).to_dict(orient="records"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr_dir", default="outputs/overnight_multilingual/attribution")
    ap.add_argument("--output_dir", default="outputs/overnight_multilingual/analyses/cross_concept")
    ap.add_argument("--topk", type=int, default=50)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    summaries = {}
    for lang in ["eng", "fra", "spa", "tur", "ara"]:
        s = analyze_lang(args.attr_dir, lang, args.topk, out_dir)
        if s: summaries[lang] = s

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
