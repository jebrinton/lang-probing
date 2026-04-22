"""
Wave 4 — bug-audit plots.

Produces:
  bug_audit/per_pair_metric_distribution.png
  bug_audit/concentration_of_top_features.png
  bug_audit/tok_strategy_counts.png

Driven by each cell's summary.json (written by attribute_multilingual.py).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr_dir", default="outputs/overnight_multilingual/attribution")
    ap.add_argument("--output_dir", default="outputs/overnight_multilingual/bug_audit")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for lang_dir in sorted(Path(args.attr_dir).iterdir()):
        if not lang_dir.is_dir():
            continue
        for cell_dir in sorted(lang_dir.iterdir()):
            if not cell_dir.is_dir():
                continue
            sp = cell_dir / "summary.json"
            if not sp.exists(): continue
            with open(sp) as f:
                s = json.load(f)
            rows.append({
                "lang": lang_dir.name,
                "cell": cell_dir.name,
                "n_pairs": s.get("n_pairs", 0),
                "n_single": s.get("n_single", 0),
                "n_last": s.get("n_last", 0),
                "metric_mean": s.get("sanity", {}).get("metric_mean", 0),
                "metric_neg_frac": s.get("sanity", {}).get("metric_neg_frac", 0),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "cells_summary.csv", index=False)

    if df.empty:
        print("no cells found")
        return

    # Per-pair metric mean per cell
    fig, ax = plt.subplots(figsize=(12, 4))
    df_sorted = df.sort_values(["lang", "cell"]).reset_index(drop=True)
    bars = ax.bar(range(len(df_sorted)), df_sorted["metric_mean"],
                  color=["tab:red" if m < 0 else "tab:blue" for m in df_sorted["metric_mean"]])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels(
        [f"{r['lang']}/{r['cell']}" for _, r in df_sorted.iterrows()],
        rotation=75, fontsize=7, ha="right"
    )
    ax.set_ylabel("metric_mean  (logP_orig - logP_cf)")
    ax.set_title("Per-cell mean of logP_orig - logP_cf (positive = model prefers orig)")
    plt.tight_layout()
    fig.savefig(out_dir / "per_pair_metric_distribution.png", dpi=110)
    plt.close(fig)

    # Tok strategy composition
    fig, ax = plt.subplots(figsize=(12, 4))
    w = 0.4
    xs = range(len(df_sorted))
    ax.bar(xs, df_sorted["n_single"], width=w, label="single-BPE", color="tab:blue")
    ax.bar([x + w for x in xs], df_sorted["n_last"], width=w, label="multi (last-BPE)", color="tab:orange")
    ax.set_xticks([x + w/2 for x in xs])
    ax.set_xticklabels(
        [f"{r['lang']}/{r['cell']}" for _, r in df_sorted.iterrows()],
        rotation=75, fontsize=7, ha="right"
    )
    ax.set_ylabel("# pairs")
    ax.set_title("Tokenization strategy composition per cell")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "tok_strategy_counts.png", dpi=110)
    plt.close(fig)

    # Negative metric fraction
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(xs, df_sorted["metric_neg_frac"],
           color=["tab:red" if v > 0.3 else "tab:gray" for v in df_sorted["metric_neg_frac"]])
    ax.axhline(0.3, color="r", ls="--", lw=0.7, label="0.3 threshold")
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{r['lang']}/{r['cell']}" for _, r in df_sorted.iterrows()],
        rotation=75, fontsize=7, ha="right"
    )
    ax.set_ylabel("fraction with metric<0")
    ax.set_title("Per-cell fraction of pairs where model prefers counterfactual (pathology if > 0.3)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "metric_negative_fraction.png", dpi=110)
    plt.close(fig)

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
