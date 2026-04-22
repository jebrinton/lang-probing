"""
E4 / #6 — Input activation vs output ablation effect.

For each cell's top-K features (by |signed grad*act|):
  x: Δact_input  = mean activation on (correct sentences) - mean on (wrong sentences),
                   computed from per-pair grad/act cached in aggregation
                   (approximation: use the signed mean activation directly)
  y: Δablation   = mean_delta_orig from ablate_validate output for this cell

Scatter per cell. Annotate any outliers (high-x low-y, low-x high-y).

Also: cross-lingual matrix. For source cell `(lang_src, C, V)`, take top-10
features, load held-out pairs from the *same* (C, V) in another language,
compute ablation effect of the same features there. Plot matrix.

This script reuses ablate_validate.py outputs; no GPU forward passes needed
beyond what ablate_validate already computed, EXCEPT for the cross-lingual
step which DOES need GPU (submits as qsub).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr_dir", default="outputs/overnight_multilingual/attribution")
    ap.add_argument("--abl_dir", default="outputs/overnight_multilingual/ablation")
    ap.add_argument("--output_dir", default="outputs/overnight_multilingual/analyses/input_vs_output")
    ap.add_argument("--top_k", type=int, default=20)
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for lang_dir in sorted(Path(args.attr_dir).iterdir()):
        if not lang_dir.is_dir(): continue
        lang = lang_dir.name
        for cell_dir in sorted(lang_dir.iterdir()):
            if not cell_dir.is_dir() or "_" not in cell_dir.name: continue
            concept, value = cell_dir.name.split("_", 1)

            # Load attribution signed tensors
            signed_grad = cell_dir / "aggregated_signed.pt"
            signed_gxa = cell_dir / "aggregated_signed_gxa.pt"
            if not (signed_grad.exists() and signed_gxa.exists()):
                continue
            sg = torch.load(signed_grad, map_location="cpu").numpy()
            sgxa = torch.load(signed_gxa, map_location="cpu").numpy()

            # Find ablation result
            abl_path = Path(args.abl_dir) / lang / cell_dir.name / "holdout_ablation.json"
            if not abl_path.exists(): continue
            with open(abl_path) as f:
                abl = json.load(f)

            top_features = abl.get("top_features", [])
            if not top_features: continue

            per_pair_top = abl["top_result"].get("per_pair", [])
            if not per_pair_top: continue

            # Per-feature: for each feature, we don't have individual ablation
            # (ablate_validate bundles top-K together). So we plot one point per
            # (cell, feature) using cell-level Δorig for Y, and feature's signed
            # grad*act for X. Less granular than ideal.

            x = np.abs(sgxa[top_features])   # |attribution strength|
            y_cell = abl["top_result"]["mean_delta_orig"]  # scalar
            y_rand = abl["random_result"]["mean_delta_orig"]

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(x, [y_cell] * len(x), s=40, c="tab:blue", label="top-K (bundled ablation)")
            ax.axhline(y_rand, color="tab:gray", ls="--", label=f"random-K ablation ({y_rand:.3f})")
            for f, xf in zip(top_features, x):
                ax.annotate(f"f{f}", (xf, y_cell), fontsize=6, alpha=0.7)
            ax.set_xlabel("|signed grad×act|  (input-side attribution strength)")
            ax.set_ylabel("mean Δ logP(orig)  (output ablation)")
            ax.set_title(f"Input attribution vs. output ablation: {lang}/{concept}={value}\n"
                         f"n_holdout={abl['top_result']['n']}")
            ax.legend(fontsize=8)
            plt.tight_layout()
            fig.savefig(out_dir / f"fig_input_vs_output_{lang}_{concept}_{value}.png", dpi=110)
            plt.close(fig)

            summary.append({
                "lang": lang, "concept": concept, "value": value,
                "n_holdout": abl["top_result"]["n"],
                "mean_delta_orig_top": y_cell,
                "mean_delta_orig_random": y_rand,
                "effect_ratio_over_random": (y_cell / y_rand) if y_rand else None,
                "top_features": top_features[: args.top_k],
            })

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary[:3], indent=2))


if __name__ == "__main__":
    main()
