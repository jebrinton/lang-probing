"""
E3 / #5 — Sign-flip analysis across languages.

Load per-cell `aggregated_signed_gxa.pt` for pairs of (lang, concept, value)
cells. Scatter feature-wise x=lang1 signed score vs y=lang2 signed score,
color by sign agreement. Annotate top-10 highest-magnitude opposite-sign
features.

Outputs:
  analyses/sign_flip/fig_sign_flip_<L1>_<L2>_<C>_<V>.png
  analyses/sign_flip/sign_flip_top10_<L1>_<L2>_<C>_<V>.csv
  analyses/sign_flip/summary.json
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def load_signed_gxa(attr_dir, lang, concept, value):
    path = Path(attr_dir) / lang / f"{concept}_{value}" / "aggregated_signed_gxa.pt"
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


def scatter_pair(t1, t2, l1, l2, concept, value, out_dir):
    """Scatter signed grad*act between two cells. Return top-10 sign-flip rows."""
    t1n = t1.numpy()
    t2n = t2.numpy()

    # Filter to features in top-200 of either by absolute magnitude
    mask = np.zeros_like(t1n, dtype=bool)
    for arr in [t1n, t2n]:
        idx = np.argsort(-np.abs(arr))[:200]
        mask[idx] = True
    feat_idx = np.where(mask)[0]
    x = t1n[feat_idx]
    y = t2n[feat_idx]

    # Sign agreement
    sign_same = np.sign(x) * np.sign(y) > 0
    sign_opp = np.sign(x) * np.sign(y) < 0

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x[sign_same], y[sign_same], c="tab:green", s=10, alpha=0.6,
               label=f"same sign ({sign_same.sum()})")
    ax.scatter(x[sign_opp], y[sign_opp], c="tab:red", s=18, alpha=0.8,
               label=f"opposite ({sign_opp.sum()})")
    ax.scatter(x[~(sign_same | sign_opp)], y[~(sign_same | sign_opp)],
               c="lightgrey", s=5, alpha=0.4, label="near-zero")

    # Annotate top-10 opposite-sign by |x|+|y|
    opp_rows = np.where(sign_opp)[0]
    if len(opp_rows):
        mag = np.abs(x[opp_rows]) + np.abs(y[opp_rows])
        top = opp_rows[np.argsort(-mag)[:10]]
        top_rows = []
        for i in top:
            fi = int(feat_idx[i])
            ax.annotate(f"f{fi}", (x[i], y[i]), fontsize=7, alpha=0.8)
            top_rows.append({
                "feature_idx": fi,
                f"{l1}_signed_gxa": float(x[i]),
                f"{l2}_signed_gxa": float(y[i]),
                "magnitude_sum": float(mag[np.argsort(-mag)][list(top).index(i)]) if i in top else float(np.abs(x[i]) + np.abs(y[i])),
            })
    else:
        top_rows = []

    lim = max(np.abs(x).max(), np.abs(y).max()) * 1.1 if len(x) else 1.0
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel(f"{l1} {concept}={value}  signed grad×act")
    ax.set_ylabel(f"{l2} {concept}={value}  signed grad×act")
    ax.set_title(f"Sign-flip: {l1} vs {l2}  —  {concept}={value}")
    ax.legend(fontsize=8)
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / f"fig_sign_flip_{l1}_{l2}_{concept}_{value}.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)

    if top_rows:
        pd.DataFrame(top_rows).to_csv(
            out_dir / f"sign_flip_top10_{l1}_{l2}_{concept}_{value}.csv", index=False
        )

    return {
        "pair": f"{l1} vs {l2}",
        "cell": f"{concept}={value}",
        "n_same_sign": int(sign_same.sum()),
        "n_opp_sign": int(sign_opp.sum()),
        "top_10_opposite": top_rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr_dir", default="outputs/counterfactual_attribution/attribution")
    ap.add_argument("--output_dir", default="outputs/counterfactual_attribution/analyses/sign_flip")
    args = ap.parse_args()

    attr = args.attr_dir
    out_dir = Path(args.output_dir)

    # Language pairs × cells of interest
    comparisons = [
        ("fra", "spa", "Gender", "Fem"),
        ("fra", "spa", "Gender", "Masc"),
        ("fra", "ara", "Gender", "Fem"),
        ("fra", "ara", "Gender", "Masc"),
        ("spa", "ara", "Gender", "Fem"),
        ("fra", "spa", "Number", "Plur"),
        ("fra", "spa", "Number", "Sing"),
        ("fra", "ara", "Number", "Sing"),
        ("ara", "tur", "Number", "Sing"),
    ]

    summary = []
    for l1, l2, concept, value in comparisons:
        t1 = load_signed_gxa(attr, l1, concept, value)
        t2 = load_signed_gxa(attr, l2, concept, value)
        if t1 is None or t2 is None:
            print(f"skip {l1} {l2} {concept}={value}: missing data")
            continue
        s = scatter_pair(t1, t2, l1, l2, concept, value, out_dir)
        summary.append(s)
        print(f"{l1} vs {l2} {concept}={value}: same={s['n_same_sign']} opp={s['n_opp_sign']}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
