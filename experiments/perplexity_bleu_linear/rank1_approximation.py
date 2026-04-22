"""Rank-1 SVD approximation of the BLEU matrix.

The paper claims "rank-1 approximation is 88% faithful" for Llama. This
script reconstructs that analysis:

1. Load the joined BLEU/PPL CSV.
2. Pivot BLEU into a (src, tgt) matrix.
3. SVD; compute the rank-k approximation's faithfulness = 1 - ||M - M_k||_F / ||M||_F.
4. Plot faithfulness vs rank (fig `linear_effects_ranks.png`).
5. Scatter rank-1 predicted vs actual BLEU (fig `linear_effects_{model}.png`).

Faithfulness here is the Frobenius-norm-ratio metric, equivalently the
fraction of matrix energy captured by the rank-k components.

Run:
    python experiments/perplexity_bleu_linear/rank1_approximation.py \\
        --model llama \\
        --output_dir outputs/perplexity_bleu_linear/bleu_and_ppl/rank1 \\
        --img_dir img/perplexity_bleu_linear

Data source: outputs/perplexity_bleu_linear/bleu_and_ppl/combined_results_{model}.csv
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lang_probing_src.config import OUTPUTS_DIR, IMG_DIR


logger = logging.getLogger(__name__)


def load_bleu_matrix(model: str, data_dir: Path):
    """Load combined_results CSV and pivot to (src, tgt) BLEU matrix.

    Returns (matrix, src_labels, tgt_labels).
    """
    path = data_dir / f"combined_results_{model}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")
    df = pd.read_csv(path)
    pivot = df.pivot(index="src", columns="tgt", values="bleu")
    return pivot.values, list(pivot.index), list(pivot.columns)


def rank_k_approximation(M: np.ndarray, k: int) -> np.ndarray:
    """Best rank-k approximation via truncated SVD."""
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]


def faithfulness_at_rank(M: np.ndarray, k: int) -> float:
    """1 - ||M - M_k||_F / ||M||_F, in [0, 1]."""
    M_k = rank_k_approximation(M, k)
    num = np.linalg.norm(M - M_k, ord="fro")
    den = np.linalg.norm(M, ord="fro")
    return float(1.0 - num / den) if den > 0 else float("nan")


def plot_faithfulness_vs_rank(M: np.ndarray, model: str, save_path: Path) -> np.ndarray:
    """Plot the error-vs-rank curve and return the faithfulness vector."""
    ranks = np.arange(1, min(M.shape) + 1)
    faith = np.array([faithfulness_at_rank(M, int(k)) for k in ranks])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ranks, 1.0 - faith, "o-", label="relative error $1 - f_k$")
    ax.axhline(1.0 - faith[0], color="tab:red", linestyle="--", alpha=0.5,
               label=f"rank-1 residual = {1.0 - faith[0]:.3f}")
    ax.set_xlabel("Rank $k$ of SVD approximation")
    ax.set_ylabel("Relative Frobenius error")
    ax.set_title(f"BLEU matrix: relative error vs rank ({model})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", save_path)
    return faith


def plot_rank1_vs_actual(M: np.ndarray, model: str, faith1: float,
                         save_path: Path) -> None:
    """Scatter: rank-1 predicted vs actual BLEU."""
    M1 = rank_k_approximation(M, 1)
    mask = ~np.isnan(M) & ~np.isnan(M1)
    actual = M[mask]
    predicted = M1[mask]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(actual, predicted, alpha=0.5, s=20)
    lim = (float(np.nanmin([actual.min(), predicted.min()])) - 1,
           float(np.nanmax([actual.max(), predicted.max()])) + 1)
    ax.plot(lim, lim, color="tab:red", linestyle="--", alpha=0.7, label="y = x")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Actual BLEU")
    ax.set_ylabel("Rank-1 predicted BLEU")
    ax.set_title(f"Rank-1 SVD: faithfulness = {faith1 * 100:.1f}% ({model})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", save_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["llama", "aya"], default="llama")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path(OUTPUTS_DIR) / "perplexity_bleu_linear" / "bleu_and_ppl",
        help="Dir containing combined_results_{model}.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(OUTPUTS_DIR) / "perplexity_bleu_linear" / "bleu_and_ppl" / "rank1",
    )
    parser.add_argument(
        "--img_dir",
        type=Path,
        default=Path(IMG_DIR) / "perplexity_bleu_linear",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.img_dir.mkdir(parents=True, exist_ok=True)

    M, src_labels, tgt_labels = load_bleu_matrix(args.model, args.data_dir)
    logger.info("BLEU matrix: %s (nan count=%d)",
                M.shape, int(np.isnan(M).sum()))

    # SVD requires a complete matrix — impute NaN with column (tgt) mean
    if np.isnan(M).any():
        col_means = np.nanmean(M, axis=0)
        inds = np.where(np.isnan(M))
        M_filled = M.copy()
        M_filled[inds] = np.take(col_means, inds[1])
        logger.info("Imputed %d NaN cells with column means for SVD",
                    len(inds[0]))
    else:
        M_filled = M

    # Save singular values
    U, S, Vt = np.linalg.svd(M_filled, full_matrices=False)
    pd.DataFrame({"rank": np.arange(1, len(S) + 1), "singular_value": S}).to_csv(
        args.output_dir / f"singular_values_{args.model}.csv", index=False
    )

    # Faithfulness curve + plot
    faith = plot_faithfulness_vs_rank(
        M_filled, args.model, args.img_dir / f"linear_effects_ranks_{args.model}.png"
    )

    # Rank-1 predicted vs actual scatter
    plot_rank1_vs_actual(
        M_filled, args.model, float(faith[0]),
        args.img_dir / f"linear_effects_{args.model}.png",
    )

    # CSV summary
    summary = pd.DataFrame({
        "rank": np.arange(1, len(faith) + 1),
        "faithfulness": faith,
        "relative_error": 1.0 - faith,
    })
    summary.to_csv(args.output_dir / f"faithfulness_{args.model}.csv", index=False)

    print(f"\n=== Rank-1 faithfulness for {args.model}: {faith[0] * 100:.2f}% ===")
    print(f"    (1 - ||M - M_1||_F / ||M||_F)")
    print(f"Top-5 singular values: {S[:5].round(3)}")
    print(f"\nSaved:")
    print(f"  img:  {args.img_dir / f'linear_effects_ranks_{args.model}.png'}")
    print(f"  img:  {args.img_dir / f'linear_effects_{args.model}.png'}")
    print(f"  csv:  {args.output_dir / f'faithfulness_{args.model}.csv'}")
    print(f"  csv:  {args.output_dir / f'singular_values_{args.model}.csv'}")


if __name__ == "__main__":
    main()
