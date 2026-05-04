"""
Bootstrap rank stability + alternative per-pair aggregations for GCM
translation results. Pure post-hoc; reads existing per-pair tensors,
no GPU needed.

For each direction with N=100 successful pairs:
  - B resamples with replacement of size N
  - Per resample: mean|IE| ranking, take top-K
  - Per (head | feature): bootstrap_top_k_freq = fraction of resamples
    in which it lands in top-K

Also (on full N, no resampling):
  - median |IE| per (head | feature)
  - P(|IE| > tau) for tau in {0.1, 0.2, 0.5}
  - std |IE|

Outputs:
  outputs/gcm_translation/<direction>/bootstrap_stability.json
  outputs/gcm_translation/_aggregate/bootstrap_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Per-direction bootstrap
# ---------------------------------------------------------------------------


def _bootstrap_topk_freq(
    abs_ie: np.ndarray,        # [N, M]   (M = n_layers*n_heads or SAE_DIM)
    K: int,
    B: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Returns array of shape [M], where entry m = fraction of B bootstrap
    resamples in which component m is in the top-K by mean(|IE|).
    """
    N, M = abs_ie.shape
    counts = np.zeros(M, dtype=np.int64)
    for _ in range(B):
        idx = rng.integers(0, N, size=N)
        means = abs_ie[idx].mean(axis=0)            # [M]
        # argpartition is O(M); we only need *which* M entries are top-K
        topk_mask_idx = np.argpartition(-means, K - 1)[:K]
        counts[topk_mask_idx] += 1
    return counts / B


def _alt_aggregations(
    signed_ie: np.ndarray,     # [N, M]
    taus: tuple[float, ...] = (0.1, 0.2, 0.5),
):
    """Returns dict of arrays each of shape [M]."""
    abs_ie = np.abs(signed_ie)
    return {
        "mean_abs_ie": abs_ie.mean(axis=0),
        "median_abs_ie": np.median(abs_ie, axis=0),
        "std_abs_ie": abs_ie.std(axis=0),
        "mean_signed_ie": signed_ie.mean(axis=0),
        **{
            f"p_above_{tau}": (abs_ie > tau).mean(axis=0)
            for tau in taus
        },
    }


def process_direction(
    direction_dir: Path,
    B: int,
    K_heads: int,
    K_sae: int,
    seed: int,
) -> Optional[dict]:
    """Compute bootstrap freqs + alt aggregations for one direction."""
    summary_path = direction_dir / "summary.json"
    heads_path = direction_dir / "heads_ie.pt"
    sae_path = direction_dir / "sae_ie.pt"

    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        summary = json.load(f)

    rng = np.random.default_rng(seed)
    out = {
        "direction": direction_dir.name,
        "n_pairs_attempted": summary.get("n_pairs_attempted"),
        "n_pairs_successful_heads": summary.get("n_pairs_successful_heads"),
        "n_pairs_successful_sae": summary.get("n_pairs_successful_sae"),
        "B": B,
        "seed": seed,
    }

    # --- Heads ---
    if heads_path.exists() and (summary.get("n_pairs_successful_heads") or 0) >= 95:
        heads = torch.load(heads_path, map_location="cpu", weights_only=False).float().numpy()
        # Drop any rows containing NaN (failed pairs filled with sentinel)
        keep = ~np.isnan(heads).any(axis=(1, 2))
        heads = heads[keep]
        N, L, H = heads.shape
        flat = heads.reshape(N, L * H)                         # [N, L*H]
        freq = _bootstrap_topk_freq(np.abs(flat), K_heads, B, rng)   # [L*H]
        agg = _alt_aggregations(flat)
        # Top-K by full-N mean|IE| for cross-reference
        topk_idx = np.argpartition(-agg["mean_abs_ie"], K_heads - 1)[:K_heads]
        topk_idx = topk_idx[np.argsort(-agg["mean_abs_ie"][topk_idx])]
        out["heads"] = {
            "n_used": int(N),
            "n_layers": int(L),
            "n_heads_per_layer": int(H),
            "K": K_heads,
            "top_k_by_mean_abs_ie": [
                {
                    "layer": int(i // H),
                    "head": int(i % H),
                    "mean_abs_ie": float(agg["mean_abs_ie"][i]),
                    "median_abs_ie": float(agg["median_abs_ie"][i]),
                    "std_abs_ie": float(agg["std_abs_ie"][i]),
                    "mean_signed_ie": float(agg["mean_signed_ie"][i]),
                    "p_above_0.1": float(agg["p_above_0.1"][i]),
                    "p_above_0.2": float(agg["p_above_0.2"][i]),
                    "p_above_0.5": float(agg["p_above_0.5"][i]),
                    "bootstrap_top_k_freq": float(freq[i]),
                }
                for i in topk_idx
            ],
        }

    # --- SAE ---
    if sae_path.exists() and (summary.get("n_pairs_successful_sae") or 0) >= 95:
        sae = torch.load(sae_path, map_location="cpu", weights_only=False).float().numpy()
        keep = ~np.isnan(sae).any(axis=1)
        sae = sae[keep]
        N, D = sae.shape
        freq = _bootstrap_topk_freq(np.abs(sae), K_sae, B, rng)
        agg = _alt_aggregations(sae)
        topk_idx = np.argpartition(-agg["mean_abs_ie"], K_sae - 1)[:K_sae]
        topk_idx = topk_idx[np.argsort(-agg["mean_abs_ie"][topk_idx])]
        out["sae"] = {
            "n_used": int(N),
            "sae_dim": int(D),
            "K": K_sae,
            "top_k_by_mean_abs_ie": [
                {
                    "feature_idx": int(i),
                    "mean_abs_ie": float(agg["mean_abs_ie"][i]),
                    "median_abs_ie": float(agg["median_abs_ie"][i]),
                    "std_abs_ie": float(agg["std_abs_ie"][i]),
                    "mean_signed_ie": float(agg["mean_signed_ie"][i]),
                    "p_above_0.1": float(agg["p_above_0.1"][i]),
                    "p_above_0.2": float(agg["p_above_0.2"][i]),
                    "p_above_0.5": float(agg["p_above_0.5"][i]),
                    "bootstrap_top_k_freq": float(freq[i]),
                }
                for i in topk_idx
            ],
        }

    return out


# ---------------------------------------------------------------------------
# Cross-direction aggregation
# ---------------------------------------------------------------------------


def aggregate(per_direction: list[dict], out_path: Path) -> None:
    """Aggregate the per-direction bootstrap_top_k_freq into per-feature
    summaries: how often does each feature/head appear in top-K across the
    sweep, and what's its bootstrap stability per direction.
    """
    head_summary: dict[tuple[int, int], dict] = {}
    sae_summary: dict[int, dict] = {}

    for d in per_direction:
        name = d["direction"]
        if "heads" in d:
            for h in d["heads"]["top_k_by_mean_abs_ie"]:
                key = (h["layer"], h["head"])
                rec = head_summary.setdefault(
                    key,
                    {
                        "layer": key[0],
                        "head": key[1],
                        "n_directions_in_topk": 0,
                        "directions": [],
                    },
                )
                rec["n_directions_in_topk"] += 1
                rec["directions"].append({
                    "direction": name,
                    "rank_in_direction": d["heads"]["top_k_by_mean_abs_ie"].index(h) + 1,
                    "mean_abs_ie": h["mean_abs_ie"],
                    "median_abs_ie": h["median_abs_ie"],
                    "bootstrap_top_k_freq": h["bootstrap_top_k_freq"],
                })
        if "sae" in d:
            for s in d["sae"]["top_k_by_mean_abs_ie"]:
                key = s["feature_idx"]
                rec = sae_summary.setdefault(
                    key,
                    {
                        "feature_idx": key,
                        "n_directions_in_topk": 0,
                        "directions": [],
                    },
                )
                rec["n_directions_in_topk"] += 1
                rec["directions"].append({
                    "direction": name,
                    "rank_in_direction": d["sae"]["top_k_by_mean_abs_ie"].index(s) + 1,
                    "mean_abs_ie": s["mean_abs_ie"],
                    "median_abs_ie": s["median_abs_ie"],
                    "bootstrap_top_k_freq": s["bootstrap_top_k_freq"],
                })

    # Sort by universality (n_directions, then mean bootstrap freq across them)
    def _stable_key(rec):
        return (
            -rec["n_directions_in_topk"],
            -np.mean([x["bootstrap_top_k_freq"] for x in rec["directions"]]),
        )

    heads_sorted = sorted(head_summary.values(), key=_stable_key)
    sae_sorted = sorted(sae_summary.values(), key=_stable_key)

    summary = {
        "n_directions": len(per_direction),
        "universal_heads": heads_sorted[:100],
        "universal_sae_features": sae_sorted[:200],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_dir", default="outputs/gcm_translation")
    p.add_argument("--B", type=int, default=1000)
    p.add_argument("--K_heads", type=int, default=20)
    p.add_argument("--K_sae", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    sweep = Path(args.sweep_dir)
    direction_dirs = sorted(
        d for d in sweep.iterdir()
        if d.is_dir() and "__" in d.name and d.name != "_aggregate"
    )
    print(f"Found {len(direction_dirs)} directions under {sweep}")

    per_direction = []
    for d in direction_dirs:
        out = process_direction(d, B=args.B, K_heads=args.K_heads, K_sae=args.K_sae, seed=args.seed)
        if out is None:
            print(f"  skip {d.name}: no summary.json")
            continue
        path = d / "bootstrap_stability.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        per_direction.append(out)
        h_med_freq = (
            np.median([h["bootstrap_top_k_freq"] for h in out["heads"]["top_k_by_mean_abs_ie"]])
            if "heads" in out else float("nan")
        )
        s_med_freq = (
            np.median([s["bootstrap_top_k_freq"] for s in out["sae"]["top_k_by_mean_abs_ie"]])
            if "sae" in out else float("nan")
        )
        print(f"  {d.name}: heads top-{args.K_heads} median freq={h_med_freq:.3f}, sae top-{args.K_sae} median freq={s_med_freq:.3f}")

    aggregate(per_direction, sweep / "_aggregate" / "bootstrap_summary.json")
    print(f"Wrote {sweep / '_aggregate' / 'bootstrap_summary.json'}")


if __name__ == "__main__":
    main()
