"""
Three-way decomposition of GCM IE per (head | SAE feature):

    REAL  (Phase 1, cross-lang, gold response anchored)
                       = content + cross-lang routing + translation-circuit
    NULL_CROSS  (Phase 2, cross-lang, neither response is the gold)
                       = content + cross-lang routing
    NULL_SAME  (Phase 2, src == tgt, neither response is the gold)
                       = content alone

Per (component, direction):
    translation_circuit = mean|IE|_real - mean|IE|_null_cross
    cross_lang_routing  = mean|IE|_null_cross - mean|IE|_null_same
    content_floor       = mean|IE|_null_same   (taken from src-language same-lang run)

Outputs:
  outputs/gcm_translation_null/_aggregate/three_way_decomposition.json
  outputs/gcm_translation_null/_aggregate/three_way_summary.json
  experiments/gcm_translation/img/three_way_decomposition_<heads|sae>.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_mean_abs_ie(direction_dir: Path):
    """Returns (heads_mean_abs_ie [L*H], sae_mean_abs_ie [SAE_DIM]) or
    (None, None) if data missing."""
    heads = sae = None
    h_path = direction_dir / "heads_ie.pt"
    s_path = direction_dir / "sae_ie.pt"
    if h_path.exists():
        t = torch.load(h_path, map_location="cpu", weights_only=False).float()
        # NaN-aware reduction; drop pairs that are entirely NaN
        keep = ~torch.isnan(t).any(dim=(1, 2))
        if keep.sum() > 0:
            t = t[keep]
            heads = t.abs().mean(dim=0).flatten().numpy()  # [L*H]
    if s_path.exists():
        t = torch.load(s_path, map_location="cpu", weights_only=False).float()
        keep = ~torch.isnan(t).any(dim=1)
        if keep.sum() > 0:
            t = t[keep]
            sae = t.abs().mean(dim=0).numpy()              # [SAE_DIM]
    return heads, sae


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------


def decompose(real_dir: Path, null_dir: Path, out_dir: Path, n_heads: int):
    """For each (src, tgt) cross-language direction in `real_dir`, find:
       - the matching null_cross direction (same src,tgt) under null_dir
       - the matching null_same direction (src,src) under null_dir
    Compute per-component (real - null_cross) and (null_cross - null_same).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    real_dirs = sorted(d for d in real_dir.iterdir() if d.is_dir() and "__" in d.name and d.name != "_aggregate")
    null_dirs = {d.name: d for d in null_dir.iterdir() if d.is_dir() and "__" in d.name and d.name != "_aggregate"}

    results = []
    for d_real in real_dirs:
        name = d_real.name
        src, tgt = name.split("__")
        if src == tgt:
            continue  # Phase 1 didn't run identity directions
        null_cross_name = name
        null_same_name = f"{src}__{src}"
        if null_cross_name not in null_dirs or null_same_name not in null_dirs:
            print(f"  skip {name}: missing null_cross={null_cross_name in null_dirs} null_same={null_same_name in null_dirs}")
            continue

        h_real, s_real = load_mean_abs_ie(d_real)
        h_nc, s_nc = load_mean_abs_ie(null_dirs[null_cross_name])
        h_ns, s_ns = load_mean_abs_ie(null_dirs[null_same_name])
        if h_real is None or s_real is None or h_nc is None or s_nc is None or h_ns is None or s_ns is None:
            print(f"  skip {name}: incomplete tensors")
            continue

        rec = {"direction": name, "src": src, "tgt": tgt}

        # --- Heads ---
        # Top-30 by real mean|IE|, with their null_cross / null_same values
        L_times_H = h_real.shape[0]
        top_idx = np.argsort(-h_real)[:30]
        rec["heads_top30"] = [
            {
                "layer": int(i // n_heads),
                "head": int(i % n_heads),
                "real_mean_abs_ie": float(h_real[i]),
                "null_cross_mean_abs_ie": float(h_nc[i]),
                "null_same_mean_abs_ie": float(h_ns[i]),
                "translation_circuit": float(h_real[i] - h_nc[i]),
                "cross_lang_routing": float(h_nc[i] - h_ns[i]),
            }
            for i in top_idx
        ]

        # --- SAE features ---
        top_idx = np.argsort(-s_real)[:50]
        rec["sae_top50"] = [
            {
                "feature_idx": int(i),
                "real_mean_abs_ie": float(s_real[i]),
                "null_cross_mean_abs_ie": float(s_nc[i]),
                "null_same_mean_abs_ie": float(s_ns[i]),
                "translation_circuit": float(s_real[i] - s_nc[i]),
                "cross_lang_routing": float(s_nc[i] - s_ns[i]),
            }
            for i in top_idx
        ]
        results.append(rec)

    with open(out_dir / "three_way_decomposition.json", "w") as f:
        json.dump(results, f, indent=2)

    # Cross-direction summary: for each universal head/feature (top-K of phase1
    # analyze.py output), report mean across directions.
    head_summary = {}     # (L, H) -> dict of lists
    sae_summary = {}      # feat_idx -> dict of lists
    for rec in results:
        for h in rec["heads_top30"]:
            key = (h["layer"], h["head"])
            d = head_summary.setdefault(key, {"layer": key[0], "head": key[1],
                                               "real": [], "null_cross": [], "null_same": []})
            d["real"].append(h["real_mean_abs_ie"])
            d["null_cross"].append(h["null_cross_mean_abs_ie"])
            d["null_same"].append(h["null_same_mean_abs_ie"])
        for s in rec["sae_top50"]:
            key = s["feature_idx"]
            d = sae_summary.setdefault(key, {"feature_idx": key,
                                              "real": [], "null_cross": [], "null_same": []})
            d["real"].append(s["real_mean_abs_ie"])
            d["null_cross"].append(s["null_cross_mean_abs_ie"])
            d["null_same"].append(s["null_same_mean_abs_ie"])

    def _fold(d):
        d["n_directions"] = len(d["real"])
        for k in ("real", "null_cross", "null_same"):
            d[f"{k}_mean"] = float(np.mean(d[k]))
            d[f"{k}_std"] = float(np.std(d[k]))
            del d[k]
        d["translation_circuit_mean"] = d["real_mean"] - d["null_cross_mean"]
        d["cross_lang_routing_mean"] = d["null_cross_mean"] - d["null_same_mean"]
        return d

    head_list = sorted([_fold(v) for v in head_summary.values()],
                       key=lambda x: -x["translation_circuit_mean"])
    sae_list = sorted([_fold(v) for v in sae_summary.values()],
                      key=lambda x: -x["translation_circuit_mean"])

    with open(out_dir / "three_way_summary.json", "w") as f:
        json.dump({
            "n_directions_decomposed": len(results),
            "heads_ranked_by_translation_circuit": head_list[:50],
            "sae_features_ranked_by_translation_circuit": sae_list[:100],
        }, f, indent=2)

    print(f"\nTop heads by (real - null_cross) mean translation-circuit contribution:")
    for h in head_list[:15]:
        print(f"  L{h['layer']:>2} H{h['head']:>2}  "
              f"real={h['real_mean']:+.4f}  "
              f"null_cross={h['null_cross_mean']:+.4f}  "
              f"null_same={h['null_same_mean']:+.4f}  "
              f"tc={h['translation_circuit_mean']:+.4f}  "
              f"clr={h['cross_lang_routing_mean']:+.4f}  "
              f"(n_dirs={h['n_directions']})")

    print(f"\nTop SAE features by translation-circuit contribution:")
    for s in sae_list[:15]:
        print(f"  f{s['feature_idx']:>5}  "
              f"real={s['real_mean']:.4f}  "
              f"null_cross={s['null_cross_mean']:.4f}  "
              f"null_same={s['null_same_mean']:.4f}  "
              f"tc={s['translation_circuit_mean']:+.4f}  "
              f"clr={s['cross_lang_routing_mean']:+.4f}  "
              f"(n_dirs={s['n_directions']})")

    return results, head_list, sae_list


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot(head_list, sae_list, img_dir: Path):
    import matplotlib.pyplot as plt
    img_dir.mkdir(parents=True, exist_ok=True)

    # Heads bar
    n = min(20, len(head_list))
    labels = [f"L{h['layer']}H{h['head']}" for h in head_list[:n]]
    real = np.array([h["real_mean"] for h in head_list[:n]])
    nc = np.array([h["null_cross_mean"] for h in head_list[:n]])
    ns = np.array([h["null_same_mean"] for h in head_list[:n]])

    x = np.arange(n)
    w = 0.27
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w, real, w, label="real (Phase 1)", color="crimson", alpha=0.85)
    ax.bar(x, nc, w, label="null cross-lang", color="steelblue", alpha=0.85)
    ax.bar(x + w, ns, w, label="null same-lang", color="goldenrod", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("mean |IE| (averaged over directions)")
    ax.set_title("Three-way decomposition — top heads by (real − null_cross)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(img_dir / "three_way_decomposition_heads.png", dpi=120)
    plt.close(fig)

    # SAE bar (top 30)
    n = min(30, len(sae_list))
    labels = [f"f{s['feature_idx']}" for s in sae_list[:n]]
    real = np.array([s["real_mean"] for s in sae_list[:n]])
    nc = np.array([s["null_cross_mean"] for s in sae_list[:n]])
    ns = np.array([s["null_same_mean"] for s in sae_list[:n]])

    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.bar(x - w, real, w, label="real (Phase 1)", color="crimson", alpha=0.85)
    ax.bar(x, nc, w, label="null cross-lang", color="steelblue", alpha=0.85)
    ax.bar(x + w, ns, w, label="null same-lang", color="goldenrod", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylabel("mean |IE| (averaged over directions)")
    ax.set_title("Three-way decomposition — top SAE features by (real − null_cross)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(img_dir / "three_way_decomposition_sae.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real_dir", default="outputs/gcm_translation")
    p.add_argument("--null_dir", default="outputs/gcm_translation_null")
    p.add_argument("--out_dir", default="outputs/gcm_translation_null/_aggregate")
    p.add_argument("--img_dir", default="experiments/gcm_translation/img")
    p.add_argument("--n_heads", type=int, default=32, help="heads per layer (Llama-3.1-8B = 32)")
    args = p.parse_args()

    results, head_list, sae_list = decompose(
        Path(args.real_dir), Path(args.null_dir), Path(args.out_dir), n_heads=args.n_heads
    )
    if results:
        plot(head_list, sae_list, Path(args.img_dir))
        print(f"\nWrote {args.out_dir}/three_way_decomposition.json")
        print(f"Wrote {args.out_dir}/three_way_summary.json")
        print(f"Wrote {args.img_dir}/three_way_decomposition_heads.png")
        print(f"Wrote {args.img_dir}/three_way_decomposition_sae.png")


if __name__ == "__main__":
    main()
