"""
Decomposition of GCM IE per (head | SAE feature) across four quadrants:

    real_cross  (Phase 1, cross-lang, gold response anchored)
                  = content + cross-lang routing + translation circuit
    null_cross  (Phase 2, cross-lang, neither response is gold)
                  = content + cross-lang routing
    real_same   (Phase 1 addendum, src == tgt, gold response = the source itself)
                  = content + monolingual identity-completion circuit
    null_same   (Phase 2, src == tgt, neither response is gold)
                  = content alone

Per-component contribution estimates (averaged over directions):

    translation_circuit       = real_cross - null_cross
    cross_lang_routing        = null_cross - null_same
    identity_completion       = real_same  - null_same
    cross_minus_same_real     = real_cross - real_same
        (translation circuit beyond what monolingual completion already provides)

Outputs:
  outputs/gcm_translation_null/_aggregate/three_way_decomposition.json
  outputs/gcm_translation_null/_aggregate/three_way_summary.json
  experiments/gcm_translation/img/three_way_decomposition_<heads|sae>.png
  experiments/gcm_translation/img/identity_completion_<heads|sae>.png

NOTE on cross-direction averaging: per-feature means are computed over the
SAME support set (all cross-lang directions where data is available), not
over each direction's idiosyncratic top-K. So `real_mean` and `null_cross_mean`
for a given (layer, head) are taken from EVERY direction's full mean|IE|
vector, regardless of whether the head landed in that direction's top-30.
This makes cross-feature comparison meaningful.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_mean_abs_ie(direction_dir: Path, min_pairs: int = 90):
    """Returns (heads_mean_abs_ie [L*H], sae_mean_abs_ie [SAE_DIM]) or
    (None, None) if data missing, all NaN, or fewer than `min_pairs`
    successful pairs (avoids treating ~0/100-OOM direction outputs as
    valid signal)."""
    heads = sae = None
    h_path = direction_dir / "heads_ie.pt"
    s_path = direction_dir / "sae_ie.pt"
    if h_path.exists():
        t = torch.load(h_path, map_location="cpu", weights_only=False).float()
        keep = ~torch.isnan(t).any(dim=(1, 2))
        if int(keep.sum()) >= min_pairs:
            t = t[keep]
            heads = t.abs().mean(dim=0).flatten().numpy()  # [L*H]
    if s_path.exists():
        t = torch.load(s_path, map_location="cpu", weights_only=False).float()
        keep = ~torch.isnan(t).any(dim=1)
        if int(keep.sum()) >= min_pairs:
            t = t[keep]
            sae = t.abs().mean(dim=0).numpy()              # [SAE_DIM]
    return heads, sae


def load_all(real_dir: Path, null_dir: Path, min_pairs: int = 90):
    """Load full mean|IE| vectors for every available direction across both
    sweeps. Returns dict mapping (kind, name) -> (heads, sae), where kind
    is one of 'real_cross', 'real_same', 'null_cross', 'null_same'.

    Directions with < min_pairs successful pairs are silently treated as
    if their tensors are absent (load_mean_abs_ie returns None)."""
    out = {}
    for d in sorted(real_dir.iterdir() if real_dir.exists() else []):
        if not d.is_dir() or "__" not in d.name or d.name == "_aggregate":
            continue
        src, tgt = d.name.split("__", 1)
        kind = "real_same" if src == tgt else "real_cross"
        h, s = load_mean_abs_ie(d, min_pairs=min_pairs)
        out[(kind, d.name)] = (h, s)
    for d in sorted(null_dir.iterdir() if null_dir.exists() else []):
        if not d.is_dir() or "__" not in d.name or d.name == "_aggregate":
            continue
        src, tgt = d.name.split("__", 1)
        kind = "null_same" if src == tgt else "null_cross"
        h, s = load_mean_abs_ie(d, min_pairs=min_pairs)
        out[(kind, d.name)] = (h, s)
    return out


# ---------------------------------------------------------------------------
# Per-direction record (top-K with all 4 quadrants when available)
# ---------------------------------------------------------------------------


def per_direction_record(name: str, src: str, tgt: str, n_heads: int,
                         h_real, s_real, h_nc, s_nc, h_ns, s_ns) -> dict:
    """Per-direction top-30 heads and top-50 SAE features, ranked by
    real_cross mean|IE|, with the matching null_cross / null_same values
    looked up at the same indices."""
    rec = {"direction": name, "src": src, "tgt": tgt}

    top_idx_h = np.argsort(-h_real)[:30]
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
        for i in top_idx_h
    ]
    top_idx_s = np.argsort(-s_real)[:50]
    rec["sae_top50"] = [
        {
            "feature_idx": int(i),
            "real_mean_abs_ie": float(s_real[i]),
            "null_cross_mean_abs_ie": float(s_nc[i]),
            "null_same_mean_abs_ie": float(s_ns[i]),
            "translation_circuit": float(s_real[i] - s_nc[i]),
            "cross_lang_routing": float(s_nc[i] - s_ns[i]),
        }
        for i in top_idx_s
    ]
    return rec


# ---------------------------------------------------------------------------
# Cross-direction summary — UNIFORM SUPPORT
# ---------------------------------------------------------------------------


def _stack(vecs: list[np.ndarray]) -> np.ndarray:
    """Stack a list of [D] arrays to [n_dirs, D]."""
    return np.stack(vecs, axis=0)


def universal_summary(
    candidate_idx: np.ndarray,           # [K] — component indices to summarize
    real_stack: np.ndarray,              # [n_dirs, D]
    null_cross_stack: np.ndarray,        # [n_dirs, D]
    null_same_stack: np.ndarray,         # [n_dirs, D]
):
    """For each candidate component, compute (real, null_cross, null_same)
    means and stds across ALL cross-lang directions in the stacks (not
    only those where the candidate landed in the per-direction top-K).
    """
    n_dirs = real_stack.shape[0]
    rows = []
    for i in candidate_idx:
        real_vals = real_stack[:, i]
        nc_vals = null_cross_stack[:, i]
        ns_vals = null_same_stack[:, i]
        row = {
            "component_idx": int(i),
            "n_directions": int(n_dirs),
            "real_mean": float(real_vals.mean()),
            "real_std": float(real_vals.std()),
            "null_cross_mean": float(nc_vals.mean()),
            "null_cross_std": float(nc_vals.std()),
            "null_same_mean": float(ns_vals.mean()),
            "null_same_std": float(ns_vals.std()),
        }
        row["translation_circuit_mean"] = row["real_mean"] - row["null_cross_mean"]
        row["cross_lang_routing_mean"] = row["null_cross_mean"] - row["null_same_mean"]
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Same-language identity-completion summary
# ---------------------------------------------------------------------------


def identity_completion_summary(
    candidate_idx: np.ndarray,
    real_same_stack: np.ndarray,
    null_same_stack: np.ndarray,
):
    """Per-component (real_same - null_same) — the monolingual identity-
    completion circuit, isolated by removing the content-discrimination
    floor from the gold-anchored same-lang task.
    """
    n_dirs = real_same_stack.shape[0]
    rows = []
    for i in candidate_idx:
        real_vals = real_same_stack[:, i]
        ns_vals = null_same_stack[:, i]
        row = {
            "component_idx": int(i),
            "n_directions": int(n_dirs),
            "real_same_mean": float(real_vals.mean()),
            "real_same_std": float(real_vals.std()),
            "null_same_mean": float(ns_vals.mean()),
            "null_same_std": float(ns_vals.std()),
        }
        row["identity_completion_mean"] = row["real_same_mean"] - row["null_same_mean"]
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Decomposition driver
# ---------------------------------------------------------------------------


def decompose(real_dir: Path, null_dir: Path, out_dir: Path, n_heads: int,
              require_real_same: bool = False, suffix: str = ""):
    """Compute the four-quadrant decomposition.

    require_real_same: if True, restrict cross-lang directions to those whose
        src language has a matching real_same / null_same pair. Use this for
        the "fully-populated" chart that draws all four bars per component
        from a uniform language subset.

    suffix: appended to output filenames so multiple runs (restricted vs full)
        don't overwrite each other.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = load_all(real_dir, null_dir)

    # Which src langs have real_same available with sufficient pair count?
    # Both heads and SAE must be loaded (else either modality is unusable).
    src_with_real_same = {
        name.split("__", 1)[0]
        for (kind, name), (h, s) in raw.items()
        if kind == "real_same" and h is not None and s is not None
    }

    # --- Per-direction records (cross-language only) ---
    results = []
    cross_real_h, cross_real_s = [], []
    cross_nc_h, cross_nc_s = [], []
    cross_ns_h, cross_ns_s = [], []   # null_same indexed by src lang
    cross_dir_names = []

    cross_real_dirs = sorted(
        name for (kind, name) in raw if kind == "real_cross"
    )
    for name in cross_real_dirs:
        src, tgt = name.split("__", 1)
        if require_real_same and src not in src_with_real_same:
            continue
        h_real, s_real = raw.get(("real_cross", name), (None, None))
        h_nc, s_nc = raw.get(("null_cross", name), (None, None))
        # Same-language null is shared across all cross-lang dirs with the same src
        h_ns, s_ns = raw.get(("null_same", f"{src}__{src}"), (None, None))
        if any(v is None for v in (h_real, s_real, h_nc, s_nc, h_ns, s_ns)):
            print(f"  skip {name}: missing component "
                  f"(real_cross h/s={h_real is not None}/{s_real is not None}, "
                  f"null_cross h/s={h_nc is not None}/{s_nc is not None}, "
                  f"null_same h/s={h_ns is not None}/{s_ns is not None})")
            continue
        rec = per_direction_record(name, src, tgt, n_heads,
                                   h_real, s_real, h_nc, s_nc, h_ns, s_ns)
        results.append(rec)
        cross_real_h.append(h_real); cross_real_s.append(s_real)
        cross_nc_h.append(h_nc);     cross_nc_s.append(s_nc)
        cross_ns_h.append(h_ns);     cross_ns_s.append(s_ns)
        cross_dir_names.append(name)

    with open(out_dir / f"three_way_decomposition{suffix}.json", "w") as f:
        json.dump(results, f, indent=2)

    if not results:
        print("No cross-lang directions had complete data; nothing to summarize")
        return results, [], [], [], []

    # --- Cross-direction uniform-support summary ---
    real_h_stack = _stack(cross_real_h)         # [n_dirs, L*H]
    nc_h_stack   = _stack(cross_nc_h)
    ns_h_stack   = _stack(cross_ns_h)
    real_s_stack = _stack(cross_real_s)         # [n_dirs, SAE_DIM]
    nc_s_stack   = _stack(cross_nc_s)
    ns_s_stack   = _stack(cross_ns_s)

    # Candidate set: union of per-direction top-30 heads (top-50 SAE)
    h_cands = set()
    s_cands = set()
    for rec in results:
        for h in rec["heads_top30"]:
            h_cands.add(h["layer"] * n_heads + h["head"])
        for s in rec["sae_top50"]:
            s_cands.add(s["feature_idx"])
    h_cands = np.array(sorted(h_cands), dtype=np.int64)
    s_cands = np.array(sorted(s_cands), dtype=np.int64)

    h_summary = universal_summary(h_cands, real_h_stack, nc_h_stack, ns_h_stack)
    s_summary = universal_summary(s_cands, real_s_stack, nc_s_stack, ns_s_stack)
    # Annotate heads with (layer, head) and SAE with feature_idx
    for r in h_summary:
        r["layer"] = int(r["component_idx"] // n_heads)
        r["head"] = int(r["component_idx"] % n_heads)
        del r["component_idx"]
    for r in s_summary:
        r["feature_idx"] = int(r["component_idx"])
        del r["component_idx"]

    h_summary.sort(key=lambda x: -x["translation_circuit_mean"])
    s_summary.sort(key=lambda x: -x["translation_circuit_mean"])

    # --- Same-language identity-completion summary ---
    # Only count directions where both heads and SAE were loadable (i.e.
    # ≥ min_pairs successful pairs after the gate in load_mean_abs_ie).
    same_real_dirs = sorted(
        name for (kind, name), (h, s) in raw.items()
        if kind == "real_same" and h is not None and s is not None
    )
    h_same_id = s_same_id = []
    if same_real_dirs:
        same_real_h, same_real_s = [], []
        same_null_h, same_null_s = [], []
        same_kept = []
        for name in same_real_dirs:
            h_rs, s_rs = raw.get(("real_same", name), (None, None))
            h_ns, s_ns = raw.get(("null_same", name), (None, None))
            if any(v is None for v in (h_rs, s_rs, h_ns, s_ns)):
                print(f"  skip same-lang {name}: missing component")
                continue
            same_real_h.append(h_rs); same_real_s.append(s_rs)
            same_null_h.append(h_ns); same_null_s.append(s_ns)
            same_kept.append(name)
        if same_kept:
            srh = _stack(same_real_h); snh = _stack(same_null_h)
            srs = _stack(same_real_s); sns = _stack(same_null_s)
            # Candidates: top-30 / top-50 by real_same mean across same-lang dirs
            srh_mean = srh.mean(axis=0)
            srs_mean = srs.mean(axis=0)
            h_cands_same = np.argsort(-srh_mean)[:50]
            s_cands_same = np.argsort(-srs_mean)[:100]
            h_same_id = identity_completion_summary(h_cands_same, srh, snh)
            s_same_id = identity_completion_summary(s_cands_same, srs, sns)
            for r in h_same_id:
                r["layer"] = int(r["component_idx"] // n_heads)
                r["head"] = int(r["component_idx"] % n_heads)
                del r["component_idx"]
            for r in s_same_id:
                r["feature_idx"] = int(r["component_idx"])
                del r["component_idx"]
            h_same_id.sort(key=lambda x: -x["identity_completion_mean"])
            s_same_id.sort(key=lambda x: -x["identity_completion_mean"])
            print(f"\nIdentity-completion summary built over {len(same_kept)} same-lang directions: {same_kept}")

    # --- Save summary JSON ---
    summary = {
        "n_cross_lang_directions": len(results),
        "n_same_lang_directions": len(same_real_dirs) if same_real_dirs else 0,
        "require_real_same": bool(require_real_same),
        "src_langs_with_real_same": sorted(src_with_real_same),
        "cross_lang_directions": cross_dir_names,
        "heads_ranked_by_translation_circuit": h_summary[:50],
        "sae_features_ranked_by_translation_circuit": s_summary[:100],
        "heads_ranked_by_identity_completion": h_same_id[:30] if h_same_id else [],
        "sae_features_ranked_by_identity_completion": s_same_id[:50] if s_same_id else [],
    }
    with open(out_dir / f"three_way_summary{suffix}.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- Print ---
    print(f"\nTop heads by translation-circuit contribution (real_cross − null_cross), "
          f"averaged over {len(results)} cross-lang directions on UNIFORM support:")
    for h in h_summary[:15]:
        print(f"  L{h['layer']:>2} H{h['head']:>2}  "
              f"real={h['real_mean']:+.4f}  "
              f"null_cross={h['null_cross_mean']:+.4f}  "
              f"null_same={h['null_same_mean']:+.4f}  "
              f"tc={h['translation_circuit_mean']:+.4f}  "
              f"clr={h['cross_lang_routing_mean']:+.4f}")

    print(f"\nTop SAE features by translation-circuit contribution:")
    for s in s_summary[:15]:
        print(f"  f{s['feature_idx']:>5}  "
              f"real={s['real_mean']:.4f}  "
              f"null_cross={s['null_cross_mean']:.4f}  "
              f"null_same={s['null_same_mean']:.4f}  "
              f"tc={s['translation_circuit_mean']:+.4f}  "
              f"clr={s['cross_lang_routing_mean']:+.4f}")

    if h_same_id:
        print(f"\nTop heads by identity-completion (real_same − null_same), "
              f"averaged over {len(same_real_dirs)} same-lang directions:")
        for h in h_same_id[:15]:
            print(f"  L{h['layer']:>2} H{h['head']:>2}  "
                  f"real_same={h['real_same_mean']:+.4f}  "
                  f"null_same={h['null_same_mean']:+.4f}  "
                  f"id={h['identity_completion_mean']:+.4f}")

    return results, h_summary, s_summary, h_same_id, s_same_id


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot(h_summary, s_summary, h_same_id, s_same_id, img_dir: Path,
         suffix: str = "", title_extra: str = ""):
    import matplotlib.pyplot as plt
    img_dir.mkdir(parents=True, exist_ok=True)
    w = 0.27

    title_suffix = f"\n{title_extra}" if title_extra else ""
    if h_summary:
        n = min(20, len(h_summary))
        labels = [f"L{h['layer']}H{h['head']}" for h in h_summary[:n]]
        real = np.array([h["real_mean"] for h in h_summary[:n]])
        nc = np.array([h["null_cross_mean"] for h in h_summary[:n]])
        ns = np.array([h["null_same_mean"] for h in h_summary[:n]])
        x = np.arange(n)
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(x - w, real, w, label="real_cross (Phase 1)", color="crimson", alpha=0.85)
        ax.bar(x, nc, w, label="null_cross", color="steelblue", alpha=0.85)
        ax.bar(x + w, ns, w, label="null_same", color="goldenrod", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("mean |IE| (uniform support across directions)")
        ax.set_title(f"Three-way decomposition — top heads by (real_cross − null_cross){title_suffix}")
        ax.legend(); plt.tight_layout()
        plt.savefig(img_dir / f"three_way_decomposition_heads{suffix}.png", dpi=120); plt.close(fig)

    if s_summary:
        n = min(30, len(s_summary))
        labels = [f"f{s['feature_idx']}" for s in s_summary[:n]]
        real = np.array([s["real_mean"] for s in s_summary[:n]])
        nc = np.array([s["null_cross_mean"] for s in s_summary[:n]])
        ns = np.array([s["null_same_mean"] for s in s_summary[:n]])
        x = np.arange(n)
        fig, ax = plt.subplots(figsize=(16, 4))
        ax.bar(x - w, real, w, label="real_cross", color="crimson", alpha=0.85)
        ax.bar(x, nc, w, label="null_cross", color="steelblue", alpha=0.85)
        ax.bar(x + w, ns, w, label="null_same", color="goldenrod", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_ylabel("mean |IE| (uniform support)")
        ax.set_title(f"Three-way decomposition — top SAE features by (real_cross − null_cross){title_suffix}")
        ax.legend(); plt.tight_layout()
        plt.savefig(img_dir / f"three_way_decomposition_sae{suffix}.png", dpi=120); plt.close(fig)

    if h_same_id:
        n = min(20, len(h_same_id))
        labels = [f"L{h['layer']}H{h['head']}" for h in h_same_id[:n]]
        rs = np.array([h["real_same_mean"] for h in h_same_id[:n]])
        ns = np.array([h["null_same_mean"] for h in h_same_id[:n]])
        x = np.arange(n)
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(x - w/2, rs, w, label="real_same", color="seagreen", alpha=0.85)
        ax.bar(x + w/2, ns, w, label="null_same", color="goldenrod", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("mean |IE| (across same-lang directions)")
        ax.set_title(f"Identity-completion — top heads by (real_same − null_same){title_suffix}")
        ax.legend(); plt.tight_layout()
        plt.savefig(img_dir / f"identity_completion_heads{suffix}.png", dpi=120); plt.close(fig)

    if s_same_id:
        n = min(30, len(s_same_id))
        labels = [f"f{s['feature_idx']}" for s in s_same_id[:n]]
        rs = np.array([s["real_same_mean"] for s in s_same_id[:n]])
        ns = np.array([s["null_same_mean"] for s in s_same_id[:n]])
        x = np.arange(n)
        fig, ax = plt.subplots(figsize=(16, 4))
        ax.bar(x - w/2, rs, w, label="real_same", color="seagreen", alpha=0.85)
        ax.bar(x + w/2, ns, w, label="null_same", color="goldenrod", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_ylabel("mean |IE|")
        ax.set_title(f"Identity-completion — top SAE features by (real_same − null_same){title_suffix}")
        ax.legend(); plt.tight_layout()
        plt.savefig(img_dir / f"identity_completion_sae{suffix}.png", dpi=120); plt.close(fig)


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
    p.add_argument("--mode", default="both", choices=["both", "full", "restricted"],
                   help="full = all available cross-lang dirs (same-lang real may be partial); "
                        "restricted = only cross-lang dirs whose src has matching real_same data; "
                        "both = produce both views.")
    args = p.parse_args()

    real_dir = Path(args.real_dir)
    null_dir = Path(args.null_dir)
    out_dir = Path(args.out_dir)
    img_dir = Path(args.img_dir)

    runs = []
    if args.mode in ("both", "full"):
        runs.append(("_full", False, "all available cross-lang dirs"))
    if args.mode in ("both", "restricted"):
        runs.append(("_restricted", True, "src restricted to langs with real_same"))

    raw_for_count = load_all(real_dir, null_dir)
    n_same = sum(
        1 for (k, _), (h, s) in raw_for_count.items()
        if k == "real_same" and h is not None and s is not None
    )

    for suffix, require, descr in runs:
        print(f"\n{'='*72}\n{descr}\n{'='*72}")
        results, h_summary, s_summary, h_same_id, s_same_id = decompose(
            real_dir, null_dir, out_dir, n_heads=args.n_heads,
            require_real_same=require, suffix=suffix,
        )
        if results:
            n_dirs = len(results)
            title_extra = f"({n_dirs} cross-lang dirs · {n_same} same-lang real dirs · {descr})"
            plot(h_summary, s_summary, h_same_id, s_same_id, img_dir,
                 suffix=suffix, title_extra=title_extra)
            print(f"\nWrote {out_dir}/three_way_decomposition{suffix}.json")
            print(f"Wrote {out_dir}/three_way_summary{suffix}.json")
            print(f"Wrote {img_dir}/three_way_decomposition_{{heads,sae}}{suffix}.png")
            if h_same_id:
                print(f"Wrote {img_dir}/identity_completion_{{heads,sae}}{suffix}.png")


if __name__ == "__main__":
    main()
