"""
Post-hoc analysis of counterfactual attribution results.

Loads the per-pair and aggregated results from counterfactual_attribution.py
and produces summary tables, cross-concept overlap analysis, and optionally
cross-references with existing probe-identified features.

Usage:
    python scripts/analyze_counterfactual_results.py \
        --results_dir outputs/counterfactual_attribution \
        --features_dir outputs/features
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lang_probing_src.config import OUTPUTS_DIR, FEATURES_DIR


def load_results(results_dir):
    """Load per-pair and aggregated results."""
    per_pair_path = os.path.join(results_dir, "per_pair_results.json")
    agg_path = os.path.join(results_dir, "aggregated_by_concept.json")

    with open(per_pair_path) as f:
        per_pair = json.load(f)
    with open(agg_path) as f:
        aggregated = json.load(f)

    return per_pair, aggregated


def print_summary_table(aggregated, top_n=20):
    """Print formatted summary tables."""
    print("\n" + "=" * 80)
    print("FEATURE ATTRIBUTION SUMMARY")
    print("=" * 80)

    for concept, data in sorted(aggregated.items()):
        n = data["num_pairs"]
        print(f"\n{'─' * 80}")
        print(f"  {concept} ({n} pairs)")
        print(f"{'─' * 80}")

        # By gradient
        print(f"\n  Top {top_n} by mean |gradient|:")
        print(f"  {'Rank':>4s}  {'Feature':>7s}  {'mean|grad|':>10s}  {'mean|g*a|':>10s}")
        print(f"  {'─'*4}  {'─'*7}  {'─'*10}  {'─'*10}")
        for rank, feat in enumerate(data["top_50_by_mean_grad"][:top_n], 1):
            print(
                f"  {rank:>4d}  {feat['feature_idx']:>7d}  "
                f"{feat['mean_abs_grad']:>10.5f}  "
                f"{feat['mean_abs_grad_x_act']:>10.5f}"
            )

        # By grad * activation
        print(f"\n  Top {top_n} by mean |gradient * activation|:")
        print(f"  {'Rank':>4s}  {'Feature':>7s}  {'mean|grad|':>10s}  {'mean|g*a|':>10s}")
        print(f"  {'─'*4}  {'─'*7}  {'─'*10}  {'─'*10}")
        for rank, feat in enumerate(data["top_50_by_mean_grad_x_act"][:top_n], 1):
            print(
                f"  {rank:>4d}  {feat['feature_idx']:>7d}  "
                f"{feat['mean_abs_grad']:>10.5f}  "
                f"{feat['mean_abs_grad_x_act']:>10.5f}"
            )


def analyze_cross_concept_overlap(aggregated, top_k=50, ranking="top_50_by_mean_grad"):
    """Find features that appear in top-k for multiple concepts."""
    print(f"\n\n{'=' * 80}")
    print(f"CROSS-CONCEPT FEATURE OVERLAP (top {top_k} by {ranking})")
    print(f"{'=' * 80}")

    # Collect top-k feature sets per concept
    concept_features = {}
    for concept, data in aggregated.items():
        features = set(f["feature_idx"] for f in data[ranking][:top_k])
        concept_features[concept] = features

    # Pairwise Jaccard similarity
    concepts = sorted(concept_features.keys())
    print(f"\n  Jaccard similarity between concept top-{top_k} sets:")
    print(f"  {'':>12s}", end="")
    for c in concepts:
        print(f"  {c[:8]:>8s}", end="")
    print()

    for c1 in concepts:
        print(f"  {c1[:12]:>12s}", end="")
        for c2 in concepts:
            s1, s2 = concept_features[c1], concept_features[c2]
            if len(s1 | s2) > 0:
                jaccard = len(s1 & s2) / len(s1 | s2)
            else:
                jaccard = 0.0
            print(f"  {jaccard:>8.3f}", end="")
        print()

    # Features shared across 2+ concepts
    feature_to_concepts = defaultdict(list)
    for concept, features in concept_features.items():
        for f in features:
            feature_to_concepts[f].append(concept)

    shared = {f: cs for f, cs in feature_to_concepts.items() if len(cs) >= 2}
    if shared:
        print(f"\n  Features in top-{top_k} for 2+ concepts ({len(shared)} features):")
        for feat_idx in sorted(shared, key=lambda x: -len(shared[x])):
            concepts_str = ", ".join(sorted(shared[feat_idx]))
            print(f"    feature {feat_idx:>5d}: {concepts_str}")
    else:
        print(f"\n  No features shared across concepts in top-{top_k}")


def cross_reference_with_probes(aggregated, features_dir, top_k=50):
    """
    Check if features identified by gradient attribution overlap with
    features previously identified by linear probes.
    """
    print(f"\n\n{'=' * 80}")
    print("CROSS-REFERENCE WITH PROBE-IDENTIFIED FEATURES")
    print(f"{'=' * 80}")

    if not os.path.exists(features_dir):
        print(f"\n  Features directory not found: {features_dir}")
        print("  Skipping cross-reference.")
        return

    # Load existing probe features
    probe_features = {}
    for fname in os.listdir(features_dir):
        if fname.endswith(".json"):
            concept_label = fname.replace(".json", "")
            with open(os.path.join(features_dir, fname)) as f:
                data = json.load(f)
                if isinstance(data, dict) and "features" in data:
                    probe_features[concept_label] = set(data["features"])
                elif isinstance(data, list):
                    probe_features[concept_label] = set(data)

    if not probe_features:
        print("\n  No probe feature files found.")
        return

    print(f"\n  Found probe features for: {', '.join(sorted(probe_features.keys()))}")

    for concept, data in sorted(aggregated.items()):
        grad_features = set(
            f["feature_idx"] for f in data["top_50_by_mean_grad"][:top_k]
        )
        gxa_features = set(
            f["feature_idx"] for f in data["top_50_by_mean_grad_x_act"][:top_k]
        )

        # Find matching probe feature sets
        for probe_label, probe_feats in sorted(probe_features.items()):
            if concept.lower() in probe_label.lower():
                overlap_grad = grad_features & probe_feats
                overlap_gxa = gxa_features & probe_feats

                print(f"\n  {concept} vs probe '{probe_label}':")
                print(f"    Probe features: {len(probe_feats)}")
                print(
                    f"    Overlap with top-{top_k} by |grad|: "
                    f"{len(overlap_grad)} features {sorted(overlap_grad)}"
                )
                print(
                    f"    Overlap with top-{top_k} by |g*a|: "
                    f"{len(overlap_gxa)} features {sorted(overlap_gxa)}"
                )


def per_pair_detail(per_pair, pair_id=None, top_n=10):
    """Print detailed per-pair results."""
    print(f"\n\n{'=' * 80}")
    print("PER-PAIR DETAIL")
    print(f"{'=' * 80}")

    for r in per_pair:
        if pair_id and r["pair_id"] != pair_id:
            continue

        print(f"\n{'─' * 60}")
        print(f"  Pair: {r['pair_id']}")
        print(f"  Prefix: '{r['prefix']}'")
        print(f"  Original: '{r['original_token']}' vs Counterfactual: '{r['counterfactual_token']}'")
        print(f"  Concept: {r['concept']} ({r.get('concept_value_orig', '?')} vs {r.get('concept_value_cf', '?')})")
        print(f"  Metric (logP_orig - logP_cf): {r['metric_value']:.4f}")

        sign = "+" if r["metric_value"] >= 0 else "-"
        pref = "original" if r["metric_value"] >= 0 else "counterfactual"
        print(f"  Model prefers: {pref} ({sign})")

        print(f"\n  Top {top_n} features by |grad| (last position):")
        for rank, feat in enumerate(r["top_k_by_grad_last_pos"][:top_n], 1):
            print(
                f"    {rank:>3d}. feature {feat['feature_idx']:>5d}: "
                f"grad={feat['grad']:+.4f}, act={feat['activation']:.4f}, "
                f"|g*a|={abs(feat['grad_x_act']):.4f}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze counterfactual attribution results"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join(OUTPUTS_DIR, "counterfactual_attribution"),
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        default=FEATURES_DIR,
        help="Directory with probe-identified features for cross-reference",
    )
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_n_display", type=int, default=20)
    parser.add_argument(
        "--pair_id",
        type=str,
        default=None,
        help="Show detail for a specific pair ID (or all if omitted)",
    )
    args = parser.parse_args()

    per_pair, aggregated = load_results(args.results_dir)

    print_summary_table(aggregated, top_n=args.top_n_display)
    analyze_cross_concept_overlap(aggregated, top_k=args.top_k, ranking="top_50_by_mean_grad")
    analyze_cross_concept_overlap(aggregated, top_k=args.top_k, ranking="top_50_by_mean_grad_x_act")
    cross_reference_with_probes(aggregated, args.features_dir, top_k=args.top_k)

    if args.pair_id:
        per_pair_detail(per_pair, pair_id=args.pair_id)
    else:
        # Show first 3 pairs as examples
        per_pair_detail(per_pair[:3] if len(per_pair) > 3 else per_pair)


if __name__ == "__main__":
    main()
