"""
Counterfactual Grammatical Attribution via SAE Feature Gradients.

For each sentence pair (shared prefix, two alternative next-tokens),
computes which SAE features at layer 16 are most responsible for the
model's logprob preference via gradient-based attribution.

Method:
    1. Forward prefix through Llama-3.1-8B with SAE encode/decode at L16
    2. Metric = logP(orig_token | prefix) - logP(cf_token | prefix)
    3. Backprop to SAE feature activations z to get d(metric)/d(z_i)
    4. Rank features by |grad| and |grad * activation|

Usage:
    python scripts/counterfactual_attribution.py \
        --data_file data/grammatical_pairs.json \
        --output_dir outputs/counterfactual_attribution \
        --save_raw_tensors
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

# Safety net for running from a non-installed checkout; harmless otherwise.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from lang_probing_src.config import (
    MODEL_ID,
    SAE_ID,
    LAYER_NUM,
    SAE_DIM,
    TRACER_KWARGS,
    OUTPUTS_DIR,
)
from lang_probing_src.utils import setup_model, get_device_info

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenization validation
# ---------------------------------------------------------------------------

def validate_pairs(pairs, tokenizer):
    """
    Validate that each token in the pairs encodes to exactly one token.
    Returns (valid_pairs, skipped_pairs).
    """
    valid = []
    skipped = []
    for pair in pairs:
        orig_ids = tokenizer.encode(pair["original_token"], add_special_tokens=False)
        cf_ids = tokenizer.encode(pair["counterfactual_token"], add_special_tokens=False)

        orig_ok = len(orig_ids) == 1
        cf_ok = len(cf_ids) == 1

        if not orig_ok or not cf_ok:
            reason = []
            if not orig_ok:
                reason.append(
                    f"original_token '{pair['original_token']}' -> {len(orig_ids)} tokens {orig_ids}"
                )
            if not cf_ok:
                reason.append(
                    f"counterfactual_token '{pair['counterfactual_token']}' -> {len(cf_ids)} tokens {cf_ids}"
                )
            logger.warning(f"SKIPPING {pair['id']}: {'; '.join(reason)}")
            skipped.append(pair)
        else:
            pair["_orig_tok_id"] = orig_ids[0]
            pair["_cf_tok_id"] = cf_ids[0]
            # Decode back to verify
            orig_decoded = tokenizer.decode([orig_ids[0]])
            cf_decoded = tokenizer.decode([cf_ids[0]])
            logger.info(
                f"  {pair['id']}: prefix='{pair['prefix']}' | "
                f"orig='{orig_decoded}' (id={orig_ids[0]}) | "
                f"cf='{cf_decoded}' (id={cf_ids[0]}) | "
                f"concept={pair['concept']}"
            )
            valid.append(pair)

    return valid, skipped


# ---------------------------------------------------------------------------
# Core: gradient-based SAE feature attribution
# ---------------------------------------------------------------------------

def compute_attribution(
    model, submodule, autoencoder, tokenizer, pair, device
):
    """
    For a single sentence pair, compute gradient-based attribution of SAE
    features to the logprob difference metric.

    Returns dict with:
        - metric_value: logP(orig) - logP(cf)
        - grad_last: [SAE_DIM] gradient at last token position
        - act_last: [SAE_DIM] SAE feature activations at last token position
        - grad_all: [seq_len, SAE_DIM] gradient at all positions
        - act_all: [seq_len, SAE_DIM] activations at all positions
    """
    prefix = pair["prefix"]
    orig_tok_id = pair["_orig_tok_id"]
    cf_tok_id = pair["_cf_tok_id"]

    # Tokenize prefix
    inputs = tokenizer(prefix, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]  # [1, S]

    # Llama layer outputs are always (hidden_states, ...) tuples, model is bfloat16
    layer_dtype = torch.bfloat16

    # --- Step 1: Pre-compute SAE activations (no gradients) ---
    # Cannot backprop through encode() due to Heaviside gate
    with model.trace(input_ids, **TRACER_KWARGS), torch.no_grad():
        x = submodule.output[0]  # [1, S, d_model] — always tuple[0] for Llama

        f = autoencoder.encode(x)           # [1, S, SAE_DIM]
        x_hat = autoencoder.decode(f)       # [1, S, d_model]
        residual = x - x_hat               # SAE reconstruction error

        f_saved = f.save()
        res_saved = residual.save()

    # --- Step 2: Gradient pass ---
    # Treat z as a leaf variable, backprop through decode -> rest of model -> logits
    z = f_saved.detach().clone().requires_grad_(True)
    res = res_saved.detach().clone()

    with model.trace(input_ids, **TRACER_KWARGS):
        # Replace layer 16 output with SAE reconstruction
        reconstructed = autoencoder.decode(z) + res
        reconstructed = reconstructed.to(layer_dtype)

        submodule.output[0][:] = reconstructed

        # Get logits at last token position (where next token is predicted)
        logits = model.lm_head.output       # [1, S, vocab_size]
        last_logits = logits[:, -1, :]      # [1, vocab_size]

        # Compute metric: logP(original) - logP(counterfactual)
        log_probs = F.log_softmax(last_logits.float(), dim=-1)
        metric = (log_probs[:, orig_tok_id] - log_probs[:, cf_tok_id]).mean()
        metric_saved = metric.save()

    # Backpropagate
    metric_saved.backward()

    # Extract results
    if z.grad is None:
        raise RuntimeError(
            "z.grad is None — gradients did not flow through the SAE decode. "
            "Check that autoencoder.decode() is in the computation graph."
        )

    grad = z.grad.detach().clone()          # [1, S, SAE_DIM]
    act = f_saved.detach().clone()          # [1, S, SAE_DIM]

    # Remove batch dim
    grad = grad.squeeze(0)                  # [S, SAE_DIM]
    act = act.squeeze(0)                    # [S, SAE_DIM]

    return {
        "metric_value": metric_saved.item(),
        "grad_last": grad[-1],             # [SAE_DIM]
        "act_last": act[-1],               # [SAE_DIM]
        "grad_all": grad,                  # [S, SAE_DIM]
        "act_all": act,                    # [S, SAE_DIM]
    }


# ---------------------------------------------------------------------------
# Feature ranking helpers
# ---------------------------------------------------------------------------

def get_top_k_features(grad, act, k=50):
    """
    Rank features by |grad| and |grad * act|, return top-k for both.

    Args:
        grad: [SAE_DIM] tensor
        act: [SAE_DIM] tensor
        k: number of top features to return

    Returns:
        (top_by_grad, top_by_grad_x_act) — each is a list of dicts
    """
    abs_grad = grad.abs()
    grad_x_act = (grad * act).abs()

    # Top-k by |gradient|
    vals_g, idxs_g = abs_grad.topk(k)
    top_by_grad = []
    for i in range(k):
        idx = idxs_g[i].item()
        top_by_grad.append({
            "feature_idx": idx,
            "grad": grad[idx].item(),
            "abs_grad": vals_g[i].item(),
            "activation": act[idx].item(),
            "grad_x_act": (grad[idx] * act[idx]).item(),
        })

    # Top-k by |gradient * activation|
    vals_ga, idxs_ga = grad_x_act.topk(k)
    top_by_grad_x_act = []
    for i in range(k):
        idx = idxs_ga[i].item()
        top_by_grad_x_act.append({
            "feature_idx": idx,
            "grad": grad[idx].item(),
            "abs_grad": grad[idx].abs().item(),
            "activation": act[idx].item(),
            "grad_x_act": vals_ga[i].item(),
        })

    return top_by_grad, top_by_grad_x_act


# ---------------------------------------------------------------------------
# Aggregation across sentence pairs
# ---------------------------------------------------------------------------

def aggregate_by_concept(results, k=50):
    """
    For each concept, compute mean |grad| and mean |grad*act| per feature
    across all pairs with that concept, then rank.
    """
    concept_grads = defaultdict(list)
    concept_acts = defaultdict(list)

    for r in results:
        concept = r["concept"]
        concept_grads[concept].append(r["_grad_last"])
        concept_acts[concept].append(r["_act_last"])

    aggregated = {}
    for concept in sorted(concept_grads.keys()):
        grads = torch.stack(concept_grads[concept])    # [N, SAE_DIM]
        acts = torch.stack(concept_acts[concept])      # [N, SAE_DIM]
        n_pairs = grads.shape[0]

        mean_abs_grad = grads.abs().mean(dim=0)        # [SAE_DIM]
        mean_abs_gxa = (grads * acts).abs().mean(dim=0)

        # Top-k by mean |grad|
        vals_g, idxs_g = mean_abs_grad.topk(k)
        top_by_grad = []
        for i in range(k):
            idx = idxs_g[i].item()
            top_by_grad.append({
                "feature_idx": idx,
                "mean_abs_grad": vals_g[i].item(),
                "mean_abs_grad_x_act": mean_abs_gxa[idx].item(),
                "num_pairs_in_top100": int(
                    (grads[:, idx].abs().topk(min(100, n_pairs)).values > 0).sum().item()
                ),
            })

        # Top-k by mean |grad * act|
        vals_ga, idxs_ga = mean_abs_gxa.topk(k)
        top_by_gxa = []
        for i in range(k):
            idx = idxs_ga[i].item()
            top_by_gxa.append({
                "feature_idx": idx,
                "mean_abs_grad": mean_abs_grad[idx].item(),
                "mean_abs_grad_x_act": vals_ga[i].item(),
                "num_pairs_in_top100": int(
                    ((grads[:, idx] * acts[:, idx]).abs().topk(min(100, n_pairs)).values > 0).sum().item()
                ),
            })

        aggregated[concept] = {
            "num_pairs": n_pairs,
            "top_50_by_mean_grad": top_by_grad,
            "top_50_by_mean_grad_x_act": top_by_gxa,
        }

    return aggregated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Counterfactual grammatical attribution via SAE feature gradients"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/grammatical_pairs.json",
        help="Path to sentence pairs JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(OUTPUTS_DIR, "counterfactual_attribution"),
        help="Directory to save results",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Number of top features to report per pair and per concept",
    )
    parser.add_argument(
        "--save_raw_tensors",
        action="store_true",
        help="Save full gradient and activation tensors as .pt files",
    )
    args = parser.parse_args()

    # Resolve data file path relative to project root
    # __file__ is at <project>/experiments/counterfactual_attribution/run.py,
    # so the project root is three parents up.
    project_root = Path(__file__).resolve().parent.parent.parent
    data_path = Path(args.data_file)
    if not data_path.is_absolute():
        data_path = project_root / data_path

    # --- Setup ---
    logger.info(f"Loading model {MODEL_ID} and SAE {SAE_ID}...")
    model, submodule, autoencoder, tokenizer = setup_model(MODEL_ID, SAE_ID)
    device, _ = get_device_info()
    logger.info(f"Model loaded on {device}")

    # --- Load and validate data ---
    logger.info(f"Loading sentence pairs from {data_path}")
    with open(data_path, "r") as f:
        pairs = json.load(f)
    logger.info(f"Loaded {len(pairs)} pairs")

    logger.info("Validating tokenization...")
    valid_pairs, skipped_pairs = validate_pairs(pairs, tokenizer)
    logger.info(
        f"Valid: {len(valid_pairs)}, Skipped: {len(skipped_pairs)}"
    )

    if len(valid_pairs) == 0:
        logger.error("No valid pairs! Exiting.")
        return

    # --- Create output directories ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_raw_tensors:
        raw_dir = output_dir / "raw_gradients"
        raw_dir.mkdir(exist_ok=True)

    # --- Run attribution for each pair ---
    per_pair_results = []

    for i, pair in enumerate(valid_pairs):
        logger.info(
            f"[{i+1}/{len(valid_pairs)}] Processing {pair['id']} "
            f"(concept={pair['concept']})"
        )

        try:
            result = compute_attribution(
                model, submodule, autoencoder, tokenizer, pair, device
            )
        except RuntimeError as e:
            logger.error(f"  ERROR: {e}")
            continue

        metric_val = result["metric_value"]
        logger.info(f"  metric (logP_orig - logP_cf) = {metric_val:.4f}")

        # Sanity check
        if metric_val < 0:
            logger.warning(
                f"  WARNING: metric is negative — model prefers counterfactual token"
            )

        # Rank features
        top_grad, top_gxa = get_top_k_features(
            result["grad_last"], result["act_last"], k=args.top_k
        )

        # Also compute sum-over-positions ranking
        grad_sum = result["grad_all"].sum(dim=0)   # [SAE_DIM]
        act_mean = result["act_all"].mean(dim=0)    # [SAE_DIM]
        top_grad_sum, top_gxa_sum = get_top_k_features(
            grad_sum, act_mean, k=args.top_k
        )

        pair_result = {
            "pair_id": pair["id"],
            "prefix": pair["prefix"],
            "original_token": pair["original_token"],
            "counterfactual_token": pair["counterfactual_token"],
            "concept": pair["concept"],
            "concept_value_orig": pair.get("concept_value_orig", ""),
            "concept_value_cf": pair.get("concept_value_cf", ""),
            "metric_value": metric_val,
            "top_k_by_grad_last_pos": top_grad,
            "top_k_by_grad_x_act_last_pos": top_gxa,
            "top_k_by_grad_sum_all_pos": top_grad_sum,
            "top_k_by_grad_x_act_sum_all_pos": top_gxa_sum,
        }
        # Keep tensors for aggregation
        pair_result["_grad_last"] = result["grad_last"].cpu()
        pair_result["_act_last"] = result["act_last"].cpu()

        per_pair_results.append(pair_result)

        # Optionally save raw tensors
        if args.save_raw_tensors:
            torch.save(
                result["grad_all"].cpu(),
                raw_dir / f"{pair['id']}_grad.pt",
            )
            torch.save(
                result["act_all"].cpu(),
                raw_dir / f"{pair['id']}_act.pt",
            )

        logger.info(
            f"  Top feature by |grad|: idx={top_grad[0]['feature_idx']}, "
            f"|grad|={top_grad[0]['abs_grad']:.4f}"
        )
        logger.info(
            f"  Top feature by |grad*act|: idx={top_gxa[0]['feature_idx']}, "
            f"|g*a|={top_gxa[0]['grad_x_act']:.4f}"
        )

    # --- Aggregate by concept ---
    logger.info("Aggregating results by concept...")
    aggregated = aggregate_by_concept(per_pair_results, k=args.top_k)

    for concept, agg in aggregated.items():
        logger.info(
            f"  {concept} ({agg['num_pairs']} pairs): "
            f"top feature by grad = {agg['top_50_by_mean_grad'][0]['feature_idx']}, "
            f"top feature by grad*act = {agg['top_50_by_mean_grad_x_act'][0]['feature_idx']}"
        )

    # --- Save results (strip internal tensor fields) ---
    serializable_results = []
    for r in per_pair_results:
        r_copy = {k: v for k, v in r.items() if not k.startswith("_")}
        serializable_results.append(r_copy)

    per_pair_path = output_dir / "per_pair_results.json"
    with open(per_pair_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
    logger.info(f"Saved per-pair results to {per_pair_path}")

    agg_path = output_dir / "aggregated_by_concept.json"
    with open(agg_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    logger.info(f"Saved aggregated results to {agg_path}")

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for concept, agg in aggregated.items():
        print(f"\n--- {concept} ({agg['num_pairs']} pairs) ---")
        print(f"  Top 10 features by mean |gradient|:")
        for feat in agg["top_50_by_mean_grad"][:10]:
            print(
                f"    feature {feat['feature_idx']:>5d}: "
                f"mean|grad|={feat['mean_abs_grad']:.4f}, "
                f"mean|g*a|={feat['mean_abs_grad_x_act']:.4f}"
            )
        print(f"  Top 10 features by mean |gradient * activation|:")
        for feat in agg["top_50_by_mean_grad_x_act"][:10]:
            print(
                f"    feature {feat['feature_idx']:>5d}: "
                f"mean|grad|={feat['mean_abs_grad']:.4f}, "
                f"mean|g*a|={feat['mean_abs_grad_x_act']:.4f}"
            )

    # Also save skipped pairs info
    if skipped_pairs:
        skip_path = output_dir / "skipped_pairs.json"
        with open(skip_path, "w") as f:
            json.dump(skipped_pairs, f, indent=2)
        logger.info(f"Saved {len(skipped_pairs)} skipped pairs to {skip_path}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
