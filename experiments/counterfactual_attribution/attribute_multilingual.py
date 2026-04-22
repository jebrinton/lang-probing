"""
Multilingual counterfactual SAE feature attribution.

Key differences from scripts/counterfactual_attribution.py:
  - Handles multi-token counterfactuals via last-BPE strategy
  - Per-(lang, concept, value) cell aggregation with signed + abs + signed_gxa tensors
  - cf_position_idx per pair (always the last position of constructed input)
  - Backward compatible with data/grammatical_pairs.json (single-token, lang=eng)

Bug fixes applied vs. the English prototype:
  - A1: multi-token counterfactuals handled by prefixing orig_tok_ids[:-1] to input
  - A2: drop sum-over-all-positions secondary ranking (unprincipled); keep grad[-1] only
  - A4: per-value signed aggregation (required for sign-flip analysis)
  - A5: per-pair sanity logging (metric distribution, top-feat concentration)

Note on A3 (Heaviside STE): the existing script bypasses `encode()` in the
gradient pass (wraps in torch.no_grad() and uses z=encode(x) as a leaf variable
in a second forward). Gate non-differentiability is therefore NOT an issue for
this code — gradient flows through decode() only. STE fix dropped.
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from lang_probing_src.config import MODEL_ID, SAE_ID, TRACER_KWARGS, SAE_DIM
from lang_probing_src.utils import setup_model, get_device_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def prepare_pair(pair, tokenizer):
    """
    Tokenize a pair, set tok_strategy, build input_ids and pick cf_pos.

    Input pair fields expected: prefix, original_token, counterfactual_token,
    concept, concept_value_orig, concept_value_cf, (optional) lang_code.

    Mutates pair in-place with:
      _orig_tok_ids, _cf_tok_ids, _tok_strategy ("single"|"last"),
      _input_ids (list[int]), _cf_pos (int), _last_orig_id, _last_cf_id.

    Returns True if pair is usable, False if it must be skipped (e.g.
    original and counterfactual share the same last token).
    """
    orig_ids = tokenizer.encode(pair["original_token"], add_special_tokens=False)
    cf_ids = tokenizer.encode(pair["counterfactual_token"], add_special_tokens=False)

    pair["_orig_tok_ids"] = orig_ids
    pair["_cf_tok_ids"] = cf_ids

    # Tokenize prefix (add special tokens as the model would at start of sequence)
    prefix_ids = tokenizer.encode(pair["prefix"], add_special_tokens=True)

    single = (len(orig_ids) == 1 and len(cf_ids) == 1)
    pair["_tok_strategy"] = "single" if single else "last"

    if single:
        input_ids = prefix_ids
        last_orig = orig_ids[0]
        last_cf = cf_ids[0]
    else:
        # Last-BPE strategy: context = prefix + orig_tokens[:-1].
        # Predicts position logits[-1] to compare last orig vs last cf.
        input_ids = prefix_ids + orig_ids[:-1]
        last_orig = orig_ids[-1]
        last_cf = cf_ids[-1]

    # Skip degenerate case where last tokens coincide (common artifact of
    # multi-token pairs when only internal morphology differs)
    if last_orig == last_cf:
        return False

    pair["_input_ids"] = input_ids
    pair["_cf_pos"] = len(input_ids) - 1  # logits[-1] predicts last comparison token
    pair["_last_orig_id"] = last_orig
    pair["_last_cf_id"] = last_cf
    return True


def compute_attribution(model, submodule, autoencoder, pair, device):
    """Gradient-based SAE attribution. Returns dict with signed grad and act
    at cf_pos (shape [SAE_DIM] each) + metric value."""
    input_ids = torch.tensor([pair["_input_ids"]], device=device)
    cf_pos = pair["_cf_pos"]
    orig_id = pair["_last_orig_id"]
    cf_id = pair["_last_cf_id"]
    layer_dtype = torch.bfloat16

    # Step 1: Cache SAE features without gradients
    with model.trace(input_ids, **TRACER_KWARGS), torch.no_grad():
        x = submodule.output[0]
        f = autoencoder.encode(x)
        x_hat = autoencoder.decode(f)
        residual = x - x_hat
        f_saved = f.save()
        res_saved = residual.save()

    # Step 2: Backprop with z as leaf variable
    z = f_saved.detach().clone().requires_grad_(True)
    res = res_saved.detach().clone()

    with model.trace(input_ids, **TRACER_KWARGS):
        reconstructed = (autoencoder.decode(z) + res).to(layer_dtype)
        submodule.output[0][:] = reconstructed
        logits = model.lm_head.output  # [1, S, vocab]
        cf_logits = logits[:, cf_pos, :]
        log_probs = F.log_softmax(cf_logits.float(), dim=-1)
        metric = (log_probs[:, orig_id] - log_probs[:, cf_id]).mean()
        metric_saved = metric.save()

    metric_saved.backward()

    if z.grad is None:
        raise RuntimeError("z.grad is None — SAE decode not in gradient path.")

    grad = z.grad.detach().squeeze(0)       # [S, SAE_DIM]
    act = f_saved.detach().squeeze(0)       # [S, SAE_DIM]

    # SIGNED grad and act at cf_pos. Callers derive abs / grad*act as needed.
    return {
        "metric_value": metric_saved.item(),
        "grad_cf": grad[cf_pos].cpu(),      # [SAE_DIM] signed
        "act_cf": act[cf_pos].cpu(),        # [SAE_DIM] non-negative
    }


def aggregate_cell(per_pair, k=50):
    """Per-(lang, concept, value) aggregation. Returns dict with signed means
    and top-K by |grad| / |grad*act|, plus provenance counts."""
    grads = torch.stack([p["_grad_cf"] for p in per_pair])      # [N, SAE_DIM]
    acts = torch.stack([p["_act_cf"] for p in per_pair])        # [N, SAE_DIM]
    gxa = grads * acts                                           # [N, SAE_DIM] signed

    mean_signed_grad = grads.mean(dim=0)
    mean_signed_gxa = gxa.mean(dim=0)
    mean_abs_grad = grads.abs().mean(dim=0)
    mean_abs_gxa = gxa.abs().mean(dim=0)

    def _topk(scores, signed_reference, k):
        vals, idxs = scores.topk(k)
        return [
            {
                "feature_idx": int(idxs[i].item()),
                "score": float(vals[i].item()),
                "signed_mean": float(signed_reference[idxs[i]].item()),
            }
            for i in range(k)
        ]

    top_abs_grad = _topk(mean_abs_grad, mean_signed_grad, k)
    top_abs_gxa = _topk(mean_abs_gxa, mean_signed_gxa, k)

    n_single = sum(1 for p in per_pair if p.get("tok_strategy") == "single")
    source_counts = defaultdict(int)
    for p in per_pair:
        source_counts[p.get("source", "unknown")] += 1

    return {
        "n_pairs": len(per_pair),
        "n_single": n_single,
        "n_last": len(per_pair) - n_single,
        "source_counts": dict(source_counts),
        "top_50_by_abs_grad": top_abs_grad,
        "top_50_by_abs_gxa": top_abs_gxa,
        "_tensors": {
            "aggregated_signed": mean_signed_grad,
            "aggregated_abs": mean_abs_grad,
            "aggregated_signed_gxa": mean_signed_gxa,
            "aggregated_abs_gxa": mean_abs_gxa,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_file", required=True, type=str)
    ap.add_argument("--lang_code", required=True, type=str,
                    help="Used if pair lacks lang_code field; also determines output subdir")
    ap.add_argument("--output_dir", required=True, type=str)
    ap.add_argument("--max_pairs_per_cell", type=int, default=300)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--save_raw_tensors", action="store_true")
    ap.add_argument("--holdout_frac", type=float, default=0.2,
                    help="Fraction of pairs per cell held out for ablation validation")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    import random; random.seed(args.seed)

    outroot = Path(args.output_dir)
    outroot.mkdir(parents=True, exist_ok=True)

    logger.info(f"[{args.lang_code}] Loading {MODEL_ID} + SAE {SAE_ID}...")
    model, submodule, autoencoder, tokenizer = setup_model(MODEL_ID, SAE_ID)
    device, _ = get_device_info()
    logger.info(f"[{args.lang_code}] Model on {device}")

    with open(args.pairs_file) as f:
        pairs = json.load(f)
    logger.info(f"[{args.lang_code}] Loaded {len(pairs)} pairs from {args.pairs_file}")

    # Prepare + group by cell
    by_cell = defaultdict(list)
    n_skip = 0
    for p in pairs:
        p.setdefault("lang_code", args.lang_code)
        p.setdefault("source", "unknown")
        if not prepare_pair(p, tokenizer):
            n_skip += 1
            continue
        cell = (p["lang_code"], p["concept"], p.get("concept_value_orig", ""))
        by_cell[cell].append(p)

    logger.info(f"[{args.lang_code}] {len(by_cell)} cells; {n_skip} pairs skipped (degenerate)")

    holdout_out = outroot / "holdout_ids.json"
    holdout_by_cell = {}

    for cell, cell_pairs in sorted(by_cell.items()):
        # Cap
        random.shuffle(cell_pairs)
        cell_pairs = cell_pairs[: args.max_pairs_per_cell]
        # Hold-out split
        n_hold = max(1, int(len(cell_pairs) * args.holdout_frac))
        hold = cell_pairs[:n_hold]
        work = cell_pairs[n_hold:]
        holdout_by_cell[f"{cell[0]}|{cell[1]}|{cell[2]}"] = [p["id"] for p in hold]

        lang, concept, value = cell
        cell_dir = outroot / lang / f"{concept}_{value}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = cell_dir / "raw_gradients" if args.save_raw_tensors else None
        if raw_dir: raw_dir.mkdir(exist_ok=True)

        logger.info(f"[{lang}] cell={concept}/{value}: {len(work)} train + {len(hold)} holdout")
        t0 = time.time()

        per_pair = []
        for i, p in enumerate(work):
            try:
                r = compute_attribution(model, submodule, autoencoder, p, device)
            except Exception as e:
                logger.error(f"  [{p['id']}] {type(e).__name__}: {e}")
                continue

            row = {
                "pair_id": p["id"],
                "metric_value": r["metric_value"],
                "cf_pos": p["_cf_pos"],
                "source": p.get("source"),
                "tok_strategy": p["_tok_strategy"],
                "_grad_cf": r["grad_cf"],
                "_act_cf": r["act_cf"],
            }
            per_pair.append(row)

            if args.save_raw_tensors and raw_dir:
                torch.save(r["grad_cf"], raw_dir / f"{p['id']}_grad.pt")
                torch.save(r["act_cf"], raw_dir / f"{p['id']}_act.pt")

            if (i + 1) % 50 == 0:
                logger.info(f"  [{lang}/{concept}/{value}] {i+1}/{len(work)} done "
                            f"({(i+1)/(time.time()-t0):.2f}/s)")

        if not per_pair:
            logger.warning(f"  [{lang}/{concept}/{value}] no successful pairs")
            continue

        # Sanity stats
        metrics = [p["metric_value"] for p in per_pair]
        n_neg = sum(1 for m in metrics if m < 0)
        sanity = {
            "n_pairs": len(per_pair),
            "metric_mean": float(sum(metrics) / len(metrics)),
            "metric_neg_frac": n_neg / len(per_pair),
            "metric_min": float(min(metrics)),
            "metric_max": float(max(metrics)),
        }

        # Aggregate
        agg = aggregate_cell(per_pair, k=args.top_k)

        # Save tensors
        for name, tensor in agg["_tensors"].items():
            torch.save(tensor, cell_dir / f"{name}.pt")
        summary = {k: v for k, v in agg.items() if k != "_tensors"}
        summary["sanity"] = sanity
        summary["pair_count_budget"] = {"train": len(work), "holdout": n_hold}

        with open(cell_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Per-pair json (strip internal tensors)
        ser = [
            {k: v for k, v in p.items() if not k.startswith("_")}
            for p in per_pair
        ]
        with open(cell_dir / "per_pair_results.json", "w") as f:
            json.dump(ser, f, indent=2)

        elapsed = time.time() - t0
        logger.info(
            f"  [{lang}/{concept}/{value}] saved {len(per_pair)} pairs "
            f"in {elapsed:.1f}s; metric_mean={sanity['metric_mean']:.3f}, "
            f"neg_frac={sanity['metric_neg_frac']:.2f}"
        )

    with open(holdout_out, "w") as f:
        json.dump(holdout_by_cell, f, indent=2)
    logger.info(f"[{args.lang_code}] all cells done; holdouts at {holdout_out}")


if __name__ == "__main__":
    main()
