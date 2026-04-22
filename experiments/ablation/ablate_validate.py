"""
W6 — Causal ablation validation on holdout pairs.

For each cell (lang, concept, value), zero-ablate the top-K SAE features
(from aggregated_abs_gxa.pt) at cf_pos on held-out pairs. Measure
mean_logprob_delta on the correct verb vs. random-feature baseline.

Output: ablation/<lang>/<concept>_<value>/holdout_ablation.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from lang_probing_src.config import MODEL_ID, SAE_ID, TRACER_KWARGS, SAE_DIM
from lang_probing_src.utils import setup_model, get_device_info


def run_ablation_batch(model, submodule, autoencoder, tokenizer, pairs, feat_to_ablate, device):
    """For each pair, measure Δ logP(last_orig_id) at cf_pos when zeroing feat_to_ablate."""
    deltas = []
    # Build mask [SAE_DIM]; 0 at features to ablate, 1 elsewhere
    mask = torch.ones(32768, device=device)
    for f in feat_to_ablate:
        mask[int(f)] = 0.0

    for p in pairs:
        # Reconstruct input
        prefix_ids = tokenizer.encode(p["prefix"], add_special_tokens=True)
        orig_ids = tokenizer.encode(p["original_token"], add_special_tokens=False)
        cf_ids = tokenizer.encode(p["counterfactual_token"], add_special_tokens=False)
        if not orig_ids or not cf_ids:
            continue

        if len(orig_ids) == 1 and len(cf_ids) == 1:
            input_ids = prefix_ids
            last_orig = orig_ids[0]
            last_cf = cf_ids[0]
        else:
            input_ids = prefix_ids + orig_ids[:-1]
            last_orig = orig_ids[-1]
            last_cf = cf_ids[-1]
        if last_orig == last_cf:
            continue
        cf_pos = len(input_ids) - 1

        ids_t = torch.tensor([input_ids], device=device)
        layer_dtype = torch.bfloat16

        # Baseline (no ablation)
        with model.trace(ids_t, **TRACER_KWARGS), torch.no_grad():
            logits_base = model.lm_head.output[:, cf_pos, :].save()
        lp_base = F.log_softmax(logits_base.float(), dim=-1)
        base_orig = lp_base[0, last_orig].item()
        base_cf = lp_base[0, last_cf].item()

        # With ablation: decode(z_ablated) replacing layer output
        with model.trace(ids_t, **TRACER_KWARGS), torch.no_grad():
            x = submodule.output[0]
            f = autoencoder.encode(x)
            f_abl = f * mask
            x_hat_abl = autoencoder.decode(f_abl)
            x_hat_orig = autoencoder.decode(f)
            residual = x - x_hat_orig
            recon = (x_hat_abl + residual).to(layer_dtype)
            submodule.output[0][:] = recon
            logits_abl = model.lm_head.output[:, cf_pos, :].save()
        lp_abl = F.log_softmax(logits_abl.float(), dim=-1)
        abl_orig = lp_abl[0, last_orig].item()
        abl_cf = lp_abl[0, last_cf].item()

        deltas.append({
            "pair_id": p["id"],
            "base_orig": base_orig, "base_cf": base_cf,
            "abl_orig": abl_orig, "abl_cf": abl_cf,
            "delta_orig": abl_orig - base_orig,
            "delta_margin": (abl_orig - abl_cf) - (base_orig - base_cf),
        })

    if not deltas:
        return {"n": 0, "mean_delta_orig": 0.0, "mean_delta_margin": 0.0}

    return {
        "n": len(deltas),
        "mean_delta_orig": float(np.mean([d["delta_orig"] for d in deltas])),
        "mean_delta_margin": float(np.mean([d["delta_margin"] for d in deltas])),
        "per_pair": deltas,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr_dir", default="outputs/counterfactual_attribution/attribution")
    ap.add_argument("--pairs_root", default="data/multilingual_pairs")
    ap.add_argument("--output_dir", default="outputs/counterfactual_attribution/ablation")
    ap.add_argument("--langs", nargs="+", default=["fra", "spa", "tur", "ara", "eng"])
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--n_random", type=int, default=20)
    ap.add_argument("--max_pairs", type=int, default=30)
    ap.add_argument("--holdout_file", default=None,
                    help="JSON of {cell_key: [pair_ids]} — per-lang this is holdout_ids.json")
    args = ap.parse_args()

    out_root = Path(args.output_dir); out_root.mkdir(parents=True, exist_ok=True)

    # Load model + SAE (once)
    print("loading model + SAE ...")
    model, submodule, autoencoder, tokenizer = setup_model(MODEL_ID, SAE_ID)
    device, _ = get_device_info()
    rng = np.random.default_rng(42)

    for lang in args.langs:
        # Load pairs file
        pairs_file = Path(args.pairs_root) / f"{lang}.json"
        if lang == "eng":
            pairs_file = Path("data/grammatical_pairs.json")
        if not pairs_file.exists():
            print(f"[{lang}] pairs file missing"); continue
        with open(pairs_file) as f:
            all_pairs = json.load(f)

        # Group pairs by (concept, value) cell; first N per cell become holdout
        from collections import defaultdict
        by_cell_pairs = defaultdict(list)
        for p in all_pairs:
            p.setdefault("lang_code", lang)
            p.setdefault("source", "multiblimp" if lang != "eng" else "handcrafted_en")
            if lang == "eng":
                p["id"] = p.get("id", "eng_" + str(id(p)))
            cell_key = (p["concept"], p.get("concept_value_orig", ""))
            by_cell_pairs[cell_key].append(p)

        for (concept, value), cell_pairs in by_cell_pairs.items():
            # Take a slice of pairs as holdout
            hold_pairs = cell_pairs[: args.max_pairs]
            if not hold_pairs:
                continue

            # Top features for this cell
            agg_path = Path(args.attr_dir) / lang / f"{concept}_{value}" / "aggregated_abs_gxa.pt"
            if not agg_path.exists():
                continue
            agg = torch.load(agg_path, map_location="cpu").numpy()
            top_feats = np.argsort(-agg)[: args.top_k].tolist()
            rand_feats = rng.choice(SAE_DIM, size=args.n_random, replace=False).tolist()

            print(f"[{lang}/{concept}/{value}] ablating top-{args.top_k} on {len(hold_pairs)} pairs")
            t0 = time.time()
            res_top = run_ablation_batch(model, submodule, autoencoder, tokenizer,
                                         hold_pairs, top_feats, device)
            res_rand = run_ablation_batch(model, submodule, autoencoder, tokenizer,
                                          hold_pairs, rand_feats, device)

            cell_out = out_root / lang / f"{concept}_{value}"
            cell_out.mkdir(parents=True, exist_ok=True)
            with open(cell_out / "holdout_ablation.json", "w") as f:
                json.dump({
                    "top_features": top_feats,
                    "random_features": rand_feats,
                    "top_result": res_top,
                    "random_result": res_rand,
                }, f, indent=2)
            print(f"  top: Δorig={res_top['mean_delta_orig']:.3f} Δmargin={res_top['mean_delta_margin']:.3f}")
            print(f"  rand: Δorig={res_rand['mean_delta_orig']:.3f} Δmargin={res_rand['mean_delta_margin']:.3f}")
            print(f"  time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
