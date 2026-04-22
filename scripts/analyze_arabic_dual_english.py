"""
E2 / #4 — Arabic dual → English "two/both/pair" analysis.

For top features ranked in Arabic (Number, Dual) or (Dual, Dual) attribution:
measure their SAE activation on English FLORES sentences, split into
  A: containing any of {two, both, pair, twin, couple}
  B: no numeric quantifier at all
Welch's t + Cohen's d. Null = firing-density-matched 20 random features.

Requires GPU (forward pass on FLORES sentences).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lang_probing_src.config import MODEL_ID, SAE_ID, TRACER_KWARGS, SAE_DIM
from lang_probing_src.utils import setup_model, get_device_info

DUAL_WORDS = {"two", "both", "pair", "twin", "twins", "couple"}
NUMBER_WORDS = DUAL_WORDS | {
    "one", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "several", "many", "few", "dozen", "dozens", "hundred", "thousand",
    "multiple", "most",
}
NUM_RE = re.compile(r"^\d+$")


def classify_sentence(sent):
    tokens = set(re.findall(r"[a-zA-Z]+", sent.lower()))
    numeric = bool(NUM_RE.search(sent)) or (tokens & NUMBER_WORDS)
    dualish = bool(tokens & DUAL_WORDS)
    if dualish:
        return "A"
    if not numeric:
        return "B"
    return None  # exclude (other numerals)


def mean_sae_activation(model, submodule, autoencoder, tokenizer, sentences, device, batch_size=4):
    """Return tensor [n_sentences, SAE_DIM] of mean SAE activation per sentence.
    Uses mean-pool over token positions."""
    outs = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        toks = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
        with model.trace(toks["input_ids"], **TRACER_KWARGS), torch.no_grad():
            x = submodule.output[0]                    # [B, S, d]
            f = autoencoder.encode(x)                  # [B, S, SAE]
            attn = toks["attention_mask"].unsqueeze(-1).float()  # [B,S,1]
            f_masked = f * attn
            mean = f_masked.sum(dim=1) / attn.sum(dim=1).clamp(min=1)
            mean_saved = mean.save()
        outs.append(mean_saved.detach().cpu())
    return torch.cat(outs, dim=0) if outs else torch.zeros(0, SAE_DIM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr_dir", default="outputs/overnight_multilingual/attribution")
    ap.add_argument("--output_dir", default="outputs/overnight_multilingual/analyses/arabic_dual_english")
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--n_random", type=int, default=20)
    ap.add_argument("--max_sents", type=int, default=1000)
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Find the Arabic dual cell
    candidates = [
        ("ara", "Dual", "Dual"),
        ("ara", "Dual", "Sing"),
        ("ara", "Number", "Dual"),
    ]
    top_features = None
    top_cell = None
    for lang, concept, value in candidates:
        cell_dir = Path(args.attr_dir) / lang / f"{concept}_{value}"
        tpath = cell_dir / "aggregated_signed_gxa.pt"
        if tpath.exists():
            t = torch.load(tpath, map_location="cpu").numpy()
            order = np.argsort(-np.abs(t))[: args.top_k]
            top_features = order
            top_cell = (lang, concept, value)
            print(f"using cell {top_cell} — top-{args.top_k} features by |signed grad*act|")
            break

    if top_features is None:
        print("No Arabic dual cell found; aborting")
        (out_dir / "null_result.md").write_text(
            "No Arabic dual cell available at time of analysis. "
            "The attribution run either did not produce the Dual cell or "
            "aggregation failed. Skipped E2.\n"
        )
        return

    # Load English FLORES
    from datasets import load_dataset
    ds = load_dataset("gsarti/flores_101", "eng", split="devtest")
    binA, binB = [], []
    for r in ds:
        label = classify_sentence(r["sentence"])
        if label == "A": binA.append(r["sentence"])
        elif label == "B": binB.append(r["sentence"])
    binA = binA[: args.max_sents]
    binB = binB[: args.max_sents]
    print(f"bin A (dualish): {len(binA)} | bin B (no numerals): {len(binB)}")

    # Load model + SAE
    print("loading model + SAE ...")
    model, submodule, autoencoder, tokenizer = setup_model(MODEL_ID, SAE_ID)
    device, _ = get_device_info()

    # Compute SAE activations
    print("computing activations on bin A ...")
    act_A = mean_sae_activation(model, submodule, autoencoder, tokenizer, binA, device)
    print("computing activations on bin B ...")
    act_B = mean_sae_activation(model, submodule, autoencoder, tokenizer, binB, device)

    # Random-feature null matched on firing density
    all_mean = torch.cat([act_A, act_B], dim=0).mean(dim=0).numpy()
    firing_density = (torch.cat([act_A, act_B], dim=0) > 0).float().mean(dim=0).numpy()

    # Match each target feature with ~5 random features of similar density
    target_densities = firing_density[top_features]
    rng = np.random.default_rng(42)
    null_features = []
    used = set(top_features.tolist())
    for td in target_densities:
        tol = 0.1
        mask = (np.abs(firing_density - td) < tol) & ~np.isin(np.arange(SAE_DIM), list(used))
        pool = np.where(mask)[0]
        if len(pool) > 0:
            pick = rng.choice(pool)
            null_features.append(pick)
            used.add(pick)
    null_features = np.array(null_features)

    # Per-feature effect size (Cohen's d) and Welch's t
    def per_feature_stats(features):
        out = []
        for f in features:
            a = act_A[:, f].numpy(); b = act_B[:, f].numpy()
            if a.std() == 0 and b.std() == 0:
                d = 0.0; t = 0.0; p = 1.0
            else:
                t, p = stats.ttest_ind(a, b, equal_var=False)
                pooled = np.sqrt(((a.var(ddof=1) if a.size > 1 else 0) + (b.var(ddof=1) if b.size > 1 else 0)) / 2) or 1e-8
                d = (a.mean() - b.mean()) / pooled
            out.append({"feature_idx": int(f), "t": float(t), "p": float(p),
                        "cohen_d": float(d), "mean_A": float(a.mean()), "mean_B": float(b.mean()),
                        "firing_density": float(firing_density[f])})
        return out

    target_stats = per_feature_stats(top_features)
    null_stats = per_feature_stats(null_features)

    # Plot: effect-size bar chart, target vs null
    fig, ax = plt.subplots(figsize=(10, 5))
    xs_t = np.arange(len(target_stats))
    xs_n = np.arange(len(null_stats)) + len(target_stats) + 1
    ax.bar(xs_t, [s["cohen_d"] for s in target_stats], color="tab:blue", label="top-K Arabic Dual features")
    ax.bar(xs_n, [s["cohen_d"] for s in null_stats], color="tab:gray", label="density-matched null")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(list(xs_t) + list(xs_n))
    ax.set_xticklabels(
        [f"f{s['feature_idx']}" for s in target_stats + null_stats],
        rotation=75, fontsize=7, ha="right"
    )
    ax.set_ylabel("Cohen's d  (mean_A - mean_B)/pooled_sd")
    ax.set_title(f"Do Arabic-Dual attribution features fire more on English 'two/both/pair' sentences?\n"
                 f"cell={top_cell}  |A|={len(binA)}, |B|={len(binB)}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "fig_arabic_dual_english.png", dpi=120)
    plt.close(fig)

    summary = {
        "source_cell": list(top_cell),
        "bin_A_count": len(binA),
        "bin_B_count": len(binB),
        "target_features": target_stats,
        "null_features": null_stats,
        "target_mean_cohen_d": float(np.mean([s["cohen_d"] for s in target_stats])),
        "null_mean_cohen_d": float(np.mean([s["cohen_d"] for s in null_stats])) if null_stats else 0.0,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({
        "target_mean_d": summary["target_mean_cohen_d"],
        "null_mean_d": summary["null_mean_cohen_d"],
    }, indent=2))


if __name__ == "__main__":
    main()
