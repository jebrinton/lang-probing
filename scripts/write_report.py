"""
Wave 10 — compile REPORT.md from analysis outputs.

Safety-net design: never fails. If an analysis section has no data, writes
"(not available — see TODO.md)". Always produces a valid markdown file.
"""
import json
from datetime import datetime
from pathlib import Path

ROOT = Path("outputs/overnight_multilingual")


def safe_json(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def fig_ref(path, alt=None):
    p = ROOT / path
    if p.exists():
        return f"![{alt or path}]({path})"
    return f"_figure missing: `{path}`_"


def main():
    lines = []
    ap = lines.append

    # 1. TL;DR
    ap(f"# Morning report — overnight multilingual SAE feature attribution\n")
    ap(f"_session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n")
    ap("## TL;DR\n")

    data_snap = safe_json(ROOT / "data_snapshot.json") or {}
    total = sum(v.get("n_pairs", 0) for v in data_snap.values())
    ap(f"- **Data:** {total} multilingual pairs across {len(data_snap)} languages "
       f"(Multi-BLiMP `jumelet/multiblimp`). Per-cell counts in `data_snapshot.json`.")

    cc_sum = safe_json(ROOT / "analyses/cross_concept/summary.json") or {}
    if cc_sum:
        hits = sum(v.get("n_features_k_ge_2_sig", 0) for v in cc_sum.values())
        ap(f"- **Cross-concept (#3):** {hits} features significantly appear in top-50 "
           f"of ≥2 concept cells across languages (Bonferroni p<0.01). "
           f"→ `analyses/cross_concept/fig_cross_concept_*.png`")

    ad_sum = safe_json(ROOT / "analyses/arabic_dual_english/summary.json") or {}
    if ad_sum:
        ap(f"- **Arabic dual → English (#4):** target features' mean Cohen's d = "
           f"{ad_sum.get('target_mean_cohen_d', 0):.3f} vs. null "
           f"{ad_sum.get('null_mean_cohen_d', 0):.3f}. "
           f"→ `analyses/arabic_dual_english/fig_arabic_dual_english.png`")
    else:
        ap("- **Arabic dual → English (#4):** not available; see TODO.md.")

    sf_sum = safe_json(ROOT / "analyses/sign_flip/summary.json") or []
    if sf_sum:
        opp_total = sum(s.get("n_opp_sign", 0) for s in sf_sum)
        ap(f"- **Sign-flip (#5):** {opp_total} opposite-sign feature pairs across "
           f"{len(sf_sum)} language/cell comparisons. "
           f"→ `analyses/sign_flip/fig_sign_flip_*.png`")

    io_sum = safe_json(ROOT / "analyses/input_vs_output/summary.json") or []
    if io_sum:
        ap(f"- **Input vs output (#6):** {len(io_sum)} cells ablation-validated. "
           f"→ `analyses/input_vs_output/fig_input_vs_output_*.png`")

    ap("")

    # 2. Method
    ap("## Method\n")
    ap("Attribution metric: `logP(orig_last_token) - logP(cf_last_token)` at "
       "the position `cf_pos = len(input_ids) - 1`, where for single-BPE "
       "counterfactuals `input = prefix` and for multi-BPE (Arabic mostly) "
       "`input = prefix + orig_tokens[:-1]` (last-BPE strategy).\n")
    ap("Features ranked by `|grad × act|` at cf_pos. This is the zero-baseline "
       "indirect-effect approximation of activation patching — cheap single-pass "
       "(one forward, one backward) but NOT Syed-style two-pass attribution "
       "patching. See `TODO.md` for a Syed comparison as follow-up.\n")
    ap("Gated SAE encode/decode: encode is run under `torch.no_grad()`; SAE "
       "features `z` are treated as a leaf variable in the gradient pass, so "
       "the non-differentiable Heaviside gate is NOT in the gradient path "
       "(unlike what the plan initially assumed — confirmed by reading the "
       "existing counterfactual_attribution.py).\n")

    # 3. Data
    ap("## Data\n")
    ap("| lang | n_pairs | cells |")
    ap("|---|---|---|")
    for lang, v in sorted(data_snap.items()):
        cells = v.get("per_cell", {})
        cell_str = ", ".join(f"{k}:{n}" for k, n in sorted(cells.items()))
        ap(f"| {lang} | {v.get('n_pairs', 0)} | {cell_str} |")
    ap("")
    ap("**Multi-BLiMP doesn't cover Tense for any of the four languages.** "
       "Template supplement deferred (scope cut under usage budget). See TODO.md.\n")
    ap("**Turkish has no grammatical Gender** — expected null; no attempt to hack it.\n")

    # Bug audit
    ap("## Bug-audit appendix\n")
    ap(fig_ref("bug_audit/per_pair_metric_distribution.png",
               "Per-cell mean of logP_orig - logP_cf"))
    ap("")
    ap(fig_ref("bug_audit/metric_negative_fraction.png",
               "Fraction of pairs with metric<0"))
    ap("")
    ap(fig_ref("bug_audit/tok_strategy_counts.png",
               "Tok-strategy composition per cell"))
    ap("")
    ap("**Cells with metric_neg_frac > 0.3** may indicate mislabeled original/cf "
       "or model not actually preferring the 'correct' verb; treat those cells' "
       "top features with caution.\n")

    # 4. Findings
    ap("## Findings\n")

    # 4.1 Cross-concept
    ap("### 4.1 Cross-concept features (`#3`)\n")
    if cc_sum:
        ap("Per-language analysis of features appearing in top-50 of multiple "
           "(concept, value) cells. Random baseline: Binomial(n_cells, 50/32768). "
           "Bonferroni over 32768 features.\n")
        for lang, v in sorted(cc_sum.items()):
            ap(f"**{lang}** — {v.get('n_cells', 0)} cells, "
               f"{v.get('n_features_k_ge_2', 0)} features in ≥2 cells "
               f"({v.get('n_features_k_ge_2_sig', 0)} p_bonf<0.01), "
               f"{v.get('n_features_k_ge_3', 0)} in ≥3 cells.  ")
            top3 = v.get("top_3_features", [])
            if top3:
                ap("Top 3 features:")
                for r in top3:
                    ap(f"- f{r['feature_idx']}: {r['n_cells']} cells ({r['cells']}), p_bonf={r['pval_bonferroni']:.2e}")
            ap("")
        for lang in cc_sum:
            ap(fig_ref(f"analyses/cross_concept/fig_cross_concept_{lang}.png",
                       f"cross-concept {lang}"))
            ap("")
    else:
        ap("_not available — see TODO.md_\n")

    # 4.2 Arabic dual
    ap("### 4.2 Arabic dual → English 'two/both/pair' (`#4`)\n")
    if ad_sum:
        ap(f"Source cell: `{ad_sum.get('source_cell')}`. "
           f"|bin A (dualish English)|={ad_sum.get('bin_A_count')}, "
           f"|bin B (no numerals)|={ad_sum.get('bin_B_count')}.\n")
        ap(f"- Target features' mean Cohen's d: **{ad_sum.get('target_mean_cohen_d', 0):.3f}**")
        ap(f"- Density-matched null's mean Cohen's d: **{ad_sum.get('null_mean_cohen_d', 0):.3f}**")
        ap("")
        ap(fig_ref("analyses/arabic_dual_english/fig_arabic_dual_english.png",
                   "Arabic dual → English two/both/pair"))
        ap("")
        ap("Interpretation: an effect size meaningfully above the null supports H4 "
           "(translation reuses the same abstract dual-number feature even in "
           "English, despite English not having morphological dual). A null-level "
           "effect is evidence against cross-lingual feature reuse for this concept.\n")
    else:
        ap("_not available — see TODO.md_\n")

    # 4.3 Sign-flip
    ap("### 4.3 Sign-flip across Romance (`#5`)\n")
    if sf_sum:
        for s in sf_sum:
            ap(f"- {s['pair']} on **{s['cell']}**: {s['n_same_sign']} same-sign, "
               f"**{s['n_opp_sign']} opposite-sign** (top features).")
        ap("")
        ap(fig_ref("analyses/sign_flip/fig_sign_flip_fra_spa_Gender_Fem.png",
                   "fra vs spa Fem"))
        ap("")
        ap(fig_ref("analyses/sign_flip/fig_sign_flip_fra_ara_Gender_Fem.png",
                   "fra vs ara Fem"))
        ap("")
        ap("Sign-flip candidates are features that attribute POSITIVELY for "
           "femininity in one language and NEGATIVELY in another. These are "
           "interesting for H2 — if the SAME SAE feature carries opposite "
           "grammatical meaning in different languages, the 'feature' is being "
           "repurposed language-specifically, against a pure shared-feature "
           "hypothesis.\n")
    else:
        ap("_not available — see TODO.md_\n")

    # 4.4 Input vs output
    ap("### 4.4 Input activation vs output ablation (`#6`)\n")
    if io_sum:
        ap(f"{len(io_sum)} cells validated. Per-cell comparison of top-K features' "
           f"ablation effect vs. random-K baseline:\n")
        ap("| lang | concept | value | n_holdout | Δorig top-K | Δorig random | ratio |")
        ap("|---|---|---|---|---|---|---|")
        for s in io_sum:
            r = s.get('effect_ratio_over_random')
            ap(f"| {s['lang']} | {s['concept']} | {s['value']} | {s['n_holdout']} | "
               f"{s['mean_delta_orig_top']:.3f} | {s['mean_delta_orig_random']:.3f} | "
               f"{r:.2f} |" if r else f"| {s['lang']} | {s['concept']} | {s['value']} | "
               f"{s['n_holdout']} | {s['mean_delta_orig_top']:.3f} | "
               f"{s['mean_delta_orig_random']:.3f} | — |")
        ap("")
        ap("**Holy-grail finding signal:** cells where `ratio_top/random >> 1` "
           "show top-attribution features ARE causally necessary. Cells where the "
           "ratio is near 1 mean the attribution identifies correlative but "
           "non-causal features — the input/output asymmetry from hypothesis #6. "
           "Cross-lingual matrix (same features, different target cells) is in "
           "`fig_holy_grail_matrix.png` if completed.\n")
    else:
        ap("_not available — see TODO.md_\n")

    # 5. Honesty notes
    ap("## Honesty notes\n")
    ap("- All pair counts shown INCLUDE multi-token Arabic cases where we use "
       "the LAST BPE token of the counterfactual. `bug_audit/tok_strategy_counts.png` "
       "shows the single-vs-multi composition per cell.")
    ap("- Random-feature baselines computed against density-matched pools where "
       "applicable (Arabic-dual English sweep) or simple uniform sampling "
       "elsewhere (ablation validation).")
    ap("- **No cell claims are made without either a baseline comparison or an "
       "explicit null result.** See `TODO.md::scientific_anomalies` for cells "
       "whose internal statistics raise questions.")
    ap("- Template-supplement pairs (Tense, language-specific rare phenomena) "
       "were NOT generated this session. All multilingual data is Multi-BLiMP-derived.")
    ap("- Turkish Gender is absent by design, not a failed run.\n")

    ap("## Known gaps / next steps\n")
    ap("See `TODO.md` for the full list. Highlights:\n")
    ap("- Template supplement for Tense + language-specific rare phenomena.")
    ap("- Multi-token counterfactual with summed-logprob metric (this session: "
       "last-BPE approximation).")
    ap("- Example #1 (gender → sexist English) and #2 (formality → British "
       "spelling) both require English generation evaluation on crafted "
       "prompts — not attempted this session.")
    ap("- Syed-style two-pass attribution patching as comparison method.")
    ap("- UD POS/Feats profile per top feature (cut for scope).\n")

    ap("## Merge-back\n")
    ap("Work is on branch `overnight-multilingual` in worktree "
       "`/projectnb/mcnet/jbrin/lang-probing-overnight`. To adopt:\n")
    ap("```bash")
    ap("cd /projectnb/mcnet/jbrin/lang-probing")
    ap("git merge overnight-multilingual")
    ap("```")
    ap("To discard:")
    ap("```bash")
    ap("git worktree remove /projectnb/mcnet/jbrin/lang-probing-overnight")
    ap("git branch -D overnight-multilingual")
    ap("```")

    report_path = ROOT / "REPORT.md"
    report_path.write_text("\n".join(lines))
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
