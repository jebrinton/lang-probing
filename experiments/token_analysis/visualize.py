"""
Stage 2: HTML visualization for per-token SAE ablation analysis.

Reads cached JSON files produced by analyze_tokens.py and generates
self-contained HTML dashboards (inline CSS, no JS dependencies).

Usage:
    python scripts/visualize_tokens.py --input_dir outputs/token_analysis/ --output_dir outputs/token_analysis/html/
    python scripts/visualize_tokens.py --input outputs/token_analysis/foo.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _orange_intensity(norm: float) -> str:
    """Return a CSS background-color for SAE activation intensity (0..1)."""
    norm = _clamp(norm, 0.0, 1.0)
    # White -> deep orange
    r = int(255)
    g = int(255 - norm * 165)
    b = int(255 - norm * 255)
    return f"rgb({r},{g},{b})"


def _logprob_delta_color(norm: float) -> str:
    """Return a CSS background-color for logprob delta (−1..+1 normalised).

    norm < 0  → red  (probability decreased)
    norm > 0  → blue (probability increased)
    """
    norm = _clamp(norm, -1.0, 1.0)
    intensity = abs(norm)
    if norm < 0:
        r = int(255)
        g = int(255 - intensity * 160)
        b = int(255 - intensity * 160)
    else:
        r = int(255 - intensity * 160)
        g = int(255 - intensity * 160)
        b = int(255)
    return f"rgb({r},{g},{b})"


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

_CSS = """
body { font-family: monospace; font-size: 14px; margin: 20px; background: #fafafa; }
h1 { font-size: 18px; margin-bottom: 6px; }
.meta-table { border-collapse: collapse; margin-bottom: 12px; }
.meta-table td, .meta-table th { border: 1px solid #ccc; padding: 4px 10px; }
.meta-table th { background: #eee; }
.panels { display: flex; gap: 24px; margin-bottom: 20px; flex-wrap: wrap; }
.panel { border: 1px solid #ccc; padding: 12px; border-radius: 4px; background: #fff; min-width: 300px; }
.panel h2 { font-size: 14px; margin: 0 0 10px; }
.tokens { display: flex; flex-wrap: wrap; gap: 6px; }
.tok {
    display: inline-block;
    padding: 4px 6px;
    border-radius: 3px;
    cursor: default;
    min-width: 30px;
    text-align: center;
    position: relative;
}
.tok.ablated { text-decoration: underline; text-decoration-thickness: 2px; }
.tok.special { color: #999; font-size: 11px; }
.tok .delta-val {
    display: block;
    font-size: 10px;
    color: #333;
    margin-top: 2px;
    font-weight: normal;
}
.detail-table { border-collapse: collapse; width: 100%; font-size: 12px; }
.detail-table th, .detail-table td { border: 1px solid #ddd; padding: 4px 8px; text-align: left; }
.detail-table th { background: #eee; }
.detail-table tr:nth-child(even) { background: #f9f9f9; }
.ablated-row { background: #fff3e0 !important; }
"""

_SPECIAL_TOKENS = {"<|begin_of_text|>", "<|end_of_text|>", "<pad>", "</s>", "<s>"}


def _is_special(text: str) -> bool:
    return text.strip() in _SPECIAL_TOKENS or (text.startswith("<|") and text.endswith("|>"))


def _fmt(v: Optional[float], decimals: int = 3) -> str:
    if v is None:
        return "—"
    return f"{v:+.{decimals}f}"


def _build_html(data: dict) -> str:
    meta = data["metadata"]
    tokens = data["tokens"]
    masks = data["masks"]
    sae = data["sae_activations"]
    logprobs = data["logprobs"]

    S = len(tokens)
    feature_ids = sae["feature_ids"]
    sae_vals = sae["values"]  # [S, K]

    ablate_mask = masks["ablate_mask"]
    prob_mask = masks["prob_mask"]

    orig = logprobs["original"]       # [S], position 0 is null
    interv = logprobs["intervention"]
    delta = logprobs["delta"]

    # Normalise SAE activations for color
    sae_sums = [sum(sae_vals[i]) for i in range(S)]
    max_sae = max(sae_sums) if max(sae_sums) > 0 else 1.0

    # Normalise logprob deltas for color
    valid_deltas = [d for d in delta if d is not None]
    max_abs_delta = max(abs(d) for d in valid_deltas) if valid_deltas else 1.0
    if max_abs_delta == 0.0:
        max_abs_delta = 1.0

    # ---- Header / metadata ----
    target_lang = meta.get("target_lang") or "—"
    rows_html = "".join(
        f"<tr><th>{k}</th><td>{v}</td></tr>"
        for k, v in [
            ("concept", meta.get("concept", "—")),
            ("value", meta.get("value", "—")),
            ("k", meta.get("k", "—")),
            ("source_lang", meta.get("source_lang", "—")),
            ("target_lang", target_lang),
            ("feats", meta.get("feats", "—")),
            ("ablate_loc", meta.get("ablate_loc", "—")),
            ("prob_loc", meta.get("prob_loc", "—")),
            ("use_probe", meta.get("use_probe", False)),
        ]
    )
    feat_ids_str = ", ".join(str(f) for f in feature_ids)

    header_html = f"""
<h1>{meta.get('name', 'Experiment')}</h1>
<table class="meta-table">{rows_html}</table>
<p><strong>Feature IDs:</strong> [{feat_ids_str}]</p>
"""

    # ---- SAE panel ----
    sae_toks = []
    for i, tok in enumerate(tokens):
        text = tok["text"]
        is_special = _is_special(text)
        css = "tok special" if is_special else "tok"
        if ablate_mask[i]:
            css += " ablated"
        bg = _orange_intensity(sae_sums[i] / max_sae)
        per_feat = ", ".join(f"feat {feature_ids[j]}: {sae_vals[i][j]:.3f}" for j in range(len(feature_ids)))
        title = f"pos {i} | sum={sae_sums[i]:.3f} | {per_feat}"
        sae_toks.append(
            f'<span class="{css}" style="background:{bg}" title="{title}">{text}</span>'
        )

    sae_panel = f"""
<div class="panel">
  <h2>SAE Feature Activations</h2>
  <div class="tokens">{"".join(sae_toks)}</div>
  <p style="font-size:11px;margin-top:8px;color:#555">
    Orange intensity = sum of concept-feature activations &nbsp;|&nbsp; Underline = ablated
  </p>
</div>
"""

    # ---- Logprob delta panel ----
    lp_toks = []
    for i, tok in enumerate(tokens):
        text = tok["text"]
        is_special = _is_special(text)
        css = "tok special" if is_special else "tok"
        if ablate_mask[i]:
            css += " ablated"

        d = delta[i]
        if d is not None:
            norm = d / max_abs_delta
            bg = _logprob_delta_color(norm)
            delta_label = f'<span class="delta-val">{_fmt(d)}</span>'
        else:
            bg = "#f0f0f0"
            delta_label = '<span class="delta-val">—</span>'

        # Bold tokens in the logprob measurement sequence (prob_mask)
        inner_text = f"<strong>{text}</strong>" if prob_mask[i] else text

        title = (
            f"pos {i} | orig={_fmt(orig[i])} | interv={_fmt(interv[i])} | delta={_fmt(d)}"
        )
        lp_toks.append(
            f'<span class="{css}" style="background:{bg}" title="{title}">'
            f"{inner_text}{delta_label}"
            f"</span>"
        )

    lp_panel = f"""
<div class="panel">
  <h2>Logprob Delta After Ablation</h2>
  <div class="tokens">{"".join(lp_toks)}</div>
  <p style="font-size:11px;margin-top:8px;color:#555">
    Red = prob decreased &nbsp;|&nbsp; Blue = prob increased &nbsp;|&nbsp;
    Underline = ablated &nbsp;|&nbsp; <strong>Bold</strong> = measured (prob_mask) &nbsp;|&nbsp;
    Value = logprob delta
  </p>
</div>
"""

    # ---- Detail table ----
    detail_rows = []
    for i, tok in enumerate(tokens):
        row_css = "ablated-row" if ablate_mask[i] else ""
        detail_rows.append(
            f"<tr class=\"{row_css}\">"
            f"<td>{i}</td>"
            f"<td>{tok['text']}</td>"
            f"<td>{'✓' if ablate_mask[i] else ''}</td>"
            f"<td>{'✓' if prob_mask[i] else ''}</td>"
            f"<td>{sae_sums[i]:.4f}</td>"
            f"<td>{' | '.join(f'{feature_ids[j]}:{sae_vals[i][j]:.3f}' for j in range(len(feature_ids)))}</td>"
            f"<td>{_fmt(orig[i])}</td>"
            f"<td>{_fmt(interv[i])}</td>"
            f"<td>{_fmt(delta[i])}</td>"
            "</tr>"
        )

    detail_table = f"""
<h2 style="font-size:14px">Per-Token Detail</h2>
<table class="detail-table">
  <thead>
    <tr>
      <th>Pos</th><th>Token</th><th>Ablated</th><th>Measured</th>
      <th>SAE Sum</th><th>Features</th>
      <th>LogP Orig</th><th>LogP Interv</th><th>Delta</th>
    </tr>
  </thead>
  <tbody>{"".join(detail_rows)}</tbody>
</table>
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{meta.get('name', 'Token Analysis')}</title>
  <style>{_CSS}</style>
</head>
<body>
{header_html}
<div class="panels">
{sae_panel}
{lp_panel}
</div>
{detail_table}
</body>
</html>
"""


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def visualize_file(input_path: Path, output_path: Path) -> None:
    with open(input_path) as f:
        data = json.load(f)
    html = _build_html(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"  {input_path.name} → {output_path}")


def main(args: argparse.Namespace) -> None:
    if args.input:
        input_path = Path(args.input)
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.with_suffix(".html")
        visualize_file(input_path, output_path)
    else:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        json_files = sorted(input_dir.glob("*.json"))
        if not json_files:
            print(f"No JSON files found in {input_dir}")
            return
        for p in json_files:
            visualize_file(p, output_dir / p.with_suffix(".html").name)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize per-token SAE ablation data as HTML")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--input", help="Single JSON input file")
    grp.add_argument("--input_dir", help="Directory of JSON input files")
    parser.add_argument("--output", help="Single HTML output file (used with --input)")
    parser.add_argument("--output_dir", help="Output directory for HTML files (used with --input_dir)")
    args = parser.parse_args()

    if args.input_dir and not args.output_dir:
        parser.error("--output_dir is required when using --input_dir")

    main(args)
