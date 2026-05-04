"""
Build manifest.json + standalone HTML fragments for one harness run.

The dashboard.html template polls manifest.json every couple of seconds; new
panels appear without a page reload. Each panel is rendered to a self-
contained HTML fragment so the dashboard can show one at a time (stepper)
or all at once (grid).
"""
from __future__ import annotations

import html
import json
import time
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------


def _signed_color(value: float, max_abs: float) -> str:
    """Diverging teal (-) → white (0) → orange (+).
    Returns rgba string with alpha = |value| / max_abs."""
    if max_abs <= 0:
        return "rgba(255, 255, 255, 0)"
    a = max(0.0, min(1.0, abs(value) / max_abs))
    if value >= 0:
        # warm orange
        r, g, b = 240, 140, 60
    else:
        # cool teal
        r, g, b = 60, 170, 200
    return f"rgba({r}, {g}, {b}, {a:.3f})"


def _positive_color(value: float, max_abs: float) -> str:
    """Single-hue (purple) for non-negative scalars (attention, SAE acts)."""
    if max_abs <= 0:
        return "rgba(255, 255, 255, 0)"
    a = max(0.0, min(1.0, abs(value) / max_abs))
    return f"rgba(120, 90, 200, {a:.3f})"


# ---------------------------------------------------------------------------
# Token strip
# ---------------------------------------------------------------------------


def render_token_strip(strip_data: List[Dict], *, signed: bool = False,
                       title: Optional[str] = None) -> str:
    """Return an HTML fragment with one <span> per token, bg ∝ value."""
    if not strip_data:
        return ""
    max_abs = max(abs(d["value"]) for d in strip_data) or 1.0
    spans = []
    for d in strip_data:
        tok_html = html.escape(d["token"]).replace("\n", "↵\n")
        color = _signed_color(d["value"], max_abs) if signed else _positive_color(d["value"], max_abs)
        spans.append(
            f'<span class="tok" style="background:{color}" '
            f'title="{html.escape(repr(d["token"]))}  v={d["value"]:.4f}">{tok_html}</span>'
        )
    head = f'<div class="strip-title">{html.escape(title)}</div>' if title else ""
    legend = (
        f'<div class="strip-legend">max |v| = {max_abs:.4f}'
        f' &nbsp;|&nbsp; {"signed (teal − / orange +)" if signed else "magnitude (purple)"}</div>'
    )
    return (
        f'<div class="strip-block">{head}'
        f'<div class="strip">{"".join(spans)}</div>'
        f'{legend}</div>'
    )


# ---------------------------------------------------------------------------
# Logit-lens table
# ---------------------------------------------------------------------------


def render_logit_lens(ll_data: Dict) -> str:
    """ll_data: {<anchor>: {"position_token": str, "topk_with_norm": [...],
                            "topk_no_norm": [...]}}"""
    if not ll_data:
        return ""
    blocks = []
    for anchor, d in ll_data.items():
        rows_with = "".join(
            f'<tr><td>{html.escape(t["token"])}</td>'
            f'<td class="num">{t["prob"]:.3f}</td></tr>'
            for t in d["topk_with_norm"]
        )
        rows_no = "".join(
            f'<tr><td>{html.escape(t["token"])}</td>'
            f'<td class="num">{t["prob"]:.3f}</td></tr>'
            for t in d["topk_no_norm"]
        )
        blocks.append(f"""
<div class="ll-anchor">
  <div class="ll-title">logit lens @ <code>{html.escape(anchor)}</code>
    (token = <code>{html.escape(repr(d["position_token"]))}</code>)</div>
  <div class="ll-pair">
    <div class="ll-col">
      <div class="ll-sub">with final norm</div>
      <table class="ll-table"><thead><tr><th>token</th><th>prob</th></tr></thead>
      <tbody>{rows_with}</tbody></table>
    </div>
    <div class="ll-col">
      <div class="ll-sub">no norm (raw projection)</div>
      <table class="ll-table"><thead><tr><th>token</th><th>prob</th></tr></thead>
      <tbody>{rows_no}</tbody></table>
    </div>
  </div>
</div>
""")
    return f'<div class="ll-block">{"".join(blocks)}</div>'


# ---------------------------------------------------------------------------
# Panel renderers
# ---------------------------------------------------------------------------


def render_observe_panel(panel: Dict) -> str:
    """One panel from observe(): SAE feature OR attention head."""
    title = html.escape(panel["title"])
    subtitle = html.escape(panel.get("subtitle", ""))
    body = ""
    if panel["kind"] == "sae":
        body = render_token_strip(panel["strip"], signed=False,
                                  title="per-token activation")
    elif panel["kind"] == "head":
        strips_html = []
        # Stable mode order: iii (default) first
        for mode in ["iii", "i", "ii"]:
            if mode in panel["strips"]:
                s = panel["strips"][mode]
                strips_html.append(
                    render_token_strip(
                        s["data"], signed=False,
                        title=f"{s['label']} — axis: {s['axis']}",
                    )
                )
        body = "".join(strips_html) + render_logit_lens(panel.get("logit_lens", {}))
    return f"""
<section class="panel" data-kind="{panel['kind']}" data-id="{panel['id']}">
  <header class="panel-header">
    <h2>{title}</h2>
    <div class="panel-sub">{subtitle}</div>
  </header>
  <div class="panel-body">{body}</div>
</section>
"""


def render_intervene_panel(panel: Dict) -> str:
    """One intervene() result: baseline vs intervened side-by-side."""
    s = panel["sample"]
    op = panel["op"]
    delta = panel.get("gold_logp_delta")
    delta_str = "—" if delta is None else f"{delta:+.3f} nats"
    base_lp = panel.get("gold_logp_baseline")
    intv_lp = panel.get("gold_logp_intervened")
    base_lp_str = "—" if base_lp is None else f"{base_lp:.3f}"
    intv_lp_str = "—" if intv_lp is None else f"{intv_lp:.3f}"
    return f"""
<section class="panel intervene" data-kind="intervene" data-id="{html.escape(s['name'] + '__' + op['label'])}">
  <header class="panel-header">
    <h2>{html.escape(s['name'])} &nbsp;·&nbsp; <span class="op">{html.escape(op['label'])}</span></h2>
    <div class="panel-sub">{html.escape(s['src_lang'])} → {html.escape(s['tgt_lang'])}</div>
  </header>
  <div class="panel-body">
    <div class="kv"><span class="k">prompt source:</span> <span class="v">{html.escape(s['src'])}</span></div>
    <div class="kv"><span class="k">gold target:</span> <span class="v">{html.escape(s['tgt']) or '—'}</span></div>
    <div class="kv"><span class="k">log p(gold):</span>
      <span class="v">baseline {base_lp_str} &nbsp;→&nbsp; intervened {intv_lp_str}
        &nbsp;(<strong>Δ = {delta_str}</strong>)</span></div>
    <div class="generations">
      <div class="gen">
        <div class="gen-title">baseline output</div>
        <pre class="gen-text">{html.escape(panel['baseline_text'])}</pre>
      </div>
      <div class="gen">
        <div class="gen-title">intervened output</div>
        <pre class="gen-text intervened">{html.escape(panel['intervened_text'])}</pre>
      </div>
    </div>
  </div>
</section>
"""


# ---------------------------------------------------------------------------
# Top-level: render an observe() / intervene() result to a run directory
# ---------------------------------------------------------------------------


def render_observe_result(result: Dict, run_dir: Path) -> List[Dict]:
    """Write one HTML fragment per panel, return the manifest entries."""
    run_dir.mkdir(parents=True, exist_ok=True)
    sample_header = render_sample_header(result)
    entries = []
    for i, panel in enumerate(result["panels"]):
        frag = sample_header + render_observe_panel(panel)
        fname = f"observe_{result['sample']['name']}__{panel['kind']}_{panel['id'].replace('.', '_')}.html"
        (run_dir / fname).write_text(frag)
        entries.append({
            "kind": panel["kind"],
            "id": panel["id"],
            "title": panel["title"],
            "sample": result["sample"]["name"],
            "src_lang": result["sample"]["src_lang"],
            "tgt_lang": result["sample"]["tgt_lang"],
            "fragment": fname,
            "mode": "observe",
            "ts": time.time(),
        })
    return entries


def render_intervene_result(result: Dict, run_dir: Path) -> List[Dict]:
    run_dir.mkdir(parents=True, exist_ok=True)
    frag = render_intervene_panel(result)
    fname = f"intervene_{result['sample']['name']}__{result['op']['kind']}_{(result['op'].get('feature_idx') or str(result['op'].get('layer'))+'_'+str(result['op'].get('head')))}.html"
    fname = fname.replace("/", "_")
    (run_dir / fname).write_text(frag)
    return [{
        "kind": "intervene",
        "id": result["op"]["label"],
        "title": result["sample"]["name"] + " · " + result["op"]["label"],
        "sample": result["sample"]["name"],
        "src_lang": result["sample"]["src_lang"],
        "tgt_lang": result["sample"]["tgt_lang"],
        "fragment": fname,
        "mode": "intervene",
        "ts": time.time(),
    }]


def render_sample_header(result: Dict) -> str:
    s = result["sample"]
    tokens_summary = " ".join(html.escape(t) for t in result["tokens"][:result["last_src_idx"] + 1][-12:])
    return f"""
<aside class="sample-header">
  <div><strong>{html.escape(s['name'])}</strong> &nbsp;·&nbsp;
       {html.escape(s['src_lang'])} → {html.escape(s['tgt_lang'])}
       &nbsp;·&nbsp; prompt_len={result['prompt_len']}, last_src_idx={result['last_src_idx']}</div>
  <div class="sample-src">SRC: {html.escape(s['src'])}</div>
  <div class="sample-tgt">TGT: {html.escape(s['tgt']) or '—'}</div>
  <div class="sample-tail">…last 12 prompt tokens: <code>{tokens_summary}</code></div>
</aside>
"""


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def write_manifest(out_dir: Path, entries: List[Dict], run_label: Optional[str] = None) -> None:
    """Write/append a manifest.json that the dashboard polls."""
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text()).get("entries", [])
    else:
        existing = []
    manifest = {
        "version": 1,
        "updated_at": time.time(),
        "run_label": run_label,
        "entries": existing + entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))


def reset_manifest(out_dir: Path) -> None:
    """Wipe the manifest (e.g. on REPL `clear`)."""
    (out_dir / "manifest.json").write_text(json.dumps({
        "version": 1,
        "updated_at": time.time(),
        "entries": [],
    }, indent=2))
