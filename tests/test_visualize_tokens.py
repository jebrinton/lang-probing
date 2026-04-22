"""
Tests for scripts/visualize_tokens.py — pure Python, no GPU required.

Run with:
    pytest tests/test_visualize_tokens.py
"""

import sys
import os
import json
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from scripts.visualize_tokens import _build_html, _orange_intensity, _logprob_delta_color, visualize_file


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_cache(
    name="test_exp",
    tokens=None,
    ablate_mask=None,
    prob_mask=None,
    sae_values=None,
    delta=None,
):
    """Build a minimal valid cache dict."""
    if tokens is None:
        tokens = [
            {"id": 1, "text": "<|begin_of_text|>", "position": 0},
            {"id": 100, "text": "El",   "position": 1},
            {"id": 200, "text": " gato","position": 2},
            {"id": 300, "text": " comió","position": 3},
        ]
    S = len(tokens)
    K = 3
    feature_ids = [10, 20, 30]

    if ablate_mask is None:
        ablate_mask = [False, True, True, False]
    if prob_mask is None:
        prob_mask = [False, True, True, True]
    if sae_values is None:
        sae_values = [[0.0, 0.0, 0.0], [0.5, 0.1, 0.0], [0.3, 0.4, 0.2], [0.0, 0.0, 0.1]]
    if delta is None:
        delta = [None, -0.24, -0.74, 0.10]

    orig =  [None, -3.21, -2.15, -1.80]
    interv = [None if d is None else o + d for o, d in zip(orig, delta)]

    return {
        "metadata": {
            "name": name,
            "source_lang": "Spanish",
            "target_lang": None,
            "concept": "Tense",
            "value": "Past",
            "k": K,
            "feats": "input",
            "ablate_loc": "source",
            "prob_loc": "source",
            "use_probe": False,
            "feature_indices": feature_ids,
        },
        "tokens": tokens,
        "masks": {
            "source_mask": [True] * S,
            "target_mask": [False] * S,
            "ablate_mask": ablate_mask,
            "prob_mask": prob_mask,
        },
        "sae_activations": {
            "feature_ids": feature_ids,
            "values": sae_values,
        },
        "logprobs": {
            "original": orig,
            "intervention": interv,
            "delta": delta,
        },
    }


# ---------------------------------------------------------------------------
# Color helper tests (no file I/O needed)
# ---------------------------------------------------------------------------

class TestColorHelpers:
    def test_orange_zero(self):
        color = _orange_intensity(0.0)
        assert color == "rgb(255,255,255)"

    def test_orange_one(self):
        color = _orange_intensity(1.0)
        assert color == "rgb(255,90,0)"

    def test_orange_clamps_below_zero(self):
        assert _orange_intensity(-0.5) == _orange_intensity(0.0)

    def test_orange_clamps_above_one(self):
        assert _orange_intensity(1.5) == _orange_intensity(1.0)

    def test_delta_negative_is_red(self):
        color = _logprob_delta_color(-1.0)
        r, g, b = [int(x) for x in color[4:-1].split(",")]
        assert r == 255
        assert g < 200
        assert b < 200

    def test_delta_positive_is_blue(self):
        color = _logprob_delta_color(1.0)
        r, g, b = [int(x) for x in color[4:-1].split(",")]
        assert b == 255
        assert r < 200
        assert g < 200

    def test_delta_zero_is_white(self):
        color = _logprob_delta_color(0.0)
        assert color == "rgb(255,255,255)"


# ---------------------------------------------------------------------------
# HTML structure tests
# ---------------------------------------------------------------------------

class TestBuildHtml:
    def test_returns_string(self):
        html = _build_html(_make_cache())
        assert isinstance(html, str)

    def test_contains_experiment_name(self):
        html = _build_html(_make_cache(name="my_experiment"))
        assert "my_experiment" in html

    def test_contains_feature_ids(self):
        html = _build_html(_make_cache())
        assert "10" in html
        assert "20" in html
        assert "30" in html

    def test_contains_token_texts(self):
        html = _build_html(_make_cache())
        assert "El" in html
        assert "gato" in html
        assert "comió" in html

    def test_ablated_tokens_have_underline_class(self):
        html = _build_html(_make_cache())
        # ablate_mask = [False, True, True, False] → "El" and " gato" ablated
        # Each ablated token should have class "ablated"
        assert html.count("ablated") >= 2

    # --- logprob measurement sequence bolding ---

    def test_prob_mask_tokens_are_bold(self):
        """Tokens in prob_mask should be wrapped in <strong>."""
        data = _make_cache()
        # prob_mask = [False, True, True, True] → "El", " gato", " comió" bolded
        html = _build_html(data)
        # Check that the logprob panel wraps measured tokens in <strong>
        assert "<strong>El</strong>" in html
        assert "<strong> gato</strong>" in html
        assert "<strong> comió</strong>" in html

    def test_non_prob_mask_token_not_bold(self):
        """Special/first token (prob_mask=False) should NOT be wrapped in <strong>."""
        data = _make_cache()
        # pos 0 is <|begin_of_text|> with prob_mask=False
        html = _build_html(data)
        assert "<strong><|begin_of_text|></strong>" not in html

    def test_all_prob_mask_false_no_bold(self):
        prob_mask = [False, False, False, False]
        data = _make_cache(prob_mask=prob_mask)
        html = _build_html(data)
        assert "<strong>El</strong>" not in html
        assert "<strong> gato</strong>" not in html

    # --- logprob delta values reported ---

    def test_delta_values_present_in_html(self):
        """Numeric logprob delta values should appear in the logprob panel."""
        data = _make_cache(delta=[None, -0.24, -0.74, 0.10])
        html = _build_html(data)
        assert "-0.240" in html
        assert "-0.740" in html
        assert "+0.100" in html

    def test_null_delta_shows_dash(self):
        """Position 0 has delta=None and should show '—'."""
        html = _build_html(_make_cache())
        assert "—" in html

    def test_delta_val_span_present(self):
        """Each token cell in logprob panel should include a delta-val span."""
        html = _build_html(_make_cache())
        assert 'class="delta-val"' in html

    # --- detail table ---

    def test_detail_table_present(self):
        html = _build_html(_make_cache())
        assert "Per-Token Detail" in html
        assert "<table" in html

    def test_detail_table_has_measured_column(self):
        html = _build_html(_make_cache())
        assert "Measured" in html

    def test_detail_table_rows_match_token_count(self):
        html = _build_html(_make_cache())
        # 4 tokens → 4 <tr> in tbody (plus 1 header row)
        assert html.count("<tr") >= 5

    # --- metadata ---

    def test_metadata_rendered(self):
        html = _build_html(_make_cache())
        assert "Tense" in html
        assert "Past" in html
        assert "Spanish" in html

    # --- edge cases ---

    def test_all_positive_deltas(self):
        data = _make_cache(delta=[None, 0.10, 0.20, 0.30])
        html = _build_html(data)
        assert "+0.100" in html
        assert "+0.200" in html

    def test_all_zero_deltas(self):
        data = _make_cache(delta=[None, 0.0, 0.0, 0.0])
        html = _build_html(data)
        assert "+0.000" in html

    def test_single_token(self):
        tokens = [{"id": 1, "text": "hi", "position": 0}]
        data = _make_cache(
            tokens=tokens,
            ablate_mask=[True],
            prob_mask=[True],
            sae_values=[[0.5, 0.0, 0.0]],
            delta=[None],
        )
        html = _build_html(data)
        assert "hi" in html

    def test_special_token_gets_special_class(self):
        html = _build_html(_make_cache())
        # <|begin_of_text|> should have "special" class
        assert "special" in html


# ---------------------------------------------------------------------------
# File I/O tests
# ---------------------------------------------------------------------------

class TestVisualizeFile:
    def test_writes_html_file(self):
        data = _make_cache()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "test.json"
            output_path = Path(tmpdir) / "test.html"
            with open(input_path, "w") as f:
                json.dump(data, f)
            visualize_file(input_path, output_path)
            assert output_path.exists()

    def test_output_is_valid_html(self):
        data = _make_cache()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "test.json"
            output_path = Path(tmpdir) / "test.html"
            with open(input_path, "w") as f:
                json.dump(data, f)
            visualize_file(input_path, output_path)
            content = output_path.read_text()
            assert content.startswith("<!DOCTYPE html>")
            assert "</html>" in content

    def test_output_contains_prob_mask_bold(self):
        data = _make_cache()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "test.json"
            output_path = Path(tmpdir) / "test.html"
            with open(input_path, "w") as f:
                json.dump(data, f)
            visualize_file(input_path, output_path)
            content = output_path.read_text()
            assert "<strong>El</strong>" in content

    def test_output_contains_delta_values(self):
        data = _make_cache(delta=[None, -0.24, -0.74, 0.10])
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "test.json"
            output_path = Path(tmpdir) / "test.html"
            with open(input_path, "w") as f:
                json.dump(data, f)
            visualize_file(input_path, output_path)
            content = output_path.read_text()
            assert "-0.240" in content

    def test_creates_output_dir_if_missing(self):
        data = _make_cache()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "test.json"
            output_path = Path(tmpdir) / "subdir" / "test.html"
            with open(input_path, "w") as f:
                json.dump(data, f)
            visualize_file(input_path, output_path)
            assert output_path.exists()
