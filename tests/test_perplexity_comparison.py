"""
Tests for perplexity comparison pipeline (correct vs wrong sentence, error rate).
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from datasets import Dataset

# Project root and src on path so we can import the relocated script module.
_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "src"))

import importlib.util as _importlib_util
_RUN_PER_PATH = os.path.join(_root, "experiments", "perplexity_bleu_linear", "run_per.py")
_spec = _importlib_util.spec_from_file_location("experiments_perplexity_run_per", _RUN_PER_PATH)
_run_per = _importlib_util.module_from_spec(_spec)
_spec.loader.exec_module(_run_per)

COL_CORRECT = _run_per.COL_CORRECT
COL_WRONG = _run_per.COL_WRONG
compute_error_rate = _run_per.compute_error_rate
perplexity_per_sentence = _run_per.perplexity_per_sentence
run_comparison = _run_per.run_comparison
run_comparison_multilang = _run_per.run_comparison_multilang
_slug = _run_per._slug


class TestComputeErrorRate(unittest.TestCase):
    """Pure logic tests for compute_error_rate (no model)."""

    def test_error_rate_half(self):
        # In 2 of 4 rows the model prefers wrong (ppl_wrong < ppl_correct)
        ppl_correct = np.array([10.0, 5.0, 3.0, 8.0])
        ppl_wrong = np.array([12.0, 4.0, 2.0, 9.0])  # row 1 and 2: wrong preferred
        rate, n_errors, n_total = compute_error_rate(ppl_correct, ppl_wrong)
        self.assertEqual(n_total, 4)
        self.assertEqual(n_errors, 2)
        self.assertAlmostEqual(rate, 0.5)

    def test_error_rate_zero(self):
        # Model always prefers correct (higher ppl on wrong)
        ppl_correct = np.array([1.0, 2.0, 3.0])
        ppl_wrong = np.array([5.0, 6.0, 7.0])
        rate, n_errors, n_total = compute_error_rate(ppl_correct, ppl_wrong)
        self.assertEqual(n_errors, 0)
        self.assertAlmostEqual(rate, 0.0)

    def test_error_rate_one(self):
        # Model always prefers wrong
        ppl_correct = np.array([5.0, 6.0])
        ppl_wrong = np.array([1.0, 2.0])
        rate, n_errors, n_total = compute_error_rate(ppl_correct, ppl_wrong)
        self.assertEqual(n_errors, 2)
        self.assertAlmostEqual(rate, 1.0)

    def test_error_rate_excludes_nan(self):
        # One row has NaN; only the other two count
        ppl_correct = np.array([10.0, np.nan, 3.0])
        ppl_wrong = np.array([12.0, np.nan, 2.0])  # row 2: wrong preferred
        rate, n_errors, n_total = compute_error_rate(ppl_correct, ppl_wrong)
        self.assertEqual(n_total, 2)
        self.assertEqual(n_errors, 1)
        self.assertAlmostEqual(rate, 0.5)

    def test_error_rate_all_nan_returns_nan(self):
        ppl_correct = np.array([np.nan, np.nan])
        ppl_wrong = np.array([np.nan, np.nan])
        rate, n_errors, n_total = compute_error_rate(ppl_correct, ppl_wrong)
        self.assertTrue(np.isnan(rate))
        self.assertEqual(n_errors, 0)
        self.assertEqual(n_total, 0)


class TestPerplexityPerSentence(unittest.TestCase):
    """Test perplexity_per_sentence shape and consistency (small model on CPU)."""

    def setUp(self):
        self.device = "cpu"
        self.model_id = "meta-llama/Llama-3.1-8B"  # Small, runs on CPU

    def test_output_length_matches_input(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.float32
        ).to(self.device)
        model.eval()

        texts = ["The cat sat.", "A dog ran.", "One more sentence here."]
        ppl = perplexity_per_sentence(
            model, tokenizer, texts, self.device, batch_size=2, max_length=32
        )
        self.assertEqual(len(ppl), len(texts))
        model.cpu()

    def test_values_positive_no_nan_for_nonempty(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.float32
        ).to(self.device)
        model.eval()

        texts = ["Hello world.", "Another sentence."]
        ppl = perplexity_per_sentence(
            model, tokenizer, texts, self.device, batch_size=2, max_length=32
        )
        self.assertEqual(len(ppl), 2)
        for i in range(len(ppl)):
            self.assertTrue(np.isfinite(ppl[i]), msg=f"ppl[{i}] should be finite")
            self.assertGreater(ppl[i], 0.0, msg=f"ppl[{i}] should be positive")


class TestRunComparisonIntegration(unittest.TestCase):
    """Integration test: run_comparison on tiny in-memory dataset (CPU)."""

    def test_run_comparison_creates_outputs_and_valid_error_rate(self):
        # Tiny in-memory dataset; pass via dataset= so we don't need HF hub
        in_memory_ds = Dataset.from_dict({
            "sen": ["The cat sat on the mat.", "A dog runs."],
            "wrong_sen": ["The cat sat on the mat", "A dog run."],
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            error_rate, dist_df = run_comparison(
                dataset_path="test_dataset",
                model_id="meta-llama/Llama-3.1-8B",
                col_correct=COL_CORRECT,
                col_wrong=COL_WRONG,
                batch_size=2,
                device="cpu",
                max_length=64,
                output_dir=tmpdir,
                dataset=in_memory_ds,
            )
            self.assertGreaterEqual(error_rate, 0.0)
            self.assertLessEqual(error_rate, 1.0)
            self.assertEqual(len(dist_df), 2)
            self.assertIn("ppl_sen", dist_df.columns)
            self.assertIn("ppl_wrong_sen", dist_df.columns)

            model_slug = _slug("meta-llama/Llama-3.1-8B")
            dataset_slug = _slug("test_dataset")
            error_csv = os.path.join(tmpdir, f"error_rate_{model_slug}_{dataset_slug}.csv")
            dist_csv = os.path.join(tmpdir, f"perplexity_distributions_{model_slug}_{dataset_slug}.csv")
            self.assertTrue(os.path.isfile(error_csv), f"Expected {error_csv}")
            self.assertTrue(os.path.isfile(dist_csv), f"Expected {dist_csv}")


class TestRunComparisonMultilang(unittest.TestCase):
    """Test run_comparison_multilang: (language x sample) matrices in NPZ and error rates in JSON."""

    def test_multilang_returns_matrices_and_error_rates_json(self):
        # Mock load_dataset to return a small dataset per language (no HF hub)
        tiny_ds = Dataset.from_dict({
            "sen": ["The cat sat.", "A dog runs."],
            "wrong_sen": ["The cat sitted.", "A dog run."],
        })

        def fake_load_dataset(path, name=None, split=None, **kwargs):
            return tiny_ds

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("perplexity_comparison.load_dataset", side_effect=fake_load_dataset):
                err_rates, ppl_sen_mat, ppl_wrong_mat, lang_codes, n_per_lang = (
                    run_comparison_multilang(
                        dataset_path="fake/dataset",
                        model_id="meta-llama/Llama-3.1-8B",
                        col_correct=COL_CORRECT,
                        col_wrong=COL_WRONG,
                        batch_size=2,
                        device="cpu",
                        max_length=64,
                        output_dir=tmpdir,
                        split="train",
                        language_codes=["eng", "spa"],
                    )
                )
            self.assertEqual(set(err_rates.keys()), {"eng", "spa"})
            for lang in ["eng", "spa"]:
                self.assertIn(lang, err_rates)
                self.assertGreaterEqual(err_rates[lang], 0.0)
                self.assertLessEqual(err_rates[lang], 1.0)
            # (n_languages, max_samples)
            self.assertEqual(ppl_sen_mat.shape[0], 2)
            self.assertEqual(ppl_sen_mat.shape[1], 2)
            self.assertEqual(ppl_wrong_mat.shape, (2, 2))
            self.assertEqual(lang_codes, ["eng", "spa"])
            self.assertEqual(list(n_per_lang), [2, 2])

            # NPZ exists and contains expected arrays
            model_slug = _slug("meta-llama/Llama-3.1-8B")
            dataset_slug = _slug("fake/dataset")
            npz_path = os.path.join(tmpdir, f"perplexity_matrices_{model_slug}_{dataset_slug}.npz")
            self.assertTrue(os.path.isfile(npz_path), npz_path)
            with np.load(npz_path, allow_pickle=True) as data:
                self.assertEqual(data["ppl_sen"].shape, (2, 2))
                self.assertEqual(data["ppl_wrong_sen"].shape, (2, 2))
                self.assertEqual(list(data["language_codes"]), ["eng", "spa"])
                np.testing.assert_array_equal(data["n_samples_per_language"], [2, 2])

            # JSON exists and has language -> error rate
            import json
            json_path = os.path.join(tmpdir, f"error_rates_by_language_{model_slug}_{dataset_slug}.json")
            self.assertTrue(os.path.isfile(json_path), json_path)
            with open(json_path) as f:
                j = json.load(f)
            self.assertEqual(set(j.keys()), {"eng", "spa"})
            for v in j.values():
                self.assertIsInstance(v, (int, float))
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)


if __name__ == "__main__":
    unittest.main()
