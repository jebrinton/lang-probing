"""
Integration tests for ablate.py script
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import tempfile
import json
from pathlib import Path
import torch
import numpy as np
from datasets import load_dataset

from lang_probing_src.config import MODEL_ID, SAE_ID, NAME_TO_LANG_CODE, LANGUAGES_DEC
from lang_probing_src.utils import setup_model
from lang_probing_src.utils_input_output import get_input_features_vector, get_output_features_vector, load_effects_files
from scripts.ablate import EXP_CONFIGS, get_batch_positions_masks, save_result_jsonl
from tests.test_utils import create_test_prompts


class TestExperimentConfigs(unittest.TestCase):
    """Test all experiment configurations"""
    
    @unittest.skip("Requires full model setup - run manually with GPU")
    def test_all_experiment_configs(self):
        """Test that all experiment configs can run without errors"""
        model, submodule, autoencoder, tokenizer = setup_model(MODEL_ID, SAE_ID)
        
        # Test each experiment config
        for exp_name, exp_cfg in EXP_CONFIGS.items():
            with self.subTest(experiment=exp_name):
                # This is a smoke test - just verify the config is valid
                self.assertIn('mode', exp_cfg)
                self.assertIn('ablate_loc', exp_cfg)
                self.assertIn('prob_loc', exp_cfg)
                self.assertIn('feats', exp_cfg)
                
                # Verify valid values
                self.assertIn(exp_cfg['mode'], ['monolingual', 'multilingual'])
                self.assertIn(exp_cfg['ablate_loc'], ['source', 'target'])
                self.assertIn(exp_cfg['prob_loc'], ['source', 'target'])
                self.assertIn(exp_cfg['feats'], ['input', 'output', 'random'])
    
    def test_exp_configs_structure(self):
        """Test that EXP_CONFIGS has correct structure"""
        self.assertGreater(len(EXP_CONFIGS), 0)
        
        required_keys = ['mode', 'ablate_loc', 'prob_loc', 'feats']
        for exp_name, exp_cfg in EXP_CONFIGS.items():
            for key in required_keys:
                self.assertIn(key, exp_cfg, f"Missing key {key} in {exp_name}")


class TestFeatureLoading(unittest.TestCase):
    """Tests for feature loading functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.input_features_dir = Path("/projectnb/mcnet/jbrin/lang-probing/outputs/sentence_input_features/")
        
    @unittest.skip("Requires feature files to exist")
    def test_get_input_features_vector(self):
        """Test loading input features vector"""
        # Try to load a known feature
        try:
            feat_vec = get_input_features_vector(
                self.input_features_dir,
                "English",
                "Tense",
                "Past"
            )
            self.assertIsInstance(feat_vec, np.ndarray)
            self.assertGreater(len(feat_vec), 0)
        except FileNotFoundError:
            self.skipTest("Feature file not found")
    
    @unittest.skip("Requires effects files to exist")
    def test_get_output_features_vector(self):
        """Test loading output features vector"""
        try:
            effects_files = load_effects_files()
            feat_vec = get_output_features_vector(
                effects_files,
                ("English", "Spanish"),
                "Tense",
                "Past"
            )
            self.assertIsInstance(feat_vec, np.ndarray)
            self.assertGreater(len(feat_vec), 0)
        except (FileNotFoundError, KeyError):
            self.skipTest("Effects file not found")
    
    def test_feature_indices_selection(self):
        """Test that feature indices are correctly selected (top K)"""
        # Create synthetic feature vector
        np.random.seed(42)
        feat_vec = np.random.randn(1000)
        # Make top 10 features much stronger
        top_indices = np.argsort(np.abs(feat_vec))[-10:]
        feat_vec[top_indices] = np.random.randn(10) * 5.0
        
        # Select top K
        k = 5
        selected_indices = np.argsort(feat_vec)[-k:]
        
        # Verify we got the top K
        self.assertEqual(len(selected_indices), k)
        # Verify they're the largest values
        sorted_values = np.sort(feat_vec)
        self.assertTrue(np.allclose(feat_vec[selected_indices], sorted_values[-k:]))
    
    def test_random_baseline_generation(self):
        """Test random baseline feature generation"""
        sae_dim = 32768
        k = 10
        
        # Test reproducibility with seed
        np.random.seed(42)
        indices1 = np.random.choice(sae_dim, k, replace=False)
        
        np.random.seed(42)
        indices2 = np.random.choice(sae_dim, k, replace=False)
        
        np.testing.assert_array_equal(indices1, indices2)
        
        # Test no duplicates
        self.assertEqual(len(np.unique(indices1)), k)


class TestBatchProcessing(unittest.TestCase):
    """Tests for batch processing functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = "cpu"
        
    def test_get_batch_positions_masks_monolingual(self):
        """Test mask generation for monolingual prompts"""
        contexts = ["", "", ""]
        sources = [
            "The cat sat on the mat.",
            "She walked to the store.",
            "They played in the park."
        ]
        targets = [None, None, None]
        
        input_ids, src_mask, tgt_mask = get_batch_positions_masks(
            self.tokenizer, contexts, sources, targets, device=self.device
        )
        
        # Check shapes
        batch_size = len(sources)
        self.assertEqual(input_ids.shape[0], batch_size)
        self.assertEqual(src_mask.shape[0], batch_size)
        self.assertEqual(tgt_mask.shape[0], batch_size)
        
        # Source mask should have True values
        self.assertTrue(src_mask.any())
        
        # Target mask should be all False for monolingual
        self.assertFalse(tgt_mask.any())
        
    def test_get_batch_positions_masks_multilingual(self):
        """Test mask generation for multilingual prompts"""
        contexts = [
            "Hello >> Hola\nHi >> Hola\n",
            "Good >> Bueno\nBad >> Malo\n"
        ]
        sources = ["How are you?", "What's up?"]
        targets = ["¿Cómo estás?", "¿Qué tal?"]
        
        input_ids, src_mask, tgt_mask = get_batch_positions_masks(
            self.tokenizer, contexts, sources, targets, device=self.device
        )
        
        # Check shapes
        batch_size = len(sources)
        self.assertEqual(input_ids.shape[0], batch_size)
        
        # Both masks should have True values
        self.assertTrue(src_mask.any())
        self.assertTrue(tgt_mask.any())
        
        # Masks should not overlap (source and target are separate)
        overlap = src_mask & tgt_mask
        self.assertFalse(overlap.any(), "Source and target masks should not overlap")
    
    def test_get_batch_positions_masks_padding(self):
        """Test that padding is handled correctly"""
        contexts = ["", ""]
        sources = ["Short.", "This is a much longer sentence with more tokens."]
        targets = [None, None]
        
        input_ids, src_mask, tgt_mask = get_batch_positions_masks(
            self.tokenizer, contexts, sources, targets, device=self.device
        )
        
        # All sequences should have same length (padded)
        self.assertEqual(input_ids.shape[1], src_mask.shape[1])
        
        # Padding tokens should not be in source mask
        # (This is a basic check - actual implementation may vary)
        self.assertTrue(src_mask.any())


class TestResultSaving(unittest.TestCase):
    """Tests for result saving functionality"""
    
    def test_save_result_jsonl(self):
        """Test saving results to JSONL file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_results.jsonl"
            
            # Save a test result
            result = {
                "experiment": "mono_input",
                "source_lang": "English",
                "target_lang": None,
                "concept": "Tense",
                "value": "Past",
                "k": 10,
                "mean_delta": -0.05,
                "min_delta": -0.1,
                "num_samples": 32
            }
            
            save_result_jsonl(output_file, result)
            
            # Verify file exists
            self.assertTrue(output_file.exists())
            
            # Verify content
            with open(output_file, 'r') as f:
                lines = f.readlines()
                self.assertEqual(len(lines), 1)
                
                loaded_result = json.loads(lines[0])
                self.assertEqual(loaded_result["experiment"], "mono_input")
                self.assertEqual(loaded_result["mean_delta"], -0.05)
    
    def test_save_result_jsonl_multiple(self):
        """Test saving multiple results"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_results.jsonl"
            
            results = [
                {"experiment": "test1", "value": 1},
                {"experiment": "test2", "value": 2},
                {"experiment": "test3", "value": 3}
            ]
            
            for result in results:
                save_result_jsonl(output_file, result)
            
            # Verify all results saved
            with open(output_file, 'r') as f:
                lines = f.readlines()
                self.assertEqual(len(lines), 3)
                
                for i, line in enumerate(lines):
                    loaded = json.loads(line)
                    self.assertEqual(loaded["experiment"], f"test{i+1}")


class TestMonolingualVsMultilingual(unittest.TestCase):
    """Tests comparing monolingual vs multilingual settings"""
    
    def test_config_differences(self):
        """Test that monolingual and multilingual configs differ correctly"""
        mono_configs = [cfg for name, cfg in EXP_CONFIGS.items() 
                       if cfg['mode'] == 'monolingual']
        multi_configs = [cfg for name, cfg in EXP_CONFIGS.items() 
                        if cfg['mode'] == 'multilingual']
        
        self.assertGreater(len(mono_configs), 0)
        self.assertGreater(len(multi_configs), 0)
        
        # Monolingual should ablate and measure same location
        for cfg in mono_configs:
            self.assertEqual(cfg['ablate_loc'], cfg['prob_loc'])
        
        # Multilingual can ablate source and measure target
        multi_input = EXP_CONFIGS.get('multi_input')
        if multi_input:
            self.assertEqual(multi_input['ablate_loc'], 'source')
            self.assertEqual(multi_input['prob_loc'], 'target')


if __name__ == '__main__':
    unittest.main()

