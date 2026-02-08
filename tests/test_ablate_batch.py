"""
Unit tests for ablate_batch function
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import gc
from pathlib import Path
import torch
import numpy as np
from transformers import AutoTokenizer
from lang_probing_src.ablate import ablate_batch
from lang_probing_src.config import MODEL_ID, SAE_ID, NAME_TO_LANG_CODE, LANGUAGES_DEC
from lang_probing_src.utils import setup_model
from tests.test_utils import create_test_prompts
from lang_probing_src.utils_input_output import get_output_features_vector, get_input_features_vector, load_effects_files, get_language_pairs_and_concepts



class TestAblateBatchCore(unittest.TestCase):
    """Core functionality tests for ablate_batch"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.submodule, self.sae, self.tokenizer = setup_model(MODEL_ID, SAE_ID)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create test prompts
        self.contexts, self.sources, self.targets = create_test_prompts(
            monolingual=True, num_samples=4
        )
        
        # Tokenize to get input_ids
        prompts = [src for src in self.sources]
        encoding = self.tokenizer(prompts, padding=True, return_tensors="pt")
        self.input_ids = encoding["input_ids"].to(self.device)
        
        # Create simple masks (all True for testing)
        batch_size, seq_len = self.input_ids.shape
        self.ablate_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=self.device)
        self.prob_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=self.device)
        self.prob_mask[:, 0] = False  # Can't predict first token

        self.K = 10
        
        input_features_dir = Path("/projectnb/mcnet/jbrin/lang-probing/outputs/sentence_input_features/")
        feats_vec = get_input_features_vector(input_features_dir, "English", "Tense", "Past")
        self.feature_indices = np.argsort(feats_vec)[-self.K:]
    
    def tearDown(self):
        """Clean up after each test"""
        # Clean up tensors
        if hasattr(self, 'input_ids'):
            del self.input_ids
        if hasattr(self, 'ablate_mask'):
            del self.ablate_mask
        if hasattr(self, 'prob_mask'):
            del self.prob_mask
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
    def test_ablation_lowers_log_prob(self):
        """Test that ablating features results in negative delta (lower probability)"""
        # This test verifies the function runs and returns correct format
        # In practice with real model, delta should be negative
        delta_p = ablate_batch(
            self.model,
            self.submodule,
            self.sae,
            self.tokenizer,
            input_ids=self.input_ids,
            feature_indices=self.feature_indices,
            ablate_mask=self.ablate_mask,
            prob_mask=self.prob_mask
        )
        
        # Check return value format
        self.assertIsInstance(delta_p, torch.Tensor)
        self.assertGreater(len(delta_p), 0)
        
        # Check that values are finite
        self.assertTrue(torch.all(torch.isfinite(delta_p)))
        
        # With real model, ablation should lower probability (negative delta)
        self.assertLess(delta_p.mean().item(), 0, "Ablation should lower log probability")
        
        # Cleanup
        del delta_p
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    def test_return_value_format(self):
        """Test that return value has correct format"""
        delta_p = ablate_batch(
            self.model,
            self.submodule,
            self.sae,
            self.tokenizer,
            input_ids=self.input_ids,
            feature_indices=self.feature_indices,
            ablate_mask=self.ablate_mask,
            prob_mask=self.prob_mask
        )
        
        # Should be 1D tensor
        self.assertEqual(len(delta_p.shape), 1)
        
        # Should have values in reasonable range (for ratio: (new - old) / old)
        # Typically between -1 and some positive value
        self.assertTrue(torch.all(delta_p >= -2.0))  # Allow some margin
        self.assertTrue(torch.all(delta_p <= 2.0))
        
        # Cleanup
        del delta_p
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    def test_random_baseline_less_effective(self):
        """Test that random features have less effect than targeted features"""
        # Targeted features (top K)
        delta_p_targeted = ablate_batch(
            self.model,
            self.model.model.layers[16],
            self.sae,
            self.tokenizer,
            input_ids=self.input_ids,
            feature_indices=self.feature_indices,
            ablate_mask=self.ablate_mask,
            prob_mask=self.prob_mask
        )
        
        # Random features
        np.random.seed(42)
        random_indices = np.random.choice(32768, self.K, replace=False)
        delta_p_random = ablate_batch(
            self.model,
            self.model.model.layers[16],
            self.sae,
            self.tokenizer,
            input_ids=self.input_ids,
            feature_indices=random_indices,
            ablate_mask=self.ablate_mask,
            prob_mask=self.prob_mask
        )
        
        # Both should return valid results
        self.assertIsInstance(delta_p_targeted, torch.Tensor)
        self.assertIsInstance(delta_p_random, torch.Tensor)
        
        # With real model, targeted ablation should have larger effect (more negative)
        self.assertLess(delta_p_targeted.mean().item(), delta_p_random.mean().item(),
                        "Targeted features should have larger effect than random")
        
        # Cleanup
        del delta_p_targeted, delta_p_random
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class TestAblateBatchMasks(unittest.TestCase):
    """Tests for mask correctness"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model, self.submodule, self.sae, self.tokenizer = setup_model(MODEL_ID, SAE_ID)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Simple test prompt
        prompt = "The cat sat on the mat."
        encoding = self.tokenizer(prompt, return_tensors="pt")
        self.input_ids = encoding["input_ids"].to(self.device)
        
        self.feature_indices = np.array([100, 200, 300])
    
    def tearDown(self):
        """Clean up after each test"""
        if hasattr(self, 'input_ids'):
            del self.input_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    def test_shift_logic(self):
        """Test that prob_mask correctly aligns with shifted logits"""
        batch_size, seq_len = self.input_ids.shape
        
        # Create mask for all positions except first
        prob_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=self.device)
        prob_mask[:, 0] = False  # Can't predict first token
        
        # Function should handle this correctly
        delta_p = ablate_batch(
            self.model,
            self.submodule,
            self.sae,
            self.tokenizer,
            input_ids=self.input_ids,
            feature_indices=self.feature_indices,
            ablate_mask=prob_mask.clone(),
            prob_mask=prob_mask
        )
        
        # Should return results for seq_len - 1 positions (after shift)
        expected_len = (seq_len - 1) if seq_len > 1 else 0
        if expected_len > 0:
            self.assertEqual(len(delta_p), expected_len)
        
        # Cleanup
        del delta_p, prob_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    def test_empty_ablate_mask(self):
        """Test with empty ablate_mask (should return zeros or handle gracefully)"""
        batch_size, seq_len = self.input_ids.shape
        
        # Empty ablate mask
        ablate_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=self.device)
        prob_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=self.device)
        prob_mask[:, 0] = False
        
        # Should handle gracefully
        delta_p = ablate_batch(
            self.model,
            self.submodule,
            self.sae,
            self.tokenizer,
            input_ids=self.input_ids,
            feature_indices=self.feature_indices,
            ablate_mask=ablate_mask,
            prob_mask=prob_mask
        )
        
        # With no ablation, delta should be close to zero
        # (or empty if no positions to measure)
        if len(delta_p) > 0:
            # Should be very small (no ablation occurred)
            self.assertTrue(torch.allclose(delta_p, torch.zeros_like(delta_p), atol=1e-3))
        
        # Cleanup
        del delta_p, ablate_mask, prob_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    def test_empty_prob_mask(self):
        """Test with empty prob_mask"""
        batch_size, seq_len = self.input_ids.shape
        
        ablate_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=self.device)
        prob_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=self.device)
        
        # Should return empty tensor
        delta_p = ablate_batch(
            self.model,
            self.submodule,
            self.sae,
            self.tokenizer,
            input_ids=self.input_ids,
            feature_indices=self.feature_indices,
            ablate_mask=ablate_mask,
            prob_mask=prob_mask
        )
        
        # Should be empty
        self.assertEqual(len(delta_p), 0)
        
        # Cleanup
        del delta_p, ablate_mask, prob_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    def test_no_overlap_masks(self):
        """Test with no overlap between ablate_mask and prob_mask"""
        batch_size, seq_len = self.input_ids.shape
        
        # Ablate first half, measure second half
        ablate_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=self.device)
        ablate_mask[:, :seq_len//2] = True
        
        prob_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=self.device)
        prob_mask[:, seq_len//2:] = True
        prob_mask[:, 0] = False  # Can't predict first
        
        # Should handle this case
        delta_p = ablate_batch(
            self.model,
            self.submodule,
            self.sae,
            self.tokenizer,
            input_ids=self.input_ids,
            feature_indices=self.feature_indices,
            ablate_mask=ablate_mask,
            prob_mask=prob_mask
        )
        
        # Should return valid results (measuring downstream effects)
        self.assertIsInstance(delta_p, torch.Tensor)
        
        # Cleanup
        del delta_p, ablate_mask, prob_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class TestAblateBatchFeatureSelection(unittest.TestCase):
    """Tests for feature selection"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model, self.submodule, self.sae, self.tokenizer = setup_model(MODEL_ID, SAE_ID)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        prompt = "The cat sat on the mat."
        encoding = self.tokenizer(prompt, return_tensors="pt")
        self.input_ids = encoding["input_ids"].to(self.device)
        
        batch_size, seq_len = self.input_ids.shape
        self.ablate_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=self.device)
        self.prob_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=self.device)
        self.prob_mask[:, 0] = False
    
    def tearDown(self):
        """Clean up after each test"""
        if hasattr(self, 'input_ids'):
            del self.input_ids
        if hasattr(self, 'ablate_mask'):
            del self.ablate_mask
        if hasattr(self, 'prob_mask'):
            del self.prob_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    def test_feature_indices_numpy_array(self):
        """Test with numpy array feature indices"""
        feature_indices = np.array([100, 200, 300], dtype=np.int32)
        
        delta_p = ablate_batch(
            self.model,
            self.submodule,
            self.sae,
            self.tokenizer,
            input_ids=self.input_ids,
            feature_indices=feature_indices,
            ablate_mask=self.ablate_mask,
            prob_mask=self.prob_mask
        )
        
        self.assertIsInstance(delta_p, torch.Tensor)
        
        # Cleanup
        del delta_p
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    def test_feature_indices_list(self):
        """Test with list feature indices"""
        feature_indices = [100, 200, 300]
        
        delta_p = ablate_batch(
            self.model,
            self.submodule,
            self.sae,
            self.tokenizer,
            input_ids=self.input_ids,
            feature_indices=feature_indices,
            ablate_mask=self.ablate_mask,
            prob_mask=self.prob_mask
        )
        
        self.assertIsInstance(delta_p, torch.Tensor)
        
        # Cleanup
        del delta_p
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    def test_feature_indices_torch_tensor(self):
        """Test with torch tensor feature indices"""
        feature_indices = torch.tensor([100, 200, 300], dtype=torch.int64)
        
        delta_p = ablate_batch(
            self.model,
            self.submodule,
            self.sae,
            self.tokenizer,
            input_ids=self.input_ids,
            feature_indices=feature_indices,
            ablate_mask=self.ablate_mask,
            prob_mask=self.prob_mask
        )
        
        self.assertIsInstance(delta_p, torch.Tensor)
        
        # Cleanup
        del delta_p
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    def test_different_k_values(self):
        """Test with different K values"""
        for k in [1, 5, 10, 50]:
            feature_indices = np.random.choice(32768, k, replace=False)
            
            delta_p = ablate_batch(
                self.model,
                self.submodule,
                self.sae,
                self.tokenizer,
                input_ids=self.input_ids,
                feature_indices=feature_indices,
                ablate_mask=self.ablate_mask,
                prob_mask=self.prob_mask
            )
            
            self.assertIsInstance(delta_p, torch.Tensor)
            
            # Cleanup after each iteration
            del delta_p
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == '__main__':
    unittest.main()

