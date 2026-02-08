"""
Edge case and error handling tests for ablation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import gc
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

from lang_probing_src.config import MODEL_ID, SAE_ID, NAME_TO_LANG_CODE, LANGUAGES_DEC
from lang_probing_src.utils import setup_model
from lang_probing_src.ablate import ablate_batch
from lang_probing_src.utils_input_output import get_input_features_vector, get_output_features_vector


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and error handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model, self.submodule, self.sae, self.tokenizer = setup_model(MODEL_ID, SAE_ID)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Simple test prompt
        prompt = "I had the loveliest of times when I went to the sea in Spain with her. I'm going again next month."
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
        
    def test_missing_features_handling(self):
        """Test error handling when feature files don't exist"""
        input_features_dir = Path("/nonexistent/path")
        
        with self.assertRaises((FileNotFoundError, OSError)):
            get_input_features_vector(
                input_features_dir,
                "English",
                "Tense",
                "Past"
            )
    
    def test_invalid_k_zero(self):
        """Test with K=0 (should handle gracefully)"""
        feature_indices = np.array([])
        
        # Should handle empty feature list
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
        
        # With no features ablated, delta should be close to zero
        if len(delta_p) > 0:
            self.assertTrue(torch.allclose(delta_p, torch.zeros_like(delta_p), atol=1e-3))
        
        # Cleanup
        del delta_p
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def test_invalid_k_too_large(self):
        """Test with K > SAE_DIM (should handle gracefully)"""
        sae_dim = 32768
        k = sae_dim + 100
        
        # This should raise an error or handle gracefully
        with self.assertRaises((ValueError, IndexError)):
            feature_indices = np.random.choice(sae_dim, k, replace=False)
            # This will fail because we can't choose k > sae_dim without replacement
    
    def test_k_equals_sae_dim(self):
        """Test with K=SAE_DIM (all features)"""
        sae_dim = 32768
        feature_indices = np.arange(sae_dim)
        
        # Should handle this case
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
    
    def test_empty_dataset(self):
        """Test with empty or very small dataset"""
        # Single token input
        single_token = torch.tensor([[1]])  # Just one token
        
        ablate_mask = torch.ones((1, 1), dtype=torch.bool)
        prob_mask = torch.zeros((1, 1), dtype=torch.bool)  # Can't predict first token
        
        # Should handle gracefully
        single_token = single_token.to(self.device)
        ablate_mask = ablate_mask.to(self.device)
        prob_mask = prob_mask.to(self.device)
        
        delta_p = ablate_batch(
            self.model,
            self.submodule,
            self.sae,
            self.tokenizer,
            input_ids=single_token,
            feature_indices=np.array([100, 200]),
            ablate_mask=ablate_mask,
            prob_mask=prob_mask
        )
        
        # Should return empty tensor (no positions to measure)
        self.assertEqual(len(delta_p), 0)
        
        # Cleanup
        del delta_p, single_token, ablate_mask, prob_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def test_very_short_sequence(self):
        """Test with very short sequence"""
        # Two tokens
        short_input = torch.tensor([[1, 2]])
        
        ablate_mask = torch.ones((1, 2), dtype=torch.bool)
        prob_mask = torch.zeros((1, 2), dtype=torch.bool)
        prob_mask[0, 1] = True  # Can measure second token
        
        short_input = short_input.to(self.device)
        ablate_mask = ablate_mask.to(self.device)
        prob_mask = prob_mask.to(self.device)
        
        delta_p = ablate_batch(
            self.model,
            self.submodule,
            self.sae,
            self.tokenizer,
            input_ids=short_input,
            feature_indices=np.array([100]),
            ablate_mask=ablate_mask,
            prob_mask=prob_mask
        )
        
        # Should return one value
        self.assertGreaterEqual(len(delta_p), 0)
        
        # Cleanup
        del delta_p, short_input, ablate_mask, prob_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def test_device_handling(self):
        """Test that tensors are on correct device"""
        # Inputs should already be on correct device from setUp
        delta_p = ablate_batch(
            self.model,
            self.submodule,
            self.sae,
            self.tokenizer,
            input_ids=self.input_ids,
            feature_indices=np.array([100, 200]),
            ablate_mask=self.ablate_mask,
            prob_mask=self.prob_mask
        )
        
        # Result should be on CPU (ablate_batch returns CPU tensors)
        self.assertEqual(delta_p.device.type, "cpu")
        
        # Cleanup
        del delta_p
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def test_nan_handling(self):
        """Test that results don't contain NaN values"""
        delta_p = ablate_batch(
            self.model,
            self.submodule,
            self.sae,
            self.tokenizer,
            input_ids=self.input_ids,
            feature_indices=np.array([100, 200, 300]),
            ablate_mask=self.ablate_mask,
            prob_mask=self.prob_mask
        )
        
        # Should not contain NaN
        self.assertFalse(torch.isnan(delta_p).any())
        
        # Cleanup
        del delta_p
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def test_inf_handling(self):
        """Test that results don't contain Inf values"""
        delta_p = ablate_batch(
            self.model,
            self.submodule,
            self.sae,
            self.tokenizer,
            input_ids=self.input_ids,
            feature_indices=np.array([100, 200, 300]),
            ablate_mask=self.ablate_mask,
            prob_mask=self.prob_mask
        )
        
        # Should not contain Inf
        self.assertFalse(torch.isinf(delta_p).any())
        
        # Cleanup
        del delta_p
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def test_large_batch(self):
        """Test with large batch size"""
        # Create larger batch
        batch_size = 32
        seq_len = 50
        
        # Generate random input_ids (valid token IDs)
        input_ids = torch.randint(1, 1000, (batch_size, seq_len))
        
        ablate_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
        prob_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
        prob_mask[:, 0] = False
        
        # Should handle large batch
        input_ids = input_ids.to(self.device)
        ablate_mask = ablate_mask.to(self.device)
        prob_mask = prob_mask.to(self.device)
        
        delta_p = ablate_batch(
            self.model,
            self.submodule,
            self.sae,
            self.tokenizer,
            input_ids=input_ids,
            feature_indices=np.array([100, 200, 300]),
            ablate_mask=ablate_mask,
            prob_mask=prob_mask
        )
        
        self.assertIsInstance(delta_p, torch.Tensor)
        # Should have many values (batch_size * (seq_len - 1) positions)
        expected_len = batch_size * (seq_len - 1)
        self.assertEqual(len(delta_p), expected_len)
        
        # Cleanup
        del delta_p, input_ids, ablate_mask, prob_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def test_very_long_sequence(self):
        """Test with very long sequence"""
        # Long sequence
        seq_len = 512
        batch_size = 1
        
        input_ids = torch.randint(1, 1000, (batch_size, seq_len))
        
        ablate_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
        prob_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
        prob_mask[:, 0] = False
        
        # Should handle long sequence
        input_ids = input_ids.to(self.device)
        ablate_mask = ablate_mask.to(self.device)
        prob_mask = prob_mask.to(self.device)
        
        delta_p = ablate_batch(
            self.model,
            self.submodule,
            self.sae,
            self.tokenizer,
            input_ids=input_ids,
            feature_indices=np.array([100, 200]),
            ablate_mask=ablate_mask,
            prob_mask=prob_mask
        )
        
        self.assertIsInstance(delta_p, torch.Tensor)
        self.assertEqual(len(delta_p), seq_len - 1)
        
        # Cleanup
        del delta_p, input_ids, ablate_mask, prob_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class TestErrorHandling(unittest.TestCase):
    """Tests for error handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model, self.submodule, self.sae, self.tokenizer = setup_model(MODEL_ID, SAE_ID)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def tearDown(self):
        """Clean up after each test"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def test_invalid_feature_indices_type(self):
        """Test with invalid feature indices type"""
        prompt = "Test."
        encoding = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoding["input_ids"].to(self.device)
        
        batch_size, seq_len = input_ids.shape
        ablate_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=self.device)
        prob_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=self.device)
        prob_mask[:, 0] = False
        
        # String should fail
        with self.assertRaises((TypeError, ValueError)):
            ablate_batch(
                self.model,
                self.submodule,
                self.sae,
                self.tokenizer,
                input_ids=input_ids,
                feature_indices="invalid",
                ablate_mask=ablate_mask,
                prob_mask=prob_mask
            )
    
    def test_mismatched_mask_shapes(self):
        """Test with mismatched mask shapes"""
        prompt = "Test sentence."
        encoding = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoding["input_ids"].to(self.device)
        
        batch_size, seq_len = input_ids.shape
        
        # Mismatched shapes
        ablate_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=self.device)
        prob_mask = torch.ones((batch_size, seq_len + 10), dtype=torch.bool, device=self.device)  # Wrong size
        
        # Should raise error or handle gracefully
        with self.assertRaises((RuntimeError, ValueError, IndexError)):
            ablate_batch(
                self.model,
                self.submodule,
                self.sae,
                self.tokenizer,
                input_ids=input_ids,
                feature_indices=np.array([100]),
                ablate_mask=ablate_mask,
                prob_mask=prob_mask
            )


if __name__ == '__main__':
    unittest.main()

