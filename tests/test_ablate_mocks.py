"""
Mock/simulation tests for fast testing without full model loading
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import torch
import numpy as np
from transformers import AutoTokenizer

from lang_probing_src.ablate import ablate_batch
from tests.test_utils import (
    create_mock_model,
    create_mock_sae,
    create_test_prompts,
    create_synthetic_feature_vector
)


class TestMockModel(unittest.TestCase):
    """Tests using mock model for fast execution"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = create_mock_model(self.tokenizer, hidden_dim=4096, vocab_size=128256)
        self.sae = create_mock_sae(sae_dim=32768, hidden_dim=4096)
        
    def test_mock_model_basic(self):
        """Test that mock model works for basic operations"""
        prompt = "The cat sat on the mat."
        encoding = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoding["input_ids"]
        
        batch_size, seq_len = input_ids.shape
        ablate_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
        prob_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
        prob_mask[:, 0] = False
        
        feature_indices = np.array([100, 200, 300])
        
        # Should run without errors
        delta_p = ablate_batch(
            self.model,
            self.model.model.layers[16],
            self.sae,
            self.tokenizer,
            input_ids=input_ids,
            feature_indices=feature_indices,
            ablate_mask=ablate_mask,
            prob_mask=prob_mask
        )
        
        self.assertIsInstance(delta_p, torch.Tensor)
        self.assertGreater(len(delta_p), 0)
    
    def test_mock_sae_encode_decode(self):
        """Test that mock SAE encode/decode works"""
        # Create test activations
        batch_size, seq_len, hidden_dim = 2, 10, 4096
        acts = torch.randn(batch_size, seq_len, hidden_dim)
        
        # Encode
        encoded = self.sae.encode(acts)
        self.assertEqual(encoded.shape, (batch_size, seq_len, 32768))
        
        # Decode
        decoded = self.sae.decode(encoded)
        self.assertEqual(decoded.shape, (batch_size, seq_len, hidden_dim))
        
        # Should be reversible (approximately)
        # Note: Mock SAE is simple, so this is a basic check
        self.assertIsInstance(decoded, torch.Tensor)
    
    def test_mock_with_synthetic_features(self):
        """Test with synthetic feature vectors"""
        # Create synthetic feature vector with known strong features
        feat_vec = create_synthetic_feature_vector(
            sae_dim=32768,
            num_strong_features=10,
            seed=42
        )
        
        # Select top K features
        k = 5
        top_indices = np.argsort(feat_vec)[-k:]
        
        # Verify we got the strongest features
        sorted_values = np.sort(feat_vec)
        self.assertTrue(np.allclose(feat_vec[top_indices], sorted_values[-k:]))
        
        # Test ablation with these features
        prompt = "Test sentence."
        encoding = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoding["input_ids"]
        
        batch_size, seq_len = input_ids.shape
        ablate_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
        prob_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
        prob_mask[:, 0] = False
        
        delta_p = ablate_batch(
            self.model,
            self.model.model.layers[16],
            self.sae,
            self.tokenizer,
            input_ids=input_ids,
            feature_indices=top_indices,
            ablate_mask=ablate_mask,
            prob_mask=prob_mask
        )
        
        self.assertIsInstance(delta_p, torch.Tensor)


class TestSyntheticFeatures(unittest.TestCase):
    """Tests with synthetic feature vectors"""
    
    def test_synthetic_feature_generation(self):
        """Test synthetic feature vector generation"""
        feat_vec = create_synthetic_feature_vector(
            sae_dim=1000,
            num_strong_features=10,
            seed=42
        )
        
        self.assertEqual(len(feat_vec), 1000)
        
        # Should have some strong features
        max_val = np.max(np.abs(feat_vec))
        self.assertGreater(max_val, 1.0)  # Strong features should be > 1.0
        
        # Top features should be identifiable
        top_k = 10
        top_indices = np.argsort(np.abs(feat_vec))[-top_k:]
        top_values = np.abs(feat_vec[top_indices])
        
        # Top values should be larger than average
        mean_abs = np.mean(np.abs(feat_vec))
        self.assertGreater(np.mean(top_values), mean_abs * 2)
    
    def test_feature_selection_from_synthetic(self):
        """Test selecting features from synthetic vector"""
        feat_vec = create_synthetic_feature_vector(seed=42)
        
        # Select top K
        k = 20
        top_indices = np.argsort(feat_vec)[-k:]
        
        # Verify selection
        self.assertEqual(len(top_indices), k)
        self.assertEqual(len(np.unique(top_indices)), k)  # No duplicates
        
        # Verify they're actually the top K
        sorted_values = np.sort(feat_vec)
        selected_values = feat_vec[top_indices]
        self.assertTrue(np.allclose(np.sort(selected_values), sorted_values[-k:]))


class TestFastSmokeTests(unittest.TestCase):
    """Fast smoke tests that verify code runs without errors"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = create_mock_model(self.tokenizer)
        self.sae = create_mock_sae()
    
    def test_smoke_test_monolingual(self):
        """Smoke test for monolingual ablation"""
        contexts, sources, targets = create_test_prompts(
            monolingual=True, num_samples=2
        )
        
        prompts = [src for src in sources]
        encoding = self.tokenizer(prompts, padding=True, return_tensors="pt")
        input_ids = encoding["input_ids"]
        
        batch_size, seq_len = input_ids.shape
        ablate_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
        prob_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
        prob_mask[:, 0] = False
        
        feature_indices = np.array([100, 200, 300])
        
        # Should run without errors
        delta_p = ablate_batch(
            self.model,
            self.model.model.layers[16],
            self.sae,
            self.tokenizer,
            input_ids=input_ids,
            feature_indices=feature_indices,
            ablate_mask=ablate_mask,
            prob_mask=prob_mask
        )
        
        self.assertIsInstance(delta_p, torch.Tensor)
    
    def test_smoke_test_multilingual(self):
        """Smoke test for multilingual ablation"""
        contexts, sources, targets = create_test_prompts(
            monolingual=False, num_samples=2
        )
        
        # Create prompts with context
        prompts = [f"{ctx}{src} >> {tgt}" for ctx, src, tgt in zip(contexts, sources, targets)]
        encoding = self.tokenizer(prompts, padding=True, return_tensors="pt")
        input_ids = encoding["input_ids"]
        
        batch_size, seq_len = input_ids.shape
        ablate_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
        prob_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
        prob_mask[:, 0] = False
        
        feature_indices = np.array([100, 200, 300])
        
        # Should run without errors
        delta_p = ablate_batch(
            self.model,
            self.model.model.layers[16],
            self.sae,
            self.tokenizer,
            input_ids=input_ids,
            feature_indices=feature_indices,
            ablate_mask=ablate_mask,
            prob_mask=prob_mask
        )
        
        self.assertIsInstance(delta_p, torch.Tensor)
    
    def test_smoke_test_different_k_values(self):
        """Smoke test with different K values"""
        prompt = "Test."
        encoding = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoding["input_ids"]
        
        batch_size, seq_len = input_ids.shape
        ablate_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
        prob_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
        prob_mask[:, 0] = False
        
        for k in [1, 5, 10, 50]:
            feature_indices = np.random.choice(32768, k, replace=False)
            
            delta_p = ablate_batch(
                self.model,
                self.model.model.layers[16],
                self.sae,
                self.tokenizer,
                input_ids=input_ids,
                feature_indices=feature_indices,
                ablate_mask=ablate_mask,
                prob_mask=prob_mask
            )
            
            self.assertIsInstance(delta_p, torch.Tensor)


if __name__ == '__main__':
    unittest.main()

