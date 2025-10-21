"""
Tests for probe training and evaluation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
import tempfile
from src.probe import (
    train_probe,
    evaluate_probe,
    save_probe,
    load_probe,
    get_probe_predictions,
    get_probe_info
)


class TestProbeTraining(unittest.TestCase):
    """Tests for probe training"""
    
    def setUp(self):
        """Create synthetic data for testing"""
        np.random.seed(42)
        
        # Create synthetic activations (100 samples, 50 features)
        self.n_samples = 100
        self.n_features = 50
        
        # Create linearly separable data
        self.train_activations = np.random.randn(self.n_samples, self.n_features)
        # Make first feature predictive
        self.train_labels = (self.train_activations[:, 0] > 0).astype(int)
        
        # Create test data
        self.test_activations = np.random.randn(50, self.n_features)
        self.test_labels = (self.test_activations[:, 0] > 0).astype(int)
    
    def test_train_probe(self):
        """Test that probe can be trained"""
        probe = train_probe(self.train_activations, self.train_labels, seed=42)
        
        # Should have correct attributes
        self.assertTrue(hasattr(probe, 'coef_'))
        self.assertTrue(hasattr(probe, 'intercept_'))
        
        # Coefficients should have correct shape
        self.assertEqual(probe.coef_.shape, (1, self.n_features))
    
    def test_evaluate_probe(self):
        """Test probe evaluation"""
        probe = train_probe(self.train_activations, self.train_labels, seed=42)
        accuracy = evaluate_probe(probe, self.test_activations, self.test_labels)
        
        # Accuracy should be a float between 0 and 1
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
        
        # Should have reasonable accuracy on this simple task
        self.assertGreater(accuracy, 0.6)
    
    def test_save_load_probe(self):
        """Test saving and loading probes"""
        probe = train_probe(self.train_activations, self.train_labels, seed=42)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            temp_path = f.name
        
        try:
            save_probe(probe, temp_path)
            
            # File should exist
            self.assertTrue(os.path.exists(temp_path))
            
            # Load probe
            loaded_probe = load_probe(temp_path)
            
            # Should have same coefficients
            np.testing.assert_array_almost_equal(
                probe.coef_, 
                loaded_probe.coef_
            )
            np.testing.assert_array_almost_equal(
                probe.intercept_,
                loaded_probe.intercept_
            )
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_get_probe_predictions(self):
        """Test getting predictions from probe"""
        probe = train_probe(self.train_activations, self.train_labels, seed=42)
        results = get_probe_predictions(probe, self.test_activations)
        
        # Should have all expected keys
        self.assertIn('predictions', results)
        self.assertIn('probabilities', results)
        self.assertIn('logits', results)
        
        # Predictions should be binary
        self.assertTrue(np.all(np.isin(results['predictions'], [0, 1])))
        
        # Probabilities should be in [0, 1]
        self.assertTrue(np.all(results['probabilities'] >= 0))
        self.assertTrue(np.all(results['probabilities'] <= 1))
        
        # Should have correct shapes
        self.assertEqual(len(results['predictions']), len(self.test_activations))
        self.assertEqual(results['probabilities'].shape[0], len(self.test_activations))
    
    def test_get_probe_info(self):
        """Test getting probe information"""
        probe = train_probe(self.train_activations, self.train_labels, seed=42)
        info = get_probe_info(probe)
        
        # Should have expected keys
        self.assertIn('n_features', info)
        self.assertIn('intercept', info)
        self.assertIn('classes', info)
        
        # Values should be correct
        self.assertEqual(info['n_features'], self.n_features)
        self.assertEqual(info['classes'], [0, 1])


if __name__ == '__main__':
    unittest.main()

