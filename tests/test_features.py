"""
Tests for feature analysis
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
import tempfile
from src.probe import train_probe
from src.features import (
    find_top_correlating_features,
    find_top_positive_negative_features,
    save_features,
    load_features,
    get_feature_indices,
    get_feature_weights,
    analyze_feature_overlap,
    get_shared_features_across_languages
)


class TestFeatureAnalysis(unittest.TestCase):
    """Tests for feature correlation analysis"""
    
    def setUp(self):
        """Create a trained probe for testing"""
        np.random.seed(42)
        
        # Create synthetic data
        n_samples = 100
        n_features = 50
        
        activations = np.random.randn(n_samples, n_features)
        # Make first 5 features strongly predictive
        labels = (activations[:, :5].sum(axis=1) > 0).astype(int)
        
        # Train probe
        self.probe = train_probe(activations, labels, seed=42)
        self.n_features = n_features
    
    def test_find_top_correlating_features(self):
        """Test finding top correlating features"""
        k = 10
        top_features = find_top_correlating_features(self.probe, k=k)
        
        # Should return list of tuples
        self.assertIsInstance(top_features, list)
        self.assertEqual(len(top_features), k)
        
        # Each item should be (idx, weight)
        for idx, weight in top_features:
            self.assertIsInstance(idx, int)
            self.assertIsInstance(weight, float)
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, self.n_features)
        
        # Should be sorted by absolute value (descending)
        weights = [abs(w) for _, w in top_features]
        self.assertEqual(weights, sorted(weights, reverse=True))
    
    def test_find_top_positive_negative_features(self):
        """Test finding top positive and negative features"""
        k = 5
        features = find_top_positive_negative_features(self.probe, k=k)
        
        # Should have both categories
        self.assertIn('positive', features)
        self.assertIn('negative', features)
        
        # Each should have k features
        self.assertEqual(len(features['positive']), k)
        self.assertEqual(len(features['negative']), k)
        
        # Positive features should have positive weights
        for _, weight in features['positive']:
            self.assertGreater(weight, 0)
        
        # Negative features should have negative weights
        for _, weight in features['negative']:
            self.assertLess(weight, 0)
    
    def test_save_load_features(self):
        """Test saving and loading features"""
        features_dict = {
            'English': find_top_correlating_features(self.probe, k=10),
            'Spanish': find_top_correlating_features(self.probe, k=10)
        }
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            save_features(features_dict, temp_path)
            
            # File should exist
            self.assertTrue(os.path.exists(temp_path))
            
            # Load features
            loaded = load_features(temp_path)
            
            # Should have same keys
            self.assertEqual(set(loaded.keys()), set(features_dict.keys()))
            
            # Should have same number of features
            for lang in features_dict:
                self.assertEqual(len(loaded[lang]), len(features_dict[lang]))
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_get_feature_indices(self):
        """Test extracting feature indices"""
        features = find_top_correlating_features(self.probe, k=10)
        
        # Get all indices
        indices = get_feature_indices(features)
        self.assertEqual(len(indices), 10)
        self.assertTrue(all(isinstance(i, int) for i in indices))
        
        # Get top 5
        indices_5 = get_feature_indices(features, top_k=5)
        self.assertEqual(len(indices_5), 5)
        self.assertEqual(indices_5, indices[:5])
    
    def test_get_feature_weights(self):
        """Test extracting feature weights"""
        features = find_top_correlating_features(self.probe, k=10)
        
        weights = get_feature_weights(features)
        self.assertEqual(len(weights), 10)
        self.assertTrue(all(isinstance(w, float) for w in weights))
    
    def test_analyze_feature_overlap(self):
        """Test analyzing overlap between feature sets"""
        # Create two feature dictionaries
        features_1 = {
            'English': [(1, 0.5), (2, 0.4), (3, 0.3)],
            'Spanish': [(4, 0.5), (5, 0.4), (6, 0.3)]
        }
        features_2 = {
            'English': [(1, 0.6), (2, 0.5), (7, 0.2)],
            'Spanish': [(4, 0.4), (8, 0.3), (9, 0.2)]
        }
        
        overlap = analyze_feature_overlap(features_1, features_2, top_k=3)
        
        # Should have both languages
        self.assertIn('English', overlap)
        self.assertIn('Spanish', overlap)
        
        # English should have overlap of 2 (indices 1 and 2)
        self.assertEqual(overlap['English']['overlap_count'], 2)
        
        # Spanish should have overlap of 1 (index 4)
        self.assertEqual(overlap['Spanish']['overlap_count'], 1)
    
    def test_get_shared_features_across_languages(self):
        """Test finding shared features across languages"""
        features_dict = {
            'English': [(1, 0.5), (2, 0.4), (3, 0.3)],
            'Spanish': [(1, 0.6), (2, 0.5), (4, 0.2)],
            'Turkish': [(1, 0.4), (5, 0.3), (6, 0.2)]
        }
        
        # Find features in at least 2 languages
        shared = get_shared_features_across_languages(features_dict, min_languages=2)
        
        # Feature 1 appears in all 3 languages
        self.assertIn(1, shared)
        self.assertEqual(shared[1]['count'], 3)
        
        # Feature 2 appears in 2 languages
        self.assertIn(2, shared)
        self.assertEqual(shared[2]['count'], 2)
        
        # Features 3, 4, 5, 6 appear in only 1 language
        self.assertNotIn(3, shared)
        self.assertNotIn(4, shared)


if __name__ == '__main__':
    unittest.main()

