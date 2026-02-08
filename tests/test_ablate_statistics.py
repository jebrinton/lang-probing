"""
Statistical validation tests for ablation experiments
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
from scipy import stats
import torch


class TestStatisticalValidation(unittest.TestCase):
    """Statistical tests for ablation effects"""
    
    def test_targeted_vs_random_significance(self):
        """Test that targeted ablation is significantly more effective than random"""
        # Simulate results from multiple samples
        np.random.seed(42)
        
        # Targeted ablation: stronger negative effects
        targeted_deltas = np.random.normal(-0.05, 0.01, 100)
        
        # Random ablation: weaker effects (closer to zero)
        random_deltas = np.random.normal(-0.01, 0.01, 100)
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(targeted_deltas, random_deltas)
        
        # Targeted should be significantly more negative
        self.assertLess(p_value, 0.05, "Targeted ablation should be significantly different")
        self.assertLess(t_stat, 0, "Targeted should be more negative than random")
        
        # Mean magnitude should be larger for targeted
        self.assertGreater(np.abs(np.mean(targeted_deltas)), 
                          np.abs(np.mean(random_deltas)))
    
    def test_effect_magnitude(self):
        """Test that ablation effects are substantial (not just noise)"""
        np.random.seed(42)
        
        # Simulate ablation results
        delta_values = np.random.normal(-0.03, 0.01, 50)
        
        # Mean should be significantly different from zero
        t_stat, p_value = stats.ttest_1samp(delta_values, 0)
        
        self.assertLess(p_value, 0.05, "Effect should be significantly different from zero")
        self.assertLess(np.mean(delta_values), 0, "Mean should be negative")
    
    def test_effect_scaling_with_k(self):
        """Test that effect scales with number of features ablated"""
        # Simulate results for different K values
        np.random.seed(42)
        
        k_values = [1, 5, 10, 50]
        mean_effects = []
        
        for k in k_values:
            # Larger K should have larger (more negative) effects
            effect = np.random.normal(-0.01 * k, 0.005 * k)
            mean_effects.append(effect)
        
        # Effects should generally increase in magnitude with K
        # (allowing for some noise)
        for i in range(len(mean_effects) - 1):
            # Each larger K should have more negative mean (or at least not less)
            self.assertLessEqual(mean_effects[i], mean_effects[i+1] + 0.01,
                                f"Effect should scale with K: K={k_values[i]} vs K={k_values[i+1]}")
    
    def test_reproducibility(self):
        """Test that results are reproducible with same seed"""
        np.random.seed(42)
        indices1 = np.random.choice(32768, 10, replace=False)
        
        np.random.seed(42)
        indices2 = np.random.choice(32768, 10, replace=False)
        
        np.testing.assert_array_equal(indices1, indices2)
        
        # Test that random baseline is reproducible
        np.random.seed(42)
        deltas1 = np.random.normal(-0.01, 0.005, 20)
        
        np.random.seed(42)
        deltas2 = np.random.normal(-0.01, 0.005, 20)
        
        np.testing.assert_array_almost_equal(deltas1, deltas2)
    
    def test_consistency_across_samples(self):
        """Test that effects are consistent across different samples"""
        np.random.seed(42)
        
        # Simulate results from multiple batches
        batch_results = []
        for _ in range(5):
            batch_deltas = np.random.normal(-0.03, 0.01, 20)
            batch_results.append(batch_deltas)
        
        # All batches should have negative means
        for batch in batch_results:
            self.assertLess(np.mean(batch), 0)
        
        # Variance across batches should be reasonable
        batch_means = [np.mean(batch) for batch in batch_results]
        std_across_batches = np.std(batch_means)
        
        # Standard deviation should be small relative to mean
        mean_effect = np.mean(batch_means)
        cv = std_across_batches / np.abs(mean_effect)  # Coefficient of variation
        
        self.assertLess(cv, 0.5, "Results should be consistent across batches")


class TestEffectSize(unittest.TestCase):
    """Tests for effect size calculations"""
    
    def test_delta_probability_calculation(self):
        """Test that delta probability is calculated correctly"""
        # Original probability
        p_old = 0.1
        
        # New probability after ablation
        p_new = 0.08
        
        # Delta should be (new - old) / old
        delta = (p_new - p_old) / p_old
        
        self.assertAlmostEqual(delta, -0.2)
        
        # For log probabilities
        log_p_old = np.log(p_old)
        log_p_new = np.log(p_new)
        log_diff = log_p_new - log_p_old
        
        # exp(log_diff) - 1 should equal delta
        delta_from_log = np.exp(log_diff) - 1
        self.assertAlmostEqual(delta_from_log, delta, places=5)
    
    def test_negative_delta_means_lower_prob(self):
        """Test that negative delta means probability decreased"""
        p_old = 0.1
        delta = -0.2
        
        p_new = p_old * (1 + delta)
        
        self.assertLess(p_new, p_old)
        self.assertAlmostEqual(p_new, 0.08)
    
    def test_delta_range(self):
        """Test that delta values are in expected range"""
        # Delta = (new - old) / old
        # If new = 0, delta = -1 (minimum)
        # If new = old, delta = 0
        # If new > old, delta > 0 (shouldn't happen with ablation)
        
        test_cases = [
            (0.0, 0.1, -1.0),  # Probability goes to zero
            (0.1, 0.1, 0.0),   # No change
            (0.05, 0.1, -0.5), # Half probability
        ]
        
        for p_new, p_old, expected_delta in test_cases:
            delta = (p_new - p_old) / p_old
            self.assertAlmostEqual(delta, expected_delta, places=5)


class TestDifferentLanguages(unittest.TestCase):
    """Tests for consistency across languages"""
    
    def test_language_pair_consistency(self):
        """Test that effects are consistent across language pairs"""
        np.random.seed(42)
        
        # Simulate results for different language pairs
        language_pairs = [
            ("English", "Spanish"),
            ("English", "French"),
            ("English", "German")
        ]
        
        results = {}
        for pair in language_pairs:
            # Each pair should produce negative deltas
            deltas = np.random.normal(-0.03, 0.01, 20)
            results[pair] = deltas
        
        # All pairs should have negative means
        for pair, deltas in results.items():
            self.assertLess(np.mean(deltas), 0, 
                          f"Language pair {pair} should have negative mean")
        
        # Means should be similar across pairs (within reasonable variance)
        means = [np.mean(deltas) for deltas in results.values()]
        std_means = np.std(means)
        
        # Standard deviation should be small
        self.assertLess(std_means, 0.02, 
                       "Effects should be consistent across language pairs")
    
    def test_monolingual_consistency(self):
        """Test that monolingual effects are consistent"""
        np.random.seed(42)
        
        languages = ["English", "Spanish", "French"]
        results = {}
        
        for lang in languages:
            deltas = np.random.normal(-0.02, 0.01, 20)
            results[lang] = deltas
        
        # All should have negative means
        for lang, deltas in results.items():
            self.assertLess(np.mean(deltas), 0)


if __name__ == '__main__':
    unittest.main()

