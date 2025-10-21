"""
Tests for data loading and processing
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
from src.data import (
    get_ud_filepath, 
    get_available_concepts, 
    ProbingDataset, 
    balance_dataset,
    concept_filter
)
from src.config import UD_BASE_FOLDER, LANGUAGES


class TestDataLoading(unittest.TestCase):
    """Tests for UD treebank loading"""
    
    def test_get_ud_filepath_train(self):
        """Test that we can find train files"""
        for language in LANGUAGES:
            filepath = get_ud_filepath(language, split='train')
            if filepath:  # May be None if file doesn't exist
                self.assertTrue(os.path.exists(filepath))
                self.assertTrue(filepath.endswith('.conllu'))
    
    def test_get_ud_filepath_test(self):
        """Test that we can find test files"""
        for language in LANGUAGES:
            filepath = get_ud_filepath(language, split='test')
            # Test files may not exist for all languages, just check format
            if filepath:
                self.assertTrue(filepath.endswith('.conllu'))
    
    def test_get_ud_filepath_invalid(self):
        """Test with invalid language"""
        filepath = get_ud_filepath("NonexistentLanguage-XXX", split='train')
        self.assertIsNone(filepath)
    
    def test_get_available_concepts(self):
        """Test concept extraction from .conllu files"""
        # Use English-PUD as test case
        filepath = get_ud_filepath("English-PUD", split='train')
        if filepath:
            concepts = get_available_concepts(filepath)
            
            # Should be a dict
            self.assertIsInstance(concepts, dict)
            
            # Should have some common concepts
            # Note: not all concepts may be present in all files
            self.assertGreater(len(concepts), 0)
            
            # Each value should be a set
            for key, value in concepts.items():
                self.assertIsInstance(value, set)


class TestProbingDataset(unittest.TestCase):
    """Tests for ProbingDataset class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.filepath = get_ud_filepath("English-PUD", split='train')
        if self.filepath:
            self.concepts = get_available_concepts(self.filepath)
    
    def test_dataset_creation(self):
        """Test dataset can be created"""
        if not self.filepath:
            self.skipTest("English-PUD train file not found")
        
        # Find a concept with values
        if not self.concepts:
            self.skipTest("No concepts found")
        
        concept_key = list(self.concepts.keys())[0]
        concept_value = list(self.concepts[concept_key])[0]
        
        dataset = ProbingDataset(self.filepath, concept_key, concept_value)
        
        # Should have loaded some data
        self.assertGreater(len(dataset), 0)
    
    def test_dataset_getitem(self):
        """Test dataset __getitem__ returns correct format"""
        if not self.filepath or not self.concepts:
            self.skipTest("Required data not available")
        
        concept_key = list(self.concepts.keys())[0]
        concept_value = list(self.concepts[concept_key])[0]
        
        dataset = ProbingDataset(self.filepath, concept_key, concept_value)
        
        if len(dataset) > 0:
            item = dataset[0]
            
            # Should be a dict with 'sentence' and 'label'
            self.assertIn('sentence', item)
            self.assertIn('label', item)
            
            # Sentence should be string
            self.assertIsInstance(item['sentence'], str)
            
            # Label should be 0 or 1
            self.assertIn(item['label'], [0, 1])


class TestDatasetBalancing(unittest.TestCase):
    """Tests for dataset balancing"""
    
    def test_balance_dataset(self):
        """Test that balancing works correctly"""
        filepath = get_ud_filepath("English-PUD", split='train')
        if not filepath:
            self.skipTest("English-PUD train file not found")
        
        concepts = get_available_concepts(filepath)
        if not concepts:
            self.skipTest("No concepts found")
        
        concept_key = list(concepts.keys())[0]
        concept_value = list(concepts[concept_key])[0]
        
        dataset = ProbingDataset(filepath, concept_key, concept_value)
        
        if len(dataset) < 10:
            self.skipTest("Not enough samples")
        
        balanced = balance_dataset(dataset, seed=42)
        
        if balanced is None:
            self.skipTest("Balancing returned None (insufficient samples)")
        
        # Count labels
        labels = [item['label'] for item in balanced]
        n_positive = sum(labels)
        n_negative = len(labels) - n_positive
        
        # Should be balanced (equal counts)
        self.assertEqual(n_positive, n_negative)


if __name__ == '__main__':
    # Set up logging to suppress warnings during tests
    import logging
    logging.basicConfig(level=logging.ERROR)
    
    unittest.main()

