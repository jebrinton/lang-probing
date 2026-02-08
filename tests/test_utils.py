"""
Test utilities for ablation tests
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from unittest.mock import MagicMock


def create_mock_sae(sae_dim: int = 32768, hidden_dim: int = 4096) -> MagicMock:
    """
    Create a simple mock SAE for testing.
    
    Args:
        sae_dim: SAE feature dimension
        hidden_dim: Hidden dimension of activations
        
    Returns:
        Mock SAE object with encode and decode methods
    """
    sae = MagicMock()
    
    # Create simple linear transformations for encode/decode
    # In reality, SAEs are more complex, but this is sufficient for testing
    encode_weight = torch.randn(sae_dim, hidden_dim) * 0.01
    decode_weight = encode_weight.T  # Simple transpose for mock
    
    def encode_fn(acts):
        # Simple linear encoding with ReLU
        encoded = torch.matmul(acts, encode_weight.T)
        return torch.relu(encoded)
    
    def decode_fn(encoded):
        # Simple linear decoding
        return torch.matmul(encoded, decode_weight)
    
    sae.encode = encode_fn
    sae.decode = decode_fn
    sae.sae_dim = sae_dim
    sae.hidden_dim = hidden_dim
    
    return sae


def create_mock_model(tokenizer, hidden_dim: int = 4096, vocab_size: int = 128256) -> MagicMock:
    """
    Create a simple mock model for testing.
    
    Args:
        tokenizer: Tokenizer to use
        hidden_dim: Hidden dimension
        vocab_size: Vocabulary size
        
    Returns:
        Mock model object with trace context manager
    """
    model = MagicMock()
    model.tokenizer = tokenizer
    
    # Create a simple mock trace context
    class MockTracer:
        def __init__(self, model, input_ids):
            self.model = model
            self.input_ids = input_ids
            self.invokes = []
            
        def __enter__(self):
            return self
            
        def __exit__(self, *args):
            pass
            
        def invoke(self, input_ids):
            return MockInvoke(self, input_ids)
    
    class MockInvoke:
        def __init__(self, tracer, input_ids):
            self.tracer = tracer
            self.input_ids = input_ids
            self._saved = {}
            
        def __enter__(self):
            return self
            
        def __exit__(self, *args):
            pass
            
        def save(self):
            # Return a saveable object
            return MockSaveable()
    
    class MockSaveable:
        def __init__(self):
            self.value = None
    
    class MockSubmodule:
        def __init__(self, hidden_dim):
            self.hidden_dim = hidden_dim
            self._input = None
            self._output = None
            self.tracer = None  # Will be set by trace context
            
        @property
        def input(self):
            if self._input is None:
                # Generate random activations
                if self.tracer is not None and hasattr(self.tracer, 'input_ids'):
                    batch_size = self.tracer.input_ids.shape[0]
                    seq_len = self.tracer.input_ids.shape[1]
                else:
                    # Default values if tracer not set
                    batch_size, seq_len = 1, 10
                self._input = torch.randn(batch_size, seq_len, self.hidden_dim)
            return self._input
        
        @input.setter
        def input(self, value):
            self._input = value
            
        @property
        def output(self):
            if self._output is None:
                # Generate random output
                if self.tracer is not None and hasattr(self.tracer, 'input_ids'):
                    batch_size = self.tracer.input_ids.shape[0]
                    seq_len = self.tracer.input_ids.shape[1]
                else:
                    # Default values if tracer not set
                    batch_size, seq_len = 1, 10
                self._output = torch.randn(batch_size, seq_len, self.hidden_dim)
            return self._output
        
        @output.setter
        def output(self, value):
            self._output = value
    
    # Create submodule
    submodule = MockSubmodule(hidden_dim)
    submodule.tracer = None  # Will be set in trace
    
    # Create lm_head mock
    class MockLMHead:
        def __init__(self, vocab_size, hidden_dim):
            self.vocab_size = vocab_size
            self.hidden_dim = hidden_dim
            self._output = None
            self.tracer = None  # Will be set by trace context
            
        @property
        def output(self):
            if self._output is None:
                # Generate random logits
                if self.tracer is not None and hasattr(self.tracer, 'input_ids'):
                    batch_size = self.tracer.input_ids.shape[0]
                    seq_len = self.tracer.input_ids.shape[1]
                else:
                    # Default values if tracer not set
                    batch_size, seq_len = 1, 10
                self._output = torch.randn(batch_size, seq_len, self.vocab_size)
            return self._output
        
        @output.setter
        def output(self, value):
            self._output = value
    
    lm_head = MockLMHead(vocab_size, hidden_dim)
    
    def trace(input_ids, **kwargs):
        tracer = MockTracer(model, input_ids)
        tracer.input_ids = input_ids
        submodule.tracer = tracer
        lm_head.tracer = tracer
        return tracer
    
    model.trace = trace
    model.model = MagicMock()
    model.model.layers = [None] * 17  # Layer 16 exists
    model.model.layers[16] = submodule
    model.lm_head = lm_head
    
    return model


def create_test_prompts(
    monolingual: bool = True,
    num_samples: int = 4,
    source_lang: str = "English",
    target_lang: Optional[str] = None
) -> Tuple[list, list, list]:
    """
    Generate test prompts with known structure.
    
    Args:
        monolingual: If True, create monolingual prompts (no target)
        num_samples: Number of prompts to generate
        source_lang: Source language name
        target_lang: Target language name (for multilingual)
        
    Returns:
        Tuple of (contexts, sources, targets) lists
    """
    contexts = []
    sources = []
    targets = []
    
    if monolingual:
        # Simple monolingual sentences
        test_sentences = [
            "The cat sat on the mat.",
            "She walked to the store.",
            "They played in the park.",
            "He read a book yesterday.",
        ]
        for i in range(num_samples):
            contexts.append("")
            sources.append(test_sentences[i % len(test_sentences)])
            targets.append(None)
    else:
        # Multilingual with 2-shot context
        test_pairs = [
            ("Hello, how are you?", "Hola, ¿cómo estás?"),
            ("I love programming.", "Me encanta programar."),
            ("The weather is nice.", "El clima está agradable."),
            ("She studies mathematics.", "Ella estudia matemáticas."),
        ]
        for i in range(num_samples):
            idx_1 = (i + 1) % num_samples
            idx_2 = (i + 2) % num_samples
            ctx = f"{test_pairs[idx_2][0]} >> {test_pairs[idx_2][1]}\n{test_pairs[idx_1][0]} >> {test_pairs[idx_1][1]}\n"
            contexts.append(ctx)
            sources.append(test_pairs[i][0])
            targets.append(test_pairs[i][1])
    
    return contexts, sources, targets


def load_test_features(
    output_dir: Path,
    language: str,
    concept: str,
    value: str,
    sae_dim: int = 32768
) -> np.ndarray:
    """
    Load or generate test feature vectors.
    
    Args:
        output_dir: Directory where features might be stored
        language: Language name
        concept: Concept name (e.g., "Tense")
        value: Concept value (e.g., "Past")
        sae_dim: SAE dimension
        
    Returns:
        Feature vector as numpy array
    """
    # Try to load from file first
    feature_path = output_dir / language / concept / value / "diff_vector.pt"
    
    if feature_path.exists():
        return torch.load(feature_path).numpy()
    else:
        # Generate synthetic feature vector with some strong features
        np.random.seed(42)
        feat_vec = np.random.randn(sae_dim) * 0.1
        # Make top 10 features much stronger
        top_indices = np.argsort(np.abs(feat_vec))[-10:]
        feat_vec[top_indices] = np.random.randn(10) * 2.0
        return feat_vec


def create_synthetic_feature_vector(
    sae_dim: int = 32768,
    num_strong_features: int = 10,
    seed: int = 42
) -> np.ndarray:
    """
    Generate synthetic feature vector with known strong features.
    
    Args:
        sae_dim: SAE dimension
        num_strong_features: Number of features with strong signal
        seed: Random seed
        
    Returns:
        Feature vector with known strong features
    """
    np.random.seed(seed)
    feat_vec = np.random.randn(sae_dim) * 0.1
    # Make top features much stronger
    top_indices = np.argsort(np.abs(feat_vec))[-num_strong_features:]
    feat_vec[top_indices] = np.random.randn(num_strong_features) * 2.0
    return feat_vec

