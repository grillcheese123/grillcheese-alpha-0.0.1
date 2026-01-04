"""
Custom SentencePiece Embedding Model using Vulkan Shaders
Fully GPU-accelerated sentence embeddings
"""

import numpy as np
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import json

logger = logging.getLogger(__name__)

class EmbedderConfig:
    """Configuration for custom embedder"""
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_dim: int = 384,
        num_layers: int = 2,
        num_heads: int = 6,
        ffn_dim: int = 1536,
        max_seq_len: int = 128,
        dropout: float = 0.1
    ):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        
        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim // num_heads
    
    @classmethod
    def from_json(cls, path: Path):
        """Load config from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_json(self, path: Path):
        """Save config to JSON file"""
        config_dict = {
            'vocab_size': self.vocab_size,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'ffn_dim': self.ffn_dim,
            'max_seq_len': self.max_seq_len,
            'dropout': self.dropout
        }
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
