"""
Dentate Gyrus Pattern Separation
Sparse expansion layer: 32D → 128D with 2% sparsity
"""
import numpy as np
import logging
from typing import List

logger = logging.getLogger(__name__)


class DentateGyrus:
    """
    Sparse expansion layer: 32D → 128D with 2% sparsity.
    Transforms similar inputs into non-overlapping representations.
    """
    
    def __init__(self, input_dim: int = 32, expansion_factor: int = 4, sparsity: float = 0.02):
        """
        Initialize Dentate Gyrus layer
        
        Args:
            input_dim: Input dimension (default: 32)
            expansion_factor: Expansion factor (default: 4, so 32 → 128)
            sparsity: Target sparsity (default: 0.02 = 2%)
        """
        self.input_dim = input_dim
        self.output_dim = input_dim * expansion_factor  # 128
        self.k = max(1, int(self.output_dim * sparsity))  # ~3 active neurons
        
        # Random sparse projection matrix
        self.W = np.random.randn(input_dim, self.output_dim).astype(np.float32) * 0.1
        
        logger.debug(f"DG initialized: {input_dim}D → {self.output_dim}D, sparsity={sparsity:.1%}, k={self.k}")
    
    def expand(self, capsule: np.ndarray) -> np.ndarray:
        """
        Sparse expansion with top-k selection.
        
        Args:
            capsule: 32D input vector
        
        Returns:
            128D sparse vector with ~2% activation
        """
        if len(capsule) != self.input_dim:
            raise ValueError(f"Expected {self.input_dim}D input, got {len(capsule)}D")
        
        # Project to high dimension
        activations = capsule @ self.W
        
        # Top-k sparsification
        threshold_idx = np.argsort(np.abs(activations))[-self.k:]
        
        sparse = np.zeros(self.output_dim, dtype=np.float32)
        sparse[threshold_idx] = activations[threshold_idx]
        
        norm = np.linalg.norm(sparse)
        if norm > 1e-8:
            sparse = sparse / norm
        
        return sparse
    
    def expand_batch(self, capsules: np.ndarray) -> np.ndarray:
        """Batch processing for efficiency."""
        if capsules.ndim != 2 or capsules.shape[1] != self.input_dim:
            raise ValueError(f"Expected (batch, {self.input_dim}), got {capsules.shape}")
        
        batch_size = capsules.shape[0]
        activations = capsules @ self.W
        
        sparse_batch = np.zeros_like(activations)
        for i in range(batch_size):
            threshold_idx = np.argsort(np.abs(activations[i]))[-self.k:]
            sparse_batch[i, threshold_idx] = activations[i, threshold_idx]
            
            norm = np.linalg.norm(sparse_batch[i])
            if norm > 1e-8:
                sparse_batch[i] /= norm
        
        return sparse_batch
