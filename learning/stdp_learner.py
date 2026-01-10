"""
STDP (Spike-Timing Dependent Plasticity) Learner for GrillCheese

Implements bio-inspired learning based on temporal correlations between 
embeddings. This strengthens associations between concepts that appear 
together in conversations.

GPU-accelerated using Vulkan compute shaders when available.
"""
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Try to import GPU backend
try:
    from vulkan_backend.vulkan_compute import VulkanCompute
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


class STDPLearner:
    """
    Spike-Timing Dependent Plasticity learner for embedding associations.
    
    Key concepts:
    - LTP (Long-Term Potentiation): Strengthen connections when tokens fire together
    - LTD (Long-Term Depression): Weaken connections over time (decay)
    - Time Window: Only nearby (in time) activations are associated
    
    This enables:
    - Learning which concepts frequently appear together
    - Building semantic associations from conversation patterns
    - Reinforcing important memory pathways
    """
    
    def __init__(
        self,
        learning_rate_plus: float = 0.01,
        learning_rate_minus: float = 0.012,
        time_window: int = 5,
        w_min: float = 0.0,
        w_max: float = 1.0,
        decay: float = 0.99,
        use_gpu: bool = True
    ):
        """
        Initialize STDP learner
        
        Args:
            learning_rate_plus: LTP learning rate (strengthening)
            learning_rate_minus: LTD learning rate (weakening)
            time_window: Time window for temporal correlation (in steps)
            w_min: Minimum weight
            w_max: Maximum weight
            decay: Passive decay factor for weights
            use_gpu: Whether to use GPU acceleration when available
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.gpu_backend = None
        if self.use_gpu:
            try:
                self.gpu_backend = VulkanCompute()
                logger.info("GPU acceleration enabled for STDP learning")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU for STDP: {e}, using CPU")
                self.use_gpu = False
        self.lr_plus = learning_rate_plus
        self.lr_minus = learning_rate_minus
        self.window = time_window
        self.w_min = w_min
        self.w_max = w_max
        self.decay = decay
        
        # Token weights: token_id -> weight (salience)
        self.token_weights: Dict[int, float] = {}
        
        # Association matrix: (token_i, token_j) -> association strength
        # Sparse representation for memory efficiency
        self.associations: Dict[tuple, float] = {}
        
        # Spike timing traces: token_id -> last_spike_time
        self.spike_traces: Dict[int, float] = {}
        self.current_time = 0.0
        
        # Statistics
        self.stats = {
            'total_updates': 0,
            'ltp_events': 0,
            'ltd_events': 0,
            'active_tokens': 0,
            'active_associations': 0
        }
    
    def process_sequence(
        self,
        token_ids: List[int],
        spikes: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Process a sequence of tokens and update weights using STDP
        
        Args:
            token_ids: List of token IDs (from embedder hash)
            spikes: Optional binary array indicating which tokens "spiked"
                   If None, assumes all tokens fire sequentially
        
        Returns:
            Dictionary with learning statistics
        """
        if not token_ids:
            return {'updates': 0}
        
        # Default: all tokens fire
        if spikes is None:
            spikes = np.ones(len(token_ids), dtype=bool)
        
        updates = 0
        ltp_count = 0
        
        for t, (token, is_spike) in enumerate(zip(token_ids, spikes)):
            if not is_spike:
                continue
            
            # Current spike time (100ms steps)
            now = self.current_time + t * 0.1
            
            # LTP: Strengthen association with recently fired tokens
            for prev_token, prev_time in self.spike_traces.items():
                dt = now - prev_time
                if 0 < dt < self.window:
                    # Hebbian: "Cells that fire together, wire together"
                    delta = self.lr_plus * np.exp(-dt)
                    self._update_weight(token, delta)
                    self._update_association(prev_token, token, delta)
                    updates += 1
                    ltp_count += 1
            
            # Update trace for this token
            self.spike_traces[token] = now
        
        self.current_time += len(token_ids) * 0.1
        
        # Periodic cleanup and decay
        if self.current_time > 100.0:
            self._decay_weights()
            self.current_time = 0.0
            self.spike_traces.clear()
        
        # Update stats
        self.stats['total_updates'] += updates
        self.stats['ltp_events'] += ltp_count
        self.stats['active_tokens'] = len(self.token_weights)
        self.stats['active_associations'] = len(self.associations)
        
        return {
            'updates': updates,
            'ltp_events': ltp_count,
            'active_tokens': len(self.token_weights)
        }
    
    def process_embedding_pair(
        self,
        emb1_indices: List[int],
        emb2_indices: List[int],
        relevance: float = 1.0
    ) -> Dict[str, Any]:
        """
        Learn association between two embeddings (e.g., query and retrieved memory)
        
        Uses GPU acceleration if available for large sequences.
        
        Args:
            emb1_indices: Token indices from first embedding
            emb2_indices: Token indices from second embedding
            relevance: Relevance score (0-1) to scale learning
        
        Returns:
            Learning statistics
        """
        # Try GPU batch processing if available and sequences are large enough
        if self.use_gpu and self.gpu_backend is not None and len(emb1_indices) + len(emb2_indices) > 20:
            try:
                return self._process_pair_gpu(emb1_indices, emb2_indices, relevance)
            except Exception as e:
                logger.debug(f"GPU STDP failed: {e}, falling back to CPU")
        
        # CPU implementation (original)
        updates = 0
        
        # Cross-associate tokens from both embeddings
        for idx1 in emb1_indices[:50]:  # Limit for efficiency
            for idx2 in emb2_indices[:50]:
                if idx1 != idx2:
                    delta = self.lr_plus * relevance * 0.5
                    self._update_association(idx1, idx2, delta)
                    updates += 1
        
        return {'updates': updates, 'relevance': relevance}
    
    def _process_pair_gpu(
        self,
        emb1_indices: List[int],
        emb2_indices: List[int],
        relevance: float = 1.0
    ) -> Dict[str, Any]:
        """
        GPU-accelerated STDP processing for embedding pairs
        
        Converts token indices to spike sequences and uses GPU STDP shader.
        """
        if not self.use_gpu or self.gpu_backend is None:
            raise RuntimeError("GPU backend not available")
        
        # Convert indices to spike sequences
        # Create sparse activation matrices for GPU STDP
        max_idx = max(max(emb1_indices, default=0), max(emb2_indices, default=0)) + 1
        seq_len = max(len(emb1_indices), len(emb2_indices))
        
        # Create binary spike sequences (batch=1, time=seq_len, dim=max_idx)
        pre_spikes = np.zeros((1, seq_len, max_idx), dtype=np.float32)
        post_spikes = np.zeros((1, seq_len, max_idx), dtype=np.float32)
        
        # Set spikes at token positions
        for t, idx in enumerate(emb1_indices[:seq_len]):
            if idx < max_idx:
                pre_spikes[0, t, idx] = 1.0
        for t, idx in enumerate(emb2_indices[:seq_len]):
            if idx < max_idx:
                post_spikes[0, t, idx] = 1.0
        
        # Initialize weights if needed
        if not hasattr(self, '_gpu_weights') or self._gpu_weights.shape[0] < max_idx:
            old_size = self._gpu_weights.shape[0] if hasattr(self, '_gpu_weights') else 0
            new_weights = np.zeros((max_idx, max_idx), dtype=np.float32)
            if hasattr(self, '_gpu_weights'):
                new_weights[:old_size, :old_size] = self._gpu_weights
            self._gpu_weights = new_weights
            self._gpu_pre_trace = np.zeros((1, max_idx), dtype=np.float32)
            self._gpu_post_trace = np.zeros((1, max_idx), dtype=np.float32)
        
        # Use GPU STDP learning
        try:
            updated_weights, updated_pre_trace, updated_post_trace = self.gpu_backend.stdp_learning(
                pre_activations=pre_spikes,
                post_activations=post_spikes,
                weights=self._gpu_weights,
                pre_trace=self._gpu_pre_trace,
                post_trace=self._gpu_post_trace,
                lr_potentiation=self.lr_plus * relevance,
                lr_depression=self.lr_minus,
                trace_decay=0.9
            )
            
            # Count updates (non-zero weight changes)
            weight_changes = np.abs(updated_weights - self._gpu_weights)
            updates = int(np.sum(weight_changes > 1e-6))
            
            # Update weights and traces
            self._gpu_weights = updated_weights
            self._gpu_pre_trace = updated_pre_trace
            self._gpu_post_trace = updated_post_trace
            
            # Update token weights from GPU weights (for compatibility)
            for i in range(max_idx):
                for j in range(max_idx):
                    if updated_weights[i, j] > 0:
                        self.token_weights[i] = max(self.token_weights.get(i, 0), float(updated_weights[i, j]))
                        if i != j:
                            self._update_association(i, j, float(updated_weights[i, j]) * 0.1)
            
            return {
                'updates': updates,
                'relevance': relevance,
                'gpu_accelerated': True
            }
        except Exception as e:
            logger.debug(f"GPU STDP processing failed: {e}")
            # Fall back to CPU
            raise
    
    def _update_weight(self, token: int, delta: float) -> None:
        """Update token weight with bounds"""
        w = self.token_weights.get(token, 0.5)
        w += delta
        self.token_weights[token] = max(self.w_min, min(self.w_max, w))
    
    def _update_association(self, token1: int, token2: int, delta: float) -> None:
        """Update association strength between two tokens"""
        key = (min(token1, token2), max(token1, token2))  # Symmetric
        w = self.associations.get(key, 0.0)
        w += delta
        self.associations[key] = max(0.0, min(1.0, w))
    
    def _decay_weights(self) -> None:
        """Apply exponential decay to all weights"""
        # Decay token weights
        for tok in list(self.token_weights.keys()):
            self.token_weights[tok] *= self.decay
            if self.token_weights[tok] < 0.01:
                del self.token_weights[tok]
        
        # Decay associations
        for key in list(self.associations.keys()):
            self.associations[key] *= self.decay
            if self.associations[key] < 0.01:
                del self.associations[key]
        
        self.stats['ltd_events'] += 1
    
    def get_modulations(self, token_ids: List[int]) -> np.ndarray:
        """
        Get STDP modulation factors for a sequence of tokens
        
        Used to modulate attention or memory retrieval based on learned salience
        """
        mods = np.ones(len(token_ids), dtype=np.float32)
        for i, tok in enumerate(token_ids):
            w = self.token_weights.get(tok, 0.0)
            # Modulation: 1.0 + alpha * weight
            mods[i] = 1.0 + (0.2 * w)
        return mods
    
    def get_associated_tokens(self, token_id: int, top_k: int = 10) -> List[tuple]:
        """Get top-k tokens most associated with given token"""
        associations = []
        for (t1, t2), strength in self.associations.items():
            if t1 == token_id:
                associations.append((t2, strength))
            elif t2 == token_id:
                associations.append((t1, strength))
        
        associations.sort(key=lambda x: x[1], reverse=True)
        return associations[:top_k]
    
    def save_state(self, path: str) -> None:
        """Save learner state to file"""
        state = {
            'token_weights': self.token_weights,
            'associations': {f"{k[0]}:{k[1]}": v for k, v in self.associations.items()},
            'stats': self.stats
        }
        with open(path, 'w') as f:
            json.dump(state, f)
        logger.info(f"STDP state saved to {path}")
    
    def load_state(self, path: str) -> None:
        """Load learner state from file"""
        if not Path(path).exists():
            logger.warning(f"STDP state file not found: {path}")
            return
        
        with open(path, 'r') as f:
            state = json.load(f)
        
        self.token_weights = {int(k): v for k, v in state.get('token_weights', {}).items()}
        
        # Reconstruct associations with tuple keys
        self.associations = {}
        for k, v in state.get('associations', {}).items():
            t1, t2 = k.split(':')
            self.associations[(int(t1), int(t2))] = v
        
        self.stats = state.get('stats', self.stats)
        logger.info(f"STDP state loaded from {path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learner statistics"""
        return {
            **self.stats,
            'total_weight': sum(self.token_weights.values()),
            'avg_weight': np.mean(list(self.token_weights.values())) if self.token_weights else 0
        }

