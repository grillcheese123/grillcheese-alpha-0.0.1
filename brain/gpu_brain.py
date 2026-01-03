"""
GPU-Accelerated Brain Components

Uses Vulkan compute shaders for high-performance brain operations:
- Place cells (spatial encoding)
- Time cells (temporal encoding)
- Theta-gamma encoding (phase coupling)
- Hebbian learning
- STDP learning
- Attention routing with prosody modulation
- Domain-based expert routing
"""
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger(__name__)

# Try to import Vulkan backend
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from vulkan_backend import VulkanCompute
    VULKAN_AVAILABLE = True
except ImportError:
    VULKAN_AVAILABLE = False
    logger.warning("Vulkan backend not available, using CPU fallback")


class GPUBrainCompute:
    """
    GPU-accelerated brain computations using Vulkan shaders
    
    Available shaders:
    - place-cell: Spatial place field encoding
    - time-cell: Temporal sequence encoding
    - theta-gamma-encoding: Phase-amplitude coupling
    - hebbian-learning: Hebbian weight updates
    - stdp-learning: STDP weight updates
    - attention-prosody-modulation: Emotional attention modulation
    - domain-router: Expert routing based on domain
    """
    
    def __init__(self, use_vulkan: bool = True):
        """
        Initialize GPU brain compute
        
        Args:
            use_vulkan: Whether to use Vulkan (falls back to CPU if unavailable)
        """
        self.use_vulkan = use_vulkan and VULKAN_AVAILABLE
        self.vulkan: Optional[VulkanCompute] = None
        
        if self.use_vulkan:
            try:
                self.vulkan = VulkanCompute()
                logger.info("[OK] GPU brain compute initialized with Vulkan")
            except Exception as e:
                logger.warning(f"Failed to initialize Vulkan: {e}")
                self.use_vulkan = False
        
        if not self.use_vulkan:
            logger.info("Using CPU fallback for brain computations")
        
        # Pre-allocated buffers for efficiency
        self._buffers: Dict[str, Any] = {}
    
    # ==================== Place Cells ====================
    
    def compute_place_cells(
        self,
        agent_position: np.ndarray,
        field_centers: np.ndarray,
        field_width: float = 1.0,
        max_rate: float = 20.0,
        baseline_rate: float = 0.1
    ) -> np.ndarray:
        """
        Compute place cell firing rates based on agent position
        
        Uses shader: place-cell.glsl
        
        Args:
            agent_position: Current position [x, y] or [x, y, z]
            field_centers: Place field centers [n_neurons, spatial_dims]
            field_width: Width of place fields
            max_rate: Maximum firing rate
            baseline_rate: Baseline firing rate
            
        Returns:
            Firing rates for each place cell [n_neurons]
        """
        n_neurons = field_centers.shape[0]
        spatial_dims = field_centers.shape[1]
        
        if self.use_vulkan and self.vulkan:
            try:
                # Use public place_cell method
                return self.vulkan.place_cell(
                    agent_position=agent_position.astype(np.float32),
                    field_centers=field_centers.astype(np.float32),
                    field_width=field_width,
                    max_rate=max_rate,
                    baseline_rate=baseline_rate,
                    spatial_dims=spatial_dims
                )
            except Exception as e:
                logger.warning(f"GPU place cell computation failed: {e}")
        
        # CPU fallback
        return self._place_cells_cpu(
            agent_position, field_centers, field_width, max_rate, baseline_rate
        )
    
    def _place_cells_cpu(
        self,
        agent_position: np.ndarray,
        field_centers: np.ndarray,
        field_width: float,
        max_rate: float,
        baseline_rate: float
    ) -> np.ndarray:
        """CPU fallback for place cell computation"""
        # Compute distances
        diffs = field_centers - agent_position
        dist_sq = np.sum(diffs ** 2, axis=1)
        
        # Gaussian tuning curve
        sigma_sq = field_width ** 2
        gaussian = np.exp(-dist_sq / (2.0 * sigma_sq))
        
        # Compute firing rates
        rates = baseline_rate + (max_rate - baseline_rate) * gaussian
        
        return rates.astype(np.float32)
    
    # ==================== Time Cells ====================
    
    def compute_time_cells(
        self,
        current_time: float,
        preferred_times: np.ndarray,
        temporal_width: float = 1.0,
        max_rate: float = 15.0,
        baseline_rate: float = 0.1,
        use_dynamics: bool = False,
        membrane_state: Optional[np.ndarray] = None,
        dt: float = 0.01,
        tau_adaptation: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute time cell firing rates based on elapsed time
        
        Uses shader: time-cell.glsl
        
        Args:
            current_time: Current time in sequence
            preferred_times: Preferred firing times per neuron [n_neurons]
            temporal_width: Width of temporal receptive field
            max_rate: Maximum firing rate
            baseline_rate: Baseline firing rate
            use_dynamics: Use dynamic temporal model
            membrane_state: Previous membrane state (for dynamics)
            dt: Time step
            tau_adaptation: Adaptation time constant
            
        Returns:
            Tuple of (firing_rates, updated_membrane_state)
        """
        n_neurons = len(preferred_times)
        
        if membrane_state is None:
            membrane_state = np.zeros(n_neurons, dtype=np.float32)
        
        if self.use_vulkan and self.vulkan:
            try:
                # Use public time_cell method
                rates = self.vulkan.time_cell(
                    current_time=current_time,
                    preferred_times=preferred_times.astype(np.float32),
                    time_constant=temporal_width,
                    max_rate=max_rate,
                    baseline_rate=baseline_rate
                )
                # Return rates and updated membrane (simple decay for CPU fallback)
                new_mem = membrane_state * 0.95 + rates * 0.05
                return rates, new_mem
            except Exception as e:
                logger.warning(f"GPU time cell computation failed: {e}")
        
        # CPU fallback
        return self._time_cells_cpu(
            current_time, preferred_times, temporal_width,
            max_rate, baseline_rate, membrane_state
        )
    
    def _time_cells_cpu(
        self,
        current_time: float,
        preferred_times: np.ndarray,
        temporal_width: float,
        max_rate: float,
        baseline_rate: float,
        membrane_state: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """CPU fallback for time cell computation"""
        t_diff = current_time - preferred_times
        sigma_sq = temporal_width ** 2
        gaussian = np.exp(-t_diff ** 2 / (2.0 * sigma_sq))
        rates = baseline_rate + (max_rate - baseline_rate) * gaussian
        
        return rates.astype(np.float32), membrane_state
    
    # ==================== Theta-Gamma Encoding ====================
    
    def compute_theta_gamma_encoding(
        self,
        positions: np.ndarray,
        embedding_dim: int,
        theta_freq: float = 8.0,
        gamma_freq: float = 40.0,
        theta_phase_offsets: Optional[np.ndarray] = None,
        gamma_phase_offsets: Optional[np.ndarray] = None,
        amplitude_mod: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute theta-gamma phase-coupled positional encoding
        
        Uses shader: theta-gamma-encoding.glsl
        
        This implements biological phase-amplitude coupling where
        gamma oscillations are modulated by theta phase.
        
        Args:
            positions: Position indices [batch, seq_len]
            embedding_dim: Output embedding dimension
            theta_freq: Theta oscillation frequency
            gamma_freq: Gamma oscillation frequency
            theta_phase_offsets: Learnable theta offsets [embedding_dim]
            gamma_phase_offsets: Learnable gamma offsets [embedding_dim]
            amplitude_mod: Amplitude modulation [embedding_dim]
            
        Returns:
            Encoded positions [batch, seq_len, embedding_dim]
        """
        batch_size = positions.shape[0]
        seq_len = positions.shape[1]
        
        # Initialize offsets if not provided
        if theta_phase_offsets is None:
            theta_phase_offsets = np.zeros(embedding_dim, dtype=np.float32)
        if gamma_phase_offsets is None:
            gamma_phase_offsets = np.zeros(embedding_dim, dtype=np.float32)
        if amplitude_mod is None:
            amplitude_mod = np.ones(embedding_dim, dtype=np.float32)
        
        if self.use_vulkan and self.vulkan:
            try:
                total_elements = batch_size * seq_len * embedding_dim
                
                pos_buf = self.vulkan.create_buffer(
                    positions.flatten().astype(np.float32).tobytes(),
                    usage='storage'
                )
                out_buf = self.vulkan.create_buffer(
                    total_elements * 4,
                    usage='storage'
                )
                theta_buf = self.vulkan.create_buffer(
                    theta_phase_offsets.astype(np.float32).tobytes(),
                    usage='storage'
                )
                gamma_buf = self.vulkan.create_buffer(
                    gamma_phase_offsets.astype(np.float32).tobytes(),
                    usage='storage'
                )
                amp_buf = self.vulkan.create_buffer(
                    amplitude_mod.astype(np.float32).tobytes(),
                    usage='storage'
                )
                
                pipeline = self.vulkan.create_pipeline('theta-gamma-encoding')
                
                max_seq_len = max(seq_len, 512)
                push_constants = np.array([
                    batch_size, seq_len, embedding_dim,
                    theta_freq, gamma_freq, max_seq_len
                ], dtype=np.float32)
                
                self.vulkan.dispatch(
                    pipeline,
                    [pos_buf, out_buf, theta_buf, gamma_buf, amp_buf],
                    push_constants.tobytes(),
                    groups=((total_elements + 255) // 256, 1, 1)
                )
                
                result = np.frombuffer(
                    self.vulkan.read_buffer(out_buf),
                    dtype=np.float32
                ).reshape(batch_size, seq_len, embedding_dim)
                
                return result
                
            except Exception as e:
                logger.warning(f"GPU theta-gamma encoding failed: {e}")
        
        # CPU fallback
        return self._theta_gamma_cpu(
            positions, embedding_dim, theta_freq, gamma_freq,
            theta_phase_offsets, gamma_phase_offsets, amplitude_mod
        )
    
    def _theta_gamma_cpu(
        self,
        positions: np.ndarray,
        embedding_dim: int,
        theta_freq: float,
        gamma_freq: float,
        theta_phase_offsets: np.ndarray,
        gamma_phase_offsets: np.ndarray,
        amplitude_mod: np.ndarray
    ) -> np.ndarray:
        """CPU fallback for theta-gamma encoding"""
        batch_size, seq_len = positions.shape
        output = np.zeros((batch_size, seq_len, embedding_dim), dtype=np.float32)
        
        max_pos = max(seq_len - 1, 1)
        
        for b in range(batch_size):
            for s in range(seq_len):
                pos = positions[b, s]
                normalized_pos = (pos / max_pos) * 2 * np.pi
                
                for d in range(embedding_dim):
                    theta_phase = normalized_pos + theta_phase_offsets[d]
                    theta_encoding = np.sin(theta_phase)
                    
                    freq_ratio = gamma_freq / theta_freq
                    gamma_phase = (normalized_pos * freq_ratio) + gamma_phase_offsets[d]
                    
                    gamma_amplitude = (np.cos(theta_phase) + 1.0) * 0.5
                    gamma_encoding = gamma_amplitude * np.sin(gamma_phase)
                    
                    combined = (theta_encoding + 0.5 * gamma_encoding) * amplitude_mod[d]
                    output[b, s, d] = combined
        
        return output
    
    # ==================== Hebbian Learning ====================
    
    def hebbian_update(
        self,
        pre_activations: np.ndarray,
        post_activations: np.ndarray,
        weights: np.ndarray,
        learning_rate: float = 0.01,
        weight_decay: float = 0.001
    ) -> np.ndarray:
        """
        Perform Hebbian weight update
        
        Uses shader: hebbian-learning.glsl
        
        Implements: ΔW = η * <pre * post> - λ * W
        
        Args:
            pre_activations: Pre-synaptic activations [batch, time, pre_dim]
            post_activations: Post-synaptic activations [batch, time, post_dim]
            weights: Current weights [post_dim, pre_dim]
            learning_rate: Learning rate η
            weight_decay: Weight decay λ
            
        Returns:
            Updated weights
        """
        batch_size = pre_activations.shape[0]
        time_steps = pre_activations.shape[1]
        pre_dim = pre_activations.shape[2]
        post_dim = post_activations.shape[2]
        
        if self.use_vulkan and self.vulkan:
            try:
                pre_buf = self.vulkan.create_buffer(
                    pre_activations.astype(np.float32).tobytes(),
                    usage='storage'
                )
                post_buf = self.vulkan.create_buffer(
                    post_activations.astype(np.float32).tobytes(),
                    usage='storage'
                )
                w_buf = self.vulkan.create_buffer(
                    weights.astype(np.float32).tobytes(),
                    usage='storage'
                )
                
                pipeline = self.vulkan.create_pipeline('hebbian-learning')
                
                push_constants = np.array([
                    batch_size, time_steps, pre_dim, post_dim,
                    learning_rate, weight_decay
                ], dtype=np.float32)
                
                self.vulkan.dispatch(
                    pipeline,
                    [pre_buf, post_buf, w_buf],
                    push_constants.tobytes(),
                    groups=((pre_dim + 15) // 16, (post_dim + 15) // 16, 1)
                )
                
                updated = np.frombuffer(
                    self.vulkan.read_buffer(w_buf),
                    dtype=np.float32
                ).reshape(post_dim, pre_dim)
                
                return updated
                
            except Exception as e:
                logger.warning(f"GPU Hebbian update failed: {e}")
        
        # CPU fallback
        return self._hebbian_cpu(
            pre_activations, post_activations, weights,
            learning_rate, weight_decay
        )
    
    def _hebbian_cpu(
        self,
        pre_activations: np.ndarray,
        post_activations: np.ndarray,
        weights: np.ndarray,
        learning_rate: float,
        weight_decay: float
    ) -> np.ndarray:
        """CPU fallback for Hebbian learning"""
        # Average over batch and time
        correlation = np.einsum('bti,btj->ij', post_activations, pre_activations)
        correlation /= (pre_activations.shape[0] * pre_activations.shape[1])
        
        # Update
        delta_w = learning_rate * correlation - weight_decay * weights
        
        return (weights + delta_w).astype(np.float32)
    
    # ==================== STDP Learning ====================
    
    def stdp_update(
        self,
        pre_activations: np.ndarray,
        post_activations: np.ndarray,
        weights: np.ndarray,
        pre_trace: np.ndarray,
        post_trace: np.ndarray,
        lr_potentiation: float = 0.01,
        lr_depression: float = 0.012,
        trace_decay: float = 0.9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform STDP weight update
        
        Uses shader: stdp-learning.glsl
        
        Args:
            pre_activations: [batch, time, pre_dim]
            post_activations: [batch, time, post_dim]
            weights: [post_dim, pre_dim]
            pre_trace: Eligibility traces [batch, pre_dim]
            post_trace: Eligibility traces [batch, post_dim]
            lr_potentiation: LTP learning rate
            lr_depression: LTD learning rate
            trace_decay: Trace decay rate
            
        Returns:
            Tuple of (updated_weights, updated_pre_trace, updated_post_trace)
        """
        batch_size = pre_activations.shape[0]
        time_steps = pre_activations.shape[1]
        pre_dim = pre_activations.shape[2]
        post_dim = post_activations.shape[2]
        
        # CPU implementation (STDP requires two passes)
        # Update traces
        avg_pre = pre_activations.mean(axis=1)  # [batch, pre_dim]
        avg_post = post_activations.mean(axis=1)  # [batch, post_dim]
        
        new_pre_trace = trace_decay * pre_trace + (1 - trace_decay) * avg_pre
        new_post_trace = trace_decay * post_trace + (1 - trace_decay) * avg_post
        
        # Compute STDP updates
        ltp = np.einsum('bi,bj->ij', new_post_trace, new_pre_trace) / batch_size
        ltd = np.einsum('bj,bi->ij', new_pre_trace, new_post_trace) / batch_size
        
        delta_w = lr_potentiation * ltp - lr_depression * ltd
        updated_weights = weights + delta_w
        
        return updated_weights.astype(np.float32), new_pre_trace.astype(np.float32), new_post_trace.astype(np.float32)
    
    # ==================== Domain Routing ====================
    
    def domain_routing(
        self,
        domain_probs: np.ndarray,
        expert_weights: np.ndarray,
        top_k: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Route to experts based on domain probabilities
        
        Uses shader: domain-router.glsl
        
        Args:
            domain_probs: Domain probabilities [batch, num_domains]
            expert_weights: Expert weights per domain [num_domains, num_experts]
            top_k: Number of experts to select
            
        Returns:
            Tuple of (routing_weights, selected_expert_indices)
        """
        batch_size = domain_probs.shape[0]
        num_domains = domain_probs.shape[1]
        num_experts = expert_weights.shape[1]
        
        # Compute routing weights
        routing_weights = domain_probs @ expert_weights  # [batch, num_experts]
        
        # Select top-k experts
        expert_indices = np.argsort(routing_weights, axis=1)[:, -top_k:][:, ::-1]
        
        return routing_weights.astype(np.float32), expert_indices.astype(np.uint32)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get GPU brain compute statistics"""
        return {
            'vulkan_available': VULKAN_AVAILABLE,
            'using_vulkan': self.use_vulkan,
            'vulkan_initialized': self.vulkan is not None
        }


class GPUSpatialMemory:
    """
    GPU-accelerated spatial memory system using place and time cells
    """
    
    def __init__(
        self,
        n_place_cells: int = 1000,
        n_time_cells: int = 100,
        spatial_dims: int = 2,
        use_vulkan: bool = True
    ):
        self.gpu = GPUBrainCompute(use_vulkan=use_vulkan)
        self.n_place_cells = n_place_cells
        self.n_time_cells = n_time_cells
        self.spatial_dims = spatial_dims
        
        # Initialize place field centers (uniformly distributed)
        self.place_centers = np.random.uniform(
            -10, 10, (n_place_cells, spatial_dims)
        ).astype(np.float32)
        
        # Initialize time cell preferences (log-spaced)
        self.time_preferences = np.logspace(
            0, 3, n_time_cells
        ).astype(np.float32)
        
        # Current state
        self.position = np.zeros(spatial_dims, dtype=np.float32)
        self.elapsed_time = 0.0
        self.time_cell_state = np.zeros(n_time_cells, dtype=np.float32)
    
    def update_position(self, new_position: np.ndarray) -> np.ndarray:
        """Update position and return place cell activations"""
        self.position = new_position.astype(np.float32)
        
        return self.gpu.compute_place_cells(
            self.position,
            self.place_centers,
            field_width=2.0,
            max_rate=20.0
        )
    
    def update_time(self, dt: float) -> np.ndarray:
        """Update time and return time cell activations"""
        self.elapsed_time += dt
        
        rates, self.time_cell_state = self.gpu.compute_time_cells(
            self.elapsed_time,
            self.time_preferences,
            temporal_width=0.3,
            use_dynamics=True,
            membrane_state=self.time_cell_state,
            dt=dt
        )
        
        return rates
    
    def reset_time(self):
        """Reset time (e.g., at start of new event)"""
        self.elapsed_time = 0.0
        self.time_cell_state = np.zeros(self.n_time_cells, dtype=np.float32)
    
    def get_spatial_context(self) -> Dict[str, np.ndarray]:
        """Get current spatial-temporal context"""
        place_rates = self.gpu.compute_place_cells(
            self.position, self.place_centers
        )
        time_rates, _ = self.gpu.compute_time_cells(
            self.elapsed_time, self.time_preferences
        )
        
        return {
            'position': self.position,
            'place_cells': place_rates,
            'time_cells': time_rates,
            'elapsed_time': self.elapsed_time
        }

