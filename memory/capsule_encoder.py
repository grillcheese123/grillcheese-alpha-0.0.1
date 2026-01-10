"""
Capsule Encoder
Encodes text into 32D cognitive feature vectors
GPU-accelerated with Vulkan shaders
"""
import numpy as np
import logging
import struct
from typing import Dict, List, Optional

from config import ModelConfig

logger = logging.getLogger(__name__)

# Try to import Vulkan backend
try:
    from vulkan_backend import VulkanCompute
    from vulkan_backend.base import VULKAN_AVAILABLE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    if VULKAN_AVAILABLE:
        from vulkan import *
    VULKAN_AVAILABLE = VULKAN_AVAILABLE
except ImportError:
    VULKAN_AVAILABLE = False


class CapsuleEncoder:
    """
    Encodes text into 32D cognitive feature vectors.
    Bio-inspired representation combining semantic content with memory dynamics.
    """
    
    def __init__(self, base_encoder: Optional[str] = None, use_embeddings: bool = False, use_gpu: bool = True):
        """
        Initialize capsule encoder
        
        Args:
            base_encoder: Base encoder model name (defaults to Granite from config)
            use_embeddings: Whether to use sentence-transformers (requires USE_EMBEDDINGS=True)
            use_gpu: Whether to use GPU acceleration (requires Vulkan)
        """
        self.use_embeddings = use_embeddings
        self.bge_model = None
        self.use_gpu = use_gpu and VULKAN_AVAILABLE
        self.vulkan = None
        
        # Initialize GPU backend if available
        if self.use_gpu:
            try:
                self.vulkan = VulkanCompute()
                logger.info("GPU acceleration enabled for capsule encoder")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU: {e}, using CPU fallback")
                self.use_gpu = False
        
        if use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                # Use Granite from config by default
                encoder_name = base_encoder or ModelConfig.EMBEDDING_MODEL
                logger.info(f"Loading Granite encoder: {encoder_name}")
                self.bge_model = SentenceTransformer(encoder_name, device='cpu')
                logger.info("Granite encoder loaded successfully")
            except ImportError:
                logger.warning("sentence-transformers not available, using hash-based encoding")
                self.use_embeddings = False
            except Exception as e:
                logger.warning(f"Failed to load encoder: {e}, using hash-based encoding")
                self.use_embeddings = False
        
        # Projection: 384D → 32D (only used if embeddings enabled)
        # Initialize with Xavier to preserve variance (GPU-accelerated if available)
        # Shape: (384, 32) for semantic @ projection where semantic is (384,)
        if self.use_embeddings:
            if self.use_gpu:
                # GPU returns (output_dim, input_dim) = (32, 384), transpose to (384, 32)
                self.projection = self._xavier_init_gpu(384, 32).T
            else:
                self.projection = np.random.randn(384, 32).astype(np.float32) * np.sqrt(2.0 / 384)
        else:
            # For hash-based, we'll use a simpler approach
            self.projection = None
    
    def _xavier_init_gpu(self, input_dim: int, output_dim: int) -> np.ndarray:
        """GPU-accelerated Xavier initialization using GPU random number generation"""
        if not self.use_gpu or self.vulkan is None:
            # CPU fallback
            return np.random.randn(output_dim, input_dim).astype(np.float32) * np.sqrt(2.0 / input_dim)
        
        try:
            # Use GPU shader for Xavier initialization
            weights = self.vulkan.xavier_init(input_dim, output_dim, seed=42)
            logger.debug(f"GPU Xavier initialization completed: {output_dim}x{input_dim} weights")
            return weights
        except (KeyError, RuntimeError) as e:
            # Shader not compiled - provide helpful message
            error_msg = str(e)
            if 'fnn-xavier-init' in error_msg or 'not found' in error_msg.lower():
                logger.info("GPU Xavier init shader not compiled, using CPU fallback")
                logger.info("  To enable GPU: Run scripts/compile_shader.ps1 fnn-xavier-init")
            else:
                logger.warning(f"GPU Xavier init failed: {e}, using CPU fallback")
            return np.random.randn(output_dim, input_dim).astype(np.float32) * np.sqrt(2.0 / input_dim)
        except Exception as e:
            logger.warning(f"GPU Xavier init failed: {e}, using CPU fallback")
            return np.random.randn(output_dim, input_dim).astype(np.float32) * np.sqrt(2.0 / input_dim)
    
    def _hash_encode(self, text: str) -> np.ndarray:
        """Hash-based encoding fallback (384D → 32D)"""
        # Simple hash-based semantic representation
        embedding = np.zeros(384, dtype=np.float32)
        
        # Tokenize by words
        words = text.lower().split()
        for i, word in enumerate(words[:512]):
            # Hash word to embedding space
            word_hash = hash(word) % 384
            val = (hash(word) % 10000) / 10000.0
            embedding[word_hash] = (embedding[word_hash] + val) % 1.0
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm
        
        return embedding
    
    def encode(self, text: str, cognitive_features: Dict[str, float]) -> np.ndarray:
        """
        Encode text into 32D capsule vector.
        
        Args:
            text: Content to encode
            cognitive_features: dict with plasticity_gain, consolidation_priority, 
                              stability, stress_link
        
        Returns:
            32D numpy array combining semantic and cognitive features
        """
        # Semantic embedding: 384D
        if self.use_embeddings and self.bge_model is not None:
            try:
                semantic = self.bge_model.encode(text, normalize_embeddings=True)
            except Exception as e:
                logger.warning(f"Encoding failed: {e}, using hash fallback")
                semantic = self._hash_encode(text)
        else:
            semantic = self._hash_encode(text)
        
        # Project to 32D semantic space (GPU-accelerated if available)
        if self.projection is not None:
            if self.use_gpu and self.vulkan is not None:
                semantic_32d = self._project_gpu(semantic, self.projection)
            else:
                semantic_32d = semantic @ self.projection
        else:
            # Direct hash-based 32D projection
            semantic_32d = semantic[:32].copy()
            if len(semantic) > 32:
                # Fold remaining dimensions
                for i in range(32, len(semantic)):
                    semantic_32d[i % 32] += semantic[i] * 0.1
        
        # Normalize (GPU-accelerated if available)
        if self.use_gpu and self.vulkan is not None:
            semantic_32d = self._normalize_gpu(semantic_32d)
        else:
            semantic_32d = semantic_32d / (np.linalg.norm(semantic_32d) + 1e-8)
        
        # Ensure semantic_32d is exactly 32D
        if len(semantic_32d) < 32:
            # Pad with zeros if needed
            padded = np.zeros(32, dtype=np.float32)
            padded[:len(semantic_32d)] = semantic_32d
            semantic_32d = padded
        elif len(semantic_32d) > 32:
            # Truncate if needed
            semantic_32d = semantic_32d[:32]
        
        # Blend with cognitive features (last 4 dimensions reserved)
        capsule = np.zeros(32, dtype=np.float32)
        capsule[:28] = semantic_32d[:28] * 0.9  # Semantic content (90%)
        
        # Blend remaining semantic dimensions (28-31) into first 28 dims with cognitive modulation
        # This creates a richer semantic representation modulated by plasticity
        plasticity_mod = cognitive_features.get('plasticity_gain', 0.5)
        for i in range(min(4, len(semantic_32d) - 28)):
            if 28 + i < len(semantic_32d):
                # Distribute the remaining semantic info across the first 28 dims
                blend_idx = i % 28
                capsule[blend_idx] += semantic_32d[28 + i] * 0.1 * plasticity_mod
        
        # Direct cognitive feature injection (last 4 dims)
        capsule[28] = cognitive_features.get('plasticity_gain', 0.5)
        capsule[29] = cognitive_features.get('consolidation_priority', 0.5)
        capsule[30] = cognitive_features.get('stability', 0.5)
        capsule[31] = cognitive_features.get('stress_link', 0.0)
        
        # Final normalization (GPU-accelerated if available)
        if self.use_gpu and self.vulkan is not None:
            return self._normalize_gpu(capsule)
        else:
            return capsule / (np.linalg.norm(capsule) + 1e-8)
    
    def _project_gpu(self, input_vec: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        GPU-accelerated matrix multiplication: input @ weights
        where input is (384,) and weights is (384, 32), result is (32,)
        """
        if not self.use_gpu or self.vulkan is None:
            return input_vec @ weights
        
        try:
            # Ensure weights are in correct shape (384, 32) for semantic @ weights
            if weights.shape != (384, 32):
                # If weights are transposed (32, 384), transpose them
                if weights.shape == (32, 384):
                    weights = weights.T
                else:
                    raise ValueError(f"Unexpected weights shape: {weights.shape}, expected (384, 32)")
            
            # Input: (384,) -> flattened for shader as (1, 384)
            # Shader expects: weights W[col][k] where col=output_dim, k=input_dim
            # For semantic @ weights where semantic is (384,) and weights is (384, 32):
            #   result[col] = sum(semantic[k] * weights[k][col])
            # Shader computes: output[row][col] = sum(input[row][k] * W[col][k])
            # So we need: W[col][k] = weights[k][col], i.e., W = weights.T
            # Transpose weights to (32, 384) for shader
            weights_for_shader = weights.T  # (32, 384)
            
            input_flat = input_vec.astype(np.float32).flatten()
            weights_flat = weights_for_shader.astype(np.float32).flatten()  # Row-major: (32*384,)
            output_flat = np.zeros(32, dtype=np.float32)
            
            # Create buffers
            buf_input, mem_input = self.vulkan.create_buffer(input_flat.nbytes, 'storage')
            buf_weights, mem_weights = self.vulkan.create_buffer(weights_flat.nbytes, 'storage')
            buf_output, mem_output = self.vulkan.create_buffer(output_flat.nbytes, 'storage')
            
            # Upload data
            self.vulkan.upload_buffer(buf_input, mem_input, input_flat)
            self.vulkan.upload_buffer(buf_weights, mem_weights, weights_flat)
            
            # Get pipeline for fnn-linear shader
            pipeline, pipeline_layout, desc_layout = self.vulkan.pipelines.get_or_create_pipeline(
                'fnn-linear', 4, push_constant_size=16
            )
            
            # Create descriptor set (fnn-linear needs 4 buffers: input, weights, bias, output)
            # Create dummy bias buffer (zeros, since has_bias=0)
            bias_flat = np.zeros(32, dtype=np.float32)
            buf_bias, mem_bias = self.vulkan.create_buffer(bias_flat.nbytes, 'storage')
            self.vulkan.upload_buffer(buf_bias, mem_bias, bias_flat)
            
            descriptor_set = self.vulkan.pipelines.get_cached_descriptor_set(
                'fnn-linear',
                [
                    (buf_input, input_flat.nbytes),
                    (buf_weights, weights_flat.nbytes),
                    (buf_bias, bias_flat.nbytes),
                    (buf_output, output_flat.nbytes)
                ]
            )
            
            # Pack push constants: batch_seq=1, input_dim=384, output_dim=32, has_bias=0
            push_constants = struct.pack('IIII', 1, 384, 32, 0)
            
            # Dispatch: 1x32 workgroups (one per output dimension)
            # _dispatch_compute expects separate integer arguments, not a tuple
            self.vulkan.core._dispatch_compute(
                pipeline, pipeline_layout, descriptor_set,
                workgroup_x=32, workgroup_y=1, workgroup_z=1,
                push_constants=push_constants
            )
            
            # Download result
            result = self.vulkan.read_buffer(mem_output, output_flat.nbytes, dtype=np.float32)
            result = result[:32]
            
            # Cleanup
            vkDestroyBuffer(self.vulkan.device, buf_input, None)
            vkDestroyBuffer(self.vulkan.device, buf_weights, None)
            vkDestroyBuffer(self.vulkan.device, buf_bias, None)
            vkDestroyBuffer(self.vulkan.device, buf_output, None)
            vkFreeMemory(self.vulkan.device, mem_input, None)
            vkFreeMemory(self.vulkan.device, mem_weights, None)
            vkFreeMemory(self.vulkan.device, mem_bias, None)
            vkFreeMemory(self.vulkan.device, mem_output, None)
            
            return result
        except Exception as e:
            logger.warning(f"GPU projection failed: {e}, using CPU fallback")
            return input_vec @ weights
    
    def _normalize_gpu(self, vec: np.ndarray) -> np.ndarray:
        """GPU-accelerated L2 normalization"""
        if not self.use_gpu or self.vulkan is None:
            norm = np.linalg.norm(vec)
            return vec / (norm + 1e-8) if norm > 1e-8 else vec
        
        try:
            vec_flat = vec.astype(np.float32).flatten()
            output_flat = np.zeros_like(vec_flat)
            
            # Compute norm on GPU (simple element-wise operations)
            # For now, use CPU norm (fast for small vectors)
            # In future, could use reduction shader
            norm = np.linalg.norm(vec_flat)
            if norm > 1e-8:
                output_flat = vec_flat / norm
            else:
                output_flat = vec_flat
            
            return output_flat.reshape(vec.shape)
        except Exception as e:
            logger.warning(f"GPU normalization failed: {e}, using CPU fallback")
            norm = np.linalg.norm(vec)
            return vec / (norm + 1e-8) if norm > 1e-8 else vec
    
    def encode_batch(self, texts: List[str], features: List[Dict[str, float]]) -> np.ndarray:
        """Batch encoding for efficiency."""
        capsules = []
        
        for text, feat in zip(texts, features):
            capsule = self.encode(text, feat)
            capsules.append(capsule)
        
        return np.array(capsules)
