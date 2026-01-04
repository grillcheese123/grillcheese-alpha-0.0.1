"""
GPU-accelerated sentence embeddings using Vulkan compute shaders
Uses existing shader infrastructure for full GPU pipeline
"""

import numpy as np
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from .embedder_config import EmbedderConfig

logger = logging.getLogger(__name__)

class VulkanEmbedder:
    """
    GPU-accelerated sentence embeddings using Vulkan compute shaders.
    Uses existing shader infrastructure for full GPU pipeline.
    """
    
    def __init__(self, gpu, config: EmbedderConfig, model_path: Optional[Path] = None):
        """
        Initialize embedder
        
        Args:
            gpu: VulkanCompute instance
            config: EmbedderConfig
            model_path: Path to model weights (if loading pre-trained)
        """
        self.gpu = gpu
        self.config = config
        
        # Initialize tokenizer
        self.tokenizer = self._init_tokenizer()
        
        # Load or initialize model weights
        if model_path and Path(model_path).exists():
            self._load_weights(model_path)
            logger.info(f"✓ Loaded weights from {model_path}")
        else:
            self._init_random_weights()
            logger.warning("Using random weights (not trained)")
            
        logger.info(f"VulkanEmbedder: {config.num_layers} layers, {config.hidden_dim}D")
    
    def _init_tokenizer(self):
        """Initialize SentencePiece tokenizer"""
        try:
            import sentencepiece as spm
            
            sp_model_path = Path("models/embedder.sp")
            
            if sp_model_path.exists():
                sp = spm.SentencePieceProcessor(model_file=str(sp_model_path))
                logger.info(f"✓ Loaded SentencePiece: {sp_model_path}")
                return sp
            else:
                logger.warning(f"SentencePiece model not found: {sp_model_path}")
                logger.warning("Using basic whitespace tokenization")
                return None
                
        except ImportError:
            logger.warning("sentencepiece not installed, using basic tokenization")
            return None
    
    def _init_random_weights(self):
        """Initialize random weights for testing"""
        self.embedding_table = np.random.randn(
            self.config.vocab_size, self.config.hidden_dim
        ).astype(np.float32) * 0.02
        
        self.layer_weights = []
        for _ in range(self.config.num_layers):
            self.layer_weights.append({
                'attention_qkv': np.random.randn(
                    self.config.hidden_dim, 3 * self.config.hidden_dim
                ).astype(np.float32) * 0.02,
                'ffn_fc1': np.random.randn(
                    self.config.hidden_dim, self.config.ffn_dim
                ).astype(np.float32) * 0.02,
                'ffn_fc2': np.random.randn(
                    self.config.ffn_dim, self.config.hidden_dim
                ).astype(np.float32) * 0.02,
            })
    
    def _load_weights(self, model_path: Path):
        """Load pre-trained weights"""
        weights = np.load(model_path)
        
        self.embedding_table = weights['embedding_table']
        self.layer_weights = []
        
        for i in range(self.config.num_layers):
            self.layer_weights.append({
                'attention_qkv': weights[f'layer_{i}_attn_qkv'],
                'ffn_fc1': weights[f'layer_{i}_ffn_fc1'],
                'ffn_fc2': weights[f'layer_{i}_ffn_fc2'],
            })
    
    def tokenize(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tokenize texts into token IDs and attention masks
        
        Returns:
            Tuple of (token_ids, attention_mask)
        """
        batch_size = len(texts)
        
        if self.tokenizer is not None:
            token_ids = []
            for text in texts:
                ids = self.tokenizer.encode(text, out_type=int)
                if len(ids) > self.config.max_seq_len:
                    ids = ids[:self.config.max_seq_len]
                token_ids.append(ids)
        else:
            token_ids = []
            for text in texts:
                words = text.lower().split()[:self.config.max_seq_len]
                ids = [hash(w) % self.config.vocab_size for w in words]
                token_ids.append(ids)
        
        max_len = min(max(len(ids) for ids in token_ids), self.config.max_seq_len)
        
        padded_ids = np.zeros((batch_size, max_len), dtype=np.int32)
        attention_mask = np.zeros((batch_size, max_len), dtype=np.uint32)
        
        for i, ids in enumerate(token_ids):
            length = min(len(ids), max_len)
            padded_ids[i, :length] = ids[:length]
            attention_mask[i, :length] = 1
        
        return padded_ids, attention_mask
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to sentence embeddings using GPU pipeline
        
        Args:
            texts: List of text strings
            
        Returns:
            Embeddings array (batch_size, hidden_dim) float32
        """
        token_ids, attention_mask = self.tokenize(texts)
        batch_size, seq_len = token_ids.shape
        
        embeddings = self._embedding_lookup(token_ids)
        embeddings = self._add_positional_encoding(embeddings, batch_size, seq_len)
        
        for layer_idx in range(self.config.num_layers):
            embeddings = self._transformer_layer(
                embeddings, attention_mask, layer_idx, batch_size, seq_len
            )
        
        sentence_embs = self._mean_pool(embeddings, attention_mask, batch_size, seq_len)
        sentence_embs = self._l2_normalize(sentence_embs, batch_size)
        
        return sentence_embs
    
    def _embedding_lookup(self, token_ids: np.ndarray) -> np.ndarray:
        """Use existing embedding-lookup shader"""
        batch_size, seq_len = token_ids.shape
        
        token_ids_flat = token_ids.flatten().astype(np.int32)
        
        embeddings = self.gpu.embedding_lookup(
            token_ids_flat,
            self.embedding_table,
            self.config.vocab_size,
            self.config.hidden_dim
        )
        
        return embeddings.reshape(batch_size, seq_len, self.config.hidden_dim)
    
    def _add_positional_encoding(self, embeddings: np.ndarray, 
                                batch_size: int, seq_len: int) -> np.ndarray:
        """Add positional encoding using new shader"""
        embeddings_flat = embeddings.astype(np.float32).flatten()
        
        output = self.gpu.position_encoding(
            embeddings_flat,
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_dim=self.config.hidden_dim,
            max_position=self.config.max_seq_len
        )
        
        return output.reshape(batch_size, seq_len, self.config.hidden_dim)
    
    def _transformer_layer(self, embeddings: np.ndarray, attention_mask: np.ndarray, 
                          layer_idx: int, batch_size: int, seq_len: int) -> np.ndarray:
        """Simplified transformer layer using CPU for now"""
        weights = self.layer_weights[layer_idx]
        
        # Simplified: just FFN + residual for now
        # Full attention implementation would use your attention shaders
        x = embeddings.reshape(-1, self.config.hidden_dim)
        
        # FFN
        x = x @ weights['ffn_fc1']
        x = self._gelu(x)
        x = x @ weights['ffn_fc2']
        
        ffn_output = x.reshape(batch_size, seq_len, self.config.hidden_dim)
        
        # Residual
        return embeddings + ffn_output * 0.1  # Scale down for stability
    
    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU activation"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    def _mean_pool(self, embeddings: np.ndarray, attention_mask: np.ndarray,
                   batch_size: int, seq_len: int) -> np.ndarray:
        """Mean pooling using new shader"""
        embeddings_flat = embeddings.flatten().astype(np.float32)
        mask_flat = attention_mask.flatten().astype(np.uint32)
        
        pooled = self.gpu.mean_pooling(
            embeddings_flat,
            mask_flat,
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_dim=self.config.hidden_dim
        )
        
        return pooled.reshape(batch_size, self.config.hidden_dim)
    
    def _l2_normalize(self, embeddings: np.ndarray, batch_size: int) -> np.ndarray:
        """L2 normalization using new shader"""
        embeddings_flat = embeddings.flatten().astype(np.float32)
        
        normalized = self.gpu.l2_normalize(
            embeddings_flat,
            batch_size=batch_size,
            hidden_dim=self.config.hidden_dim
        )
        
        return normalized.reshape(batch_size, self.config.hidden_dim)
