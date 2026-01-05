"""
Multimodal Encoder - Updated for existing GrillCheese architecture
Integrates with your Vulkan FAISS backend for GPU-accelerated retrieval
"""

import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Union, Dict, Optional
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class MultimodalEncoder:
    """Unified encoder for text, images, and audio using ONNX models"""
    
    def __init__(self, models_dir: Path, device: str = "auto"):
        self.models_dir = Path(models_dir)
        self.device = device
        
        # Match your existing memory_store embedding dimension
        self.embedding_dim = 384  # From config.py MemoryConfig.EMBEDDING_DIM
        
        self.text_session = None
        self.image_session = None
        self.audio_session = None
        
        self._init_providers()
        
    def _init_providers(self):
        """Initialize ONNX execution providers for AMD GPU"""
        if self.device == "auto":
            self.providers = [
                'DmlExecutionProvider',  # DirectML for Windows AMD
                'CPUExecutionProvider'
            ]
        elif self.device == "gpu":
            self.providers = ['DmlExecutionProvider']
        else:
            self.providers = ['CPUExecutionProvider']
            
    def load_text_encoder(self, model_path: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Load text encoder - use sentence-transformers compatible model
        
        Options:
        - all-MiniLM-L6-v2: 384-dim (MATCHES YOUR EXISTING CONFIG)
        - LaBSE: 768-dim (requires config change)
        """
        try:
            from sentence_transformers import SentenceTransformer
            
            # Load sentence-transformers model (auto-downloads)
            self.text_model = SentenceTransformer(model_path)
            logger.info(f"✓ Loaded text encoder: {model_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load text encoder: {e}")
            return False
            
    def encode_text(self, text: str, language: str = None) -> np.ndarray:
        """
        Encode text to embedding vector
        
        Returns:
            384-dim vector (matches your MemoryConfig.EMBEDDING_DIM)
        """
        if not hasattr(self, 'text_model'):
            raise RuntimeError("Text encoder not loaded")
            
        # Generate embedding
        embedding = self.text_model.encode(text, convert_to_numpy=True)
        
        # Ensure correct dimension
        if len(embedding) != self.embedding_dim:
            logger.warning(f"Embedding dimension mismatch: {len(embedding)} != {self.embedding_dim}")
            if len(embedding) > self.embedding_dim:
                embedding = embedding[:self.embedding_dim]
            else:
                padded = np.zeros(self.embedding_dim, dtype=np.float32)
                padded[:len(embedding)] = embedding
                embedding = padded
                
        return embedding.astype(np.float32)
        
    def encode_multimodal(self, 
                         text: str = None,
                         image: Union[str, Path, Image.Image] = None,
                         audio: Union[str, Path, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Encode multiple modalities"""
        embeddings = {}
        
        if text is not None and hasattr(self, 'text_model'):
            embeddings['text'] = self.encode_text(text)
            
        # Image and audio encoders can be added later
        # For now, just return text embeddings
        
        return embeddings
        
    def fuse_embeddings(self, embeddings: Dict[str, np.ndarray], 
                       weights: Dict[str, float] = None) -> np.ndarray:
        """Fuse multiple modality embeddings"""
        if not embeddings:
            raise ValueError("No embeddings to fuse")
            
        if weights is None:
            weights = {k: 1.0 for k in embeddings.keys()}
            
        # Weighted average
        fused = np.zeros(self.embedding_dim, dtype=np.float32)
        total_weight = 0.0
        
        for modality, embedding in embeddings.items():
            weight = weights.get(modality, 1.0)
            fused += embedding * weight
            total_weight += weight
            
        fused /= total_weight
        
        # L2 normalize (important for cosine similarity in your FAISS shaders!)
        norm = np.linalg.norm(fused)
        if norm > 0:
            fused /= norm
            
        return fused

    def load_image_encoder(self):
        from transformers import CLIPModel, CLIPProcessor
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def encode_image(self, image: Image.Image) -> np.ndarray:
        inputs = self.clip_processor(images=image, return_tensors="pt")
        outputs = self.clip_model.get_image_features(**inputs)
        embedding = outputs[0].detach().numpy()
        
        # Project to 384-dim
        return self._project_to_384(embedding)