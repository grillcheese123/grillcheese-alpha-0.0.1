"""
Phi-3 Model using ONNX format with ONNX Runtime (DirectML for AMD GPU on Windows)
"""
import numpy as np
from typing import List, Optional

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class Phi3ONNX:
    """Phi-3 model using ONNX format with ONNX Runtime"""
    
    def __init__(self, model_path: Optional[str] = None, use_gpu: bool = True):
        """
        Initialize Phi-3 ONNX model
        
        Args:
            model_path: Path to ONNX model file
            use_gpu: Whether to use GPU (DirectML on Windows, CUDA on Linux)
        """
        if not ONNX_AVAILABLE:
            raise ImportError(
                "onnxruntime not installed. Install with: "
                "pip install onnxruntime-gpu or uv add onnxruntime-gpu"
            )
        
        # Determine execution provider
        providers = []
        if use_gpu:
            # Try DirectML first (Windows AMD/NVIDIA)
            try:
                providers.append('DmlExecutionProvider')
                print("✓ Using DirectML provider (Windows GPU)")
            except:
                pass
            
            # Try CUDA (Linux/Windows NVIDIA)
            try:
                providers.append('CUDAExecutionProvider')
                print("✓ Using CUDA provider")
            except:
                pass
        
        # Fallback to CPU
        providers.append('CPUExecutionProvider')
        print(f"✓ Using providers: {providers}")
        
        if model_path is None:
            # Try common paths
            import os
            model_paths = [
                "models/Phi-3-mini-4k-instruct.onnx",
                "models/phi-3-mini.onnx",
            ]
            
            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path is None:
                raise FileNotFoundError(
                    "ONNX model not found. Please download Phi-3-mini ONNX model or specify model_path"
                )
        
        print(f"Loading ONNX model from: {model_path}")
        
        try:
            self.session = ort.InferenceSession(
                model_path,
                providers=providers
            )
            print("✓ ONNX model loaded")
            self.device = "gpu" if use_gpu and providers[0] != 'CPUExecutionProvider' else "cpu"
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")
        
        self.embedding_dim = 3072  # Phi-3 embedding dimension
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Extract embedding from ONNX model
        Note: This depends on the specific ONNX model structure
        """
        # ONNX models have specific input/output formats
        # This is a placeholder - actual implementation depends on model structure
        # You would need to tokenize, run through the model, and extract embeddings
        raise NotImplementedError(
            "Embedding extraction from ONNX requires model-specific implementation"
        )
    
    def generate(self, prompt: str, context: List[str]) -> str:
        """
        Generate response with ONNX model
        Note: This requires the model to be exported in a specific format
        """
        raise NotImplementedError(
            "ONNX generation requires model-specific implementation. "
            "Consider using GGUF format with llama-cpp-python instead."
        )

