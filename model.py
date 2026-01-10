"""
Phi-3 Model using PyTorch/Transformers
Provides embedding extraction and text generation
"""
import logging
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

from config import ModelConfig, LogConfig

# Configure logging
logging.basicConfig(level=LogConfig.LEVEL, format=LogConfig.FORMAT)
logger = logging.getLogger(__name__)


class Phi3Model:
    """
    Phi-3 language model for text generation and embedding extraction
    Uses HuggingFace Transformers with automatic device placement
    """
    
    def __init__(self):
        """Initialize Phi-3 model and tokenizer"""
        logger.info("Loading Phi-3 model...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Detect device
        self.device = self._detect_device()
        
        # Load model
        self.model = self._load_model()
        
        # Embedding dimension
        self.embedding_dim = ModelConfig.PHI3_EMBEDDING_DIM
    
    def _detect_device(self) -> str:
        """Detect available compute device"""
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"{LogConfig.CHECK} CUDA available: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            logger.info(f"{LogConfig.CHECK} MPS (Apple Silicon) available")
        else:
            # Check for Vulkan availability (for AMD GPUs)
            try:
                from vulkan_backend.base import VULKAN_AVAILABLE
                if VULKAN_AVAILABLE:
                    device = "cpu"  # PyTorch can't use Vulkan directly
                    logger.info(f"{LogConfig.WARNING} PyTorch model: No CUDA/MPS, using CPU")
                    logger.info("  Note: For GPU acceleration, use GGUF model (llama.cpp supports Vulkan)")
                else:
                    device = "cpu"
                    logger.info(f"{LogConfig.WARNING} No GPU detected, using CPU")
                    logger.info("  Note: AMD GPU support requires ROCm or Vulkan shaders")
            except ImportError:
                device = "cpu"
                logger.info(f"{LogConfig.WARNING} No GPU detected, using CPU")
                logger.info("  Note: AMD GPU support requires ROCm or Vulkan shaders")
        return device
    
    def _load_model(self):
        """Load the model with appropriate settings for the device"""
        try:
            if self.device == "cuda":
                model = AutoModelForCausalLM.from_pretrained(
                    "microsoft/Phi-3-mini-4k-instruct",
                    dtype=torch.float16,
                    device_map="auto"
                )
                logger.info(f"{LogConfig.CHECK} Model loaded on GPU")
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    "microsoft/Phi-3-mini-4k-instruct",
                    dtype=torch.float16,
                    device_map="cpu"
                )
                logger.info(f"{LogConfig.CHECK} Model loaded on CPU")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            if self.device != "cpu":
                logger.info("Falling back to CPU...")
                self.device = "cpu"
                return AutoModelForCausalLM.from_pretrained(
                    "microsoft/Phi-3-mini-4k-instruct",
                    dtype=torch.float16,
                    device_map="cpu"
                )
            raise
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Extract embedding from Phi-3 model
        
        Uses mean pooling over token embeddings for a fixed-size representation.
        
        Args:
            text: Input text
        
        Returns:
            Embedding vector (3072,) - mean pooled over sequence
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Get token embeddings from the embedding layer
            embeddings = self.model.model.embed_tokens(inputs['input_ids'])
            # Mean pool over sequence dimension
            embedding_tensor = embeddings.mean(dim=1).squeeze()
            
            # Convert to numpy with fallback for compatibility
            try:
                embedding = embedding_tensor.cpu().numpy().astype(np.float32)
            except (RuntimeError, AttributeError):
                embedding = np.array(embedding_tensor.cpu().tolist(), dtype=np.float32).flatten()
        
        return embedding
    
    def generate(self, prompt: str, context: List[str]) -> str:
        """
        Generate response with memory context
        
        Args:
            prompt: User prompt
            context: List of retrieved memory texts
        
        Returns:
            Generated response text
        """
        try:
            # Extract identity if present (should be first item)
            identity_text = None
            context_items = []
            
            if context:
                # First item is typically identity
                first_item = context[0]
                # Check if it looks like identity (contains "GrillCheese" or is longer/descriptive)
                if "GrillCheese" in first_item or len(first_item) > 200:
                    identity_text = first_item
                    context_items = context[1:ModelConfig.MAX_CONTEXT_ITEMS]
                else:
                    context_items = context[:ModelConfig.MAX_CONTEXT_ITEMS]
            
            # Build prompt with identity as system message
            parts = []
            
            # Add identity as system message if present
            if identity_text:
                parts.append(f"<|system|>\n{identity_text}<|end|>\n")
            
            # Add other context items (recent conversation history and semantic memories)
            if context_items:
                # Format context more naturally
                # Prioritize conversation history, then add semantic memories as optional background
                context_lines = []
                semantic_memories = []
                
                for c in context_items:
                    # Check if it's conversation history (starts with "Previous")
                    if c.startswith("Previous"):
                        context_lines.append(c)
                    else:
                        # Semantic memory - collect separately
                        semantic_memories.append(c)
                
                # Add conversation history first (most important for continuity)
                if context_lines:
                    context_text = "\n".join(context_lines)
                    parts.append(context_text)
                    parts.append("")
                
                # Add semantic memories as optional background (only if we have few enough)
                if semantic_memories and len(semantic_memories) <= 2:
                    # Only include 1-2 most relevant semantic memories to avoid confusion
                    background_text = "\n".join([f"Background: {m}" for m in semantic_memories[:2]])
                    parts.append(background_text)
                    parts.append("")
            
            # Add user prompt
            parts.append(f"<|user|>\n{prompt}<|end|>\n")
            parts.append("<|assistant|>\n")
            
            full_prompt = "\n".join(parts)
            
            # Tokenize
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                max_length=1024,
                truncation=True,
                padding=False
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Adjust settings based on device
            max_tokens = ModelConfig.MAX_NEW_TOKENS_GPU if self.device != "cpu" else ModelConfig.MAX_NEW_TOKENS_CPU
            do_sample = self.device != "cpu"
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'),
                    max_new_tokens=max_tokens,
                    do_sample=do_sample,
                    temperature=ModelConfig.TEMPERATURE if do_sample else None,
                    top_p=ModelConfig.TOP_P if do_sample else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Extract new tokens
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Clean up response
            response = response.strip()
            for stop_seq in ["\nUser:", "\n\nUser:"]:
                if stop_seq in response:
                    response = response.split(stop_seq)[0].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"[Error during generation: {str(e)}]"
