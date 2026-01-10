"""
SentencePiece Tokenizer for GrillCheese Vulkan Embeddings.

Provides a lightweight tokenizer that doesn't require PyTorch.
Compatible with common embedding models.
"""
import logging
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

# Try to import sentencepiece
try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False


class SentencePieceTokenizer:
    """
    SentencePiece tokenizer wrapper for embedding models.
    
    Supports:
    - BPE (Byte-Pair Encoding)
    - Unigram
    - Char
    - Word
    """
    
    # Special token IDs (standard for most models)
    PAD_ID = 0
    UNK_ID = 1  # Unknown token
    BOS_ID = 2  # Beginning of sentence
    EOS_ID = 3  # End of sentence
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        vocab_size: int = 32000,
        max_length: int = 512,
        add_bos: bool = False,
        add_eos: bool = False
    ):
        """
        Initialize tokenizer.
        
        Args:
            model_path: Path to .model file. If None, uses a fallback tokenizer.
            vocab_size: Vocabulary size (used if no model loaded)
            max_length: Maximum sequence length
            add_bos: Add beginning-of-sentence token
            add_eos: Add end-of-sentence token
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.sp = None
        
        if model_path and SENTENCEPIECE_AVAILABLE:
            self._load_model(model_path)
        elif model_path and not SENTENCEPIECE_AVAILABLE:
            logger.warning("sentencepiece not installed, using fallback tokenizer")
            self._init_fallback()
        else:
            self._init_fallback()
    
    def _load_model(self, model_path: str):
        """Load SentencePiece model"""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer model not found: {model_path}")
        
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(path))
        self.vocab_size = self.sp.GetPieceSize()
        
        # Get special token IDs from model
        self.PAD_ID = self.sp.pad_id() if self.sp.pad_id() >= 0 else 0
        self.UNK_ID = self.sp.unk_id() if self.sp.unk_id() >= 0 else 1
        self.BOS_ID = self.sp.bos_id() if self.sp.bos_id() >= 0 else 2
        self.EOS_ID = self.sp.eos_id() if self.sp.eos_id() >= 0 else 3
        
        logger.info(f"Loaded SentencePiece model: vocab_size={self.vocab_size}")
    
    def _init_fallback(self):
        """Initialize simple character-level fallback tokenizer"""
        logger.info("Using fallback character tokenizer")
        self.sp = None
        # Character-level with special tokens
        # Reserve first 256 positions for special tokens and ASCII
        self.vocab_size = max(self.vocab_size, 256 + 4)  # +4 for special tokens
    
    def encode(
        self,
        text: str,
        add_bos: Optional[bool] = None,
        add_eos: Optional[bool] = None,
        max_length: Optional[int] = None,
        padding: bool = True,
        return_attention_mask: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_bos: Override default add_bos
            add_eos: Override default add_eos
            max_length: Override default max_length
            padding: Pad to max_length
            return_attention_mask: Return attention mask
            
        Returns:
            Dict with 'input_ids' and optionally 'attention_mask'
        """
        add_bos = add_bos if add_bos is not None else self.add_bos
        add_eos = add_eos if add_eos is not None else self.add_eos
        max_length = max_length or self.max_length
        
        # Tokenize
        if self.sp is not None:
            token_ids = self.sp.EncodeAsIds(text)
        else:
            # Fallback: character-level
            token_ids = [ord(c) % (self.vocab_size - 4) + 4 for c in text]
        
        # Add special tokens
        if add_bos:
            token_ids = [self.BOS_ID] + token_ids
        if add_eos:
            token_ids = token_ids + [self.EOS_ID]
        
        # Truncate
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
        # Create attention mask before padding
        seq_len = len(token_ids)
        attention_mask = [1] * seq_len
        
        # Pad
        if padding and len(token_ids) < max_length:
            pad_len = max_length - len(token_ids)
            token_ids = token_ids + [self.PAD_ID] * pad_len
            attention_mask = attention_mask + [0] * pad_len
        
        result = {
            'input_ids': np.array(token_ids, dtype=np.int32),
            'seq_len': seq_len
        }
        
        if return_attention_mask:
            result['attention_mask'] = np.array(attention_mask, dtype=np.float32)
        
        return result
    
    def encode_batch(
        self,
        texts: List[str],
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Encode multiple texts.
        
        Returns:
            Dict with 'input_ids' (batch, seq) and 'attention_mask' (batch, seq)
        """
        results = [self.encode(text, **kwargs) for text in texts]
        
        return {
            'input_ids': np.stack([r['input_ids'] for r in results]),
            'attention_mask': np.stack([r['attention_mask'] for r in results]),
            'seq_lens': np.array([r['seq_len'] for r in results], dtype=np.int32)
        }
    
    def decode(self, token_ids: Union[List[int], np.ndarray]) -> str:
        """Decode token IDs back to text"""
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        
        # Remove special tokens
        token_ids = [t for t in token_ids if t not in (self.PAD_ID, self.BOS_ID, self.EOS_ID)]
        
        if self.sp is not None:
            return self.sp.DecodeIds(token_ids)
        else:
            # Fallback
            chars = []
            for t in token_ids:
                if t >= 4 and t < 256 + 4:
                    chars.append(chr(t - 4))
                else:
                    chars.append('?')
            return ''.join(chars)
    
    @property
    def vocab(self) -> int:
        """Return vocabulary size"""
        return self.vocab_size


def download_tokenizer(model_name: str = "bge-small", save_dir: str = "models") -> str:
    """
    Download tokenizer model from Hugging Face.
    
    Args:
        model_name: Name of the model (bge-small, minilm, etc.)
        save_dir: Directory to save the tokenizer
        
    Returns:
        Path to the downloaded tokenizer model file
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("huggingface_hub required: pip install huggingface_hub")
    
    # Model name to repo mapping
    model_repos = {
        "bge-small": "BAAI/bge-small-en-v1.5",
        "bge-base": "BAAI/bge-base-en-v1.5",
        "minilm": "sentence-transformers/all-MiniLM-L6-v2",
        "e5-small": "intfloat/e5-small-v2",
        "nomic": "nomic-ai/nomic-embed-text-v1.5",
    }
    
    repo_id = model_repos.get(model_name, model_name)
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Try to download tokenizer.model (SentencePiece)
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename="tokenizer.model",
            local_dir=str(save_path),
            local_dir_use_symlinks=False
        )
        logger.info(f"Downloaded tokenizer to {path}")
        return path
    except Exception as e:
        logger.warning(f"tokenizer.model not found: {e}")
    
    # Try sentencepiece.bpe.model
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename="sentencepiece.bpe.model",
            local_dir=str(save_path),
            local_dir_use_symlinks=False
        )
        logger.info(f"Downloaded tokenizer to {path}")
        return path
    except Exception as e:
        logger.warning(f"sentencepiece.bpe.model not found: {e}")
    
    raise FileNotFoundError(f"No SentencePiece tokenizer found for {model_name}")


class SimpleTokenizer:
    """
    Ultra-simple byte-level tokenizer for when SentencePiece isn't available.
    Uses UTF-8 bytes directly as tokens.
    """
    
    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3
    
    def __init__(self, max_length: int = 512, vocab_size: Optional[int] = None):
        self.max_length = max_length
        self.vocab_size = vocab_size or 260  # 256 bytes + 4 special tokens
    
    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: bool = True,
        return_attention_mask: bool = True
    ) -> Dict[str, np.ndarray]:
        """Encode text to byte-level tokens"""
        max_length = max_length or self.max_length
        
        # Convert to bytes, offset by 4 for special tokens
        token_ids = [b + 4 for b in text.encode('utf-8', errors='replace')]
        
        # Truncate
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
        seq_len = len(token_ids)
        attention_mask = [1.0] * seq_len
        
        # Pad
        if padding and seq_len < max_length:
            pad_len = max_length - seq_len
            token_ids = token_ids + [self.PAD_ID] * pad_len
            attention_mask = attention_mask + [0.0] * pad_len
        
        result = {
            'input_ids': np.array(token_ids, dtype=np.int32),
            'seq_len': seq_len
        }
        
        if return_attention_mask:
            result['attention_mask'] = np.array(attention_mask, dtype=np.float32)
        
        return result
    
    def encode_batch(self, texts: List[str], **kwargs) -> Dict[str, np.ndarray]:
        """Encode multiple texts"""
        results = [self.encode(text, **kwargs) for text in texts]
        return {
            'input_ids': np.stack([r['input_ids'] for r in results]),
            'attention_mask': np.stack([r['attention_mask'] for r in results]),
            'seq_lens': np.array([r['seq_len'] for r in results], dtype=np.int32)
        }
    
    def decode(self, token_ids: Union[List[int], np.ndarray]) -> str:
        """Decode tokens back to text"""
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        
        # Filter and convert back to bytes
        byte_values = []
        for t in token_ids:
            if t >= 4 and t < 260:
                byte_values.append(t - 4)
        
        return bytes(byte_values).decode('utf-8', errors='replace')


def get_tokenizer(
    model_path: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs
) -> Union[SentencePieceTokenizer, SimpleTokenizer]:
    """
    Get the best available tokenizer.
    
    Args:
        model_path: Path to tokenizer model file
        model_name: Model name to download tokenizer for
        **kwargs: Additional args for tokenizer
        
    Returns:
        Tokenizer instance
    """
    # Try loading from path
    if model_path:
        if SENTENCEPIECE_AVAILABLE:
            return SentencePieceTokenizer(model_path, **kwargs)
        else:
            logger.warning("sentencepiece not installed, using byte tokenizer")
            return SimpleTokenizer(**kwargs)
    
    # Try downloading
    if model_name and SENTENCEPIECE_AVAILABLE:
        try:
            path = download_tokenizer(model_name)
            return SentencePieceTokenizer(path, **kwargs)
        except Exception as e:
            logger.warning(f"Failed to download tokenizer: {e}")
    
    # Fallback to simple tokenizer
    if SENTENCEPIECE_AVAILABLE:
        return SentencePieceTokenizer(None, **kwargs)
    else:
        return SimpleTokenizer(**kwargs)


if __name__ == "__main__":
    # Test tokenizer
    tokenizer = get_tokenizer()
    
    text = "Hello, world! This is a test."
    encoded = tokenizer.encode(text, max_length=32)
    
    print(f"Text: {text}")
    print(f"Token IDs: {encoded['input_ids'][:encoded['seq_len']]}")
    print(f"Attention mask: {encoded['attention_mask'][:10]}...")
    
    decoded = tokenizer.decode(encoded['input_ids'])
    print(f"Decoded: {decoded}")
