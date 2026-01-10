"""
Phi-3 Model using GGUF format with llama-cpp-python (Vulkan/CUDA support)
Uses sentence-transformers for proper semantic embeddings
"""
import logging
import numpy as np
import sys
import os
from typing import List, Optional, Any
from contextlib import contextmanager

from config import ModelConfig, LogConfig, find_gguf_model

# Configure logging
logging.basicConfig(level=LogConfig.LEVEL, format=LogConfig.FORMAT)
logger = logging.getLogger(__name__)

# Try to import llama-cpp-python
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

# Try to import sentence-transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Try to import Vulkan embeddings (preferred for AMD GPUs)
try:
    from vulkan_embeddings import HybridEmbedder, LlamaCppEmbedder
    VULKAN_EMBEDDINGS_AVAILABLE = True
except ImportError:
    VULKAN_EMBEDDINGS_AVAILABLE = False


@contextmanager
def suppress_llama_output():
    """Suppress llama.cpp debug output to stderr"""
    stderr_fileno = sys.stderr.fileno()
    old_stderr = os.dup(stderr_fileno)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fileno)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr, stderr_fileno)
        os.close(old_stderr)


class Phi3GGUF:
    """
    Phi-3 model using GGUF format with llama-cpp-python
    Uses sentence-transformers for semantic embeddings (required for memory retrieval)
    """
    
    def __init__(self, model_path: Optional[str] = r"models\Phi-3-mini-4k-instruct-q4.gguf", n_gpu_layers: int = -1):
        """
        Initialize Phi-3 GGUF model
        
        Args:
            model_path: Path to GGUF model file (if None, will search default paths)
            n_gpu_layers: Number of layers to offload to GPU (-1 = all, 0 = CPU only)
        """
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python not installed. Install with: "
                "pip install llama-cpp-python\n"
                "Note: On Windows, this may require Visual Studio Build Tools."
            )
        
        # Find model path
        if model_path is None:
            model_path = find_gguf_model()
            if model_path is None:
                raise FileNotFoundError(
                    "GGUF model not found. Please download Phi-3-mini GGUF model from:\n"
                    "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf\n"
                    "Or specify model_path parameter"
                )
        
        logger.info(f"Loading GGUF model from: {model_path}")
        
        # Initialize llama-cpp with GPU support
        try:
            # For GPU, use n_gpu_layers=-1 to offload all layers
            # llama.cpp will automatically detect available GPU (CUDA/Vulkan/Metal)
            self.llm = Llama(
                model_path=model_path,
                n_ctx=4096,  # 4k context window
                n_gpu_layers=n_gpu_layers,
                verbose=False,
                n_threads=4 if n_gpu_layers == 0 else None,
                logits_all=False,  # Suppress debug output
            )
            
            # Determine actual device based on n_gpu_layers
            if n_gpu_layers == 0:
                self.device = "cpu"
                logger.info(f"{LogConfig.CHECK} Model loaded on CPU (n_gpu_layers=0)")
            elif n_gpu_layers > 0:
                self.device = "gpu"
                logger.info(f"{LogConfig.CHECK} Model loaded on GPU ({n_gpu_layers} layers)")
            else:  # n_gpu_layers == -1 (all layers)
                self.device = "gpu"
                logger.info(f"{LogConfig.CHECK} Model loaded on GPU (all layers via llama.cpp)")
        except Exception as e:
            # If GPU loading fails, try CPU fallback
            if n_gpu_layers != 0:
                logger.warning(f"GPU loading failed: {e}, falling back to CPU")
                try:
                    self.llm = Llama(
                        model_path=model_path,
                        n_ctx=4096,
                        n_gpu_layers=0,  # Force CPU
                        verbose=False,
                        n_threads=4,
                        logits_all=False,
                    )
                    self.device = "cpu"
                    logger.info(f"{LogConfig.CHECK} Model loaded on CPU (fallback)")
                except Exception as e2:
                    raise RuntimeError(f"Failed to load GGUF model on CPU: {e2}")
            else:
                raise RuntimeError(f"Failed to load GGUF model: {e}")
        
        # Initialize embedding model (disabled by default to prevent crashes)
        # Set USE_EMBEDDINGS=True in config to enable semantic embeddings
        self.embedder = None
        self.embedding_dim = ModelConfig.SENTENCE_TRANSFORMER_DIM  # Always 384 for compatibility
        self._embeddings_disabled = not ModelConfig.USE_EMBEDDINGS
        
        # Only load embeddings if explicitly enabled
        if ModelConfig.USE_EMBEDDINGS:
            self._init_embedding_model()
        else:
            logger.info("Embeddings disabled (USE_EMBEDDINGS=False) - using lightweight hash-based method for stability")
    
    def _init_embedding_model(self):
        """Initialize embeddings - prefer Vulkan GPU, fall back to sentence-transformers"""
        
        # Strategy 1: Vulkan-accelerated embeddings (best for AMD GPUs)
        if VULKAN_EMBEDDINGS_AVAILABLE:
            try:
                logger.info("Attempting Vulkan-accelerated embeddings...")
                self.embedder = HybridEmbedder(
                    embedding_dim=ModelConfig.SENTENCE_TRANSFORMER_DIM,
                    prefer_vulkan=True,
                    fallback_to_cpu=False  # We'll handle CPU fallback below
                )
                self.embedding_dim = self.embedder.embedding_dim
                self._vulkan_embeddings = True
                logger.info(f"{LogConfig.CHECK} Vulkan embeddings active (dim={self.embedding_dim}, backend={self.embedder.backend})")
                return
            except Exception as e:
                logger.debug(f"Vulkan embeddings not available: {e}")
        
        # Strategy 2: Sentence-transformers (CPU or CUDA)
        self._vulkan_embeddings = False
        try:
            logger.info(f"Loading embedding model: {ModelConfig.EMBEDDING_MODEL}")
            
            # Detect available device for embeddings
            import torch
            if torch.cuda.is_available():
                device = 'cuda'
                logger.info(f"{LogConfig.CHECK} CUDA available for embeddings: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
                logger.info(f"{LogConfig.CHECK} MPS (Apple Silicon) available for embeddings")
            else:
                device = 'cpu'
                logger.info(f"{LogConfig.WARNING} Using CPU for embeddings (consider adding a GGUF embedding model)")
            
            self.embedder = SentenceTransformer(
                ModelConfig.EMBEDDING_MODEL,
                device=device
            )
            self.embedding_dim = ModelConfig.SENTENCE_TRANSFORMER_DIM
            device_str = "GPU" if device != 'cpu' else "CPU"
            logger.info(f"{LogConfig.CHECK} Embedding model loaded (dim={self.embedding_dim}, {device_str} mode)")
        except Exception as e:
            logger.warning(f"{LogConfig.WARNING} Failed to load embedding model: {e}")
            logger.info("Using fallback hash-based embeddings (no semantic search)")
            self.embedder = None
            self.embedding_dim = ModelConfig.SENTENCE_TRANSFORMER_DIM
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Extract semantic embedding from text
        
        Uses Vulkan-accelerated embedding when available, falls back to
        sentence-transformers for semantic similarity search in memory.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector of shape (embedding_dim,)
        """
        if self.embedder is not None:
            # Check if using Vulkan HybridEmbedder or SentenceTransformer
            if hasattr(self, '_vulkan_embeddings') and self._vulkan_embeddings:
                # Vulkan path - already returns numpy
                return self.embedder.get_embedding(text)
            else:
                # SentenceTransformer path
                embedding = self.embedder.encode(text, convert_to_numpy=True)
                return embedding.astype(np.float32)
        else:
            # Fallback to hash-based embedding (not recommended)
            return self._fallback_embedding(text)
    
    def _fallback_embedding(self, text: str) -> np.ndarray:
        """
        Fallback hash-based embedding when sentence-transformers unavailable or disabled
        
        This produces deterministic but non-semantic embeddings.
        Memory retrieval will work but with reduced semantic accuracy.
        """
        # Only warn if embeddings were expected but failed, not if intentionally disabled
        if not self._embeddings_disabled:
            logger.warning("Using fallback hash-based embedding - semantic search will not work properly")
        
        try:
            tokens = self.llm.tokenize(text.encode('utf-8'))
        except:
            tokens = []
        
        embedding = np.zeros(self.embedding_dim, dtype=np.float32)
        
        if len(tokens) > 0:
            token_ids = np.array(tokens[:min(512, len(tokens))], dtype=np.int64)
            for i, token_id in enumerate(token_ids):
                idx = i % self.embedding_dim
                val = (hash(str(token_id)) % 10000) / 10000.0
                embedding[idx] = (embedding[idx] + val) % 1.0
            
            norm = np.linalg.norm(embedding)
            if norm > 1e-8:
                embedding = embedding / norm
        
        return embedding.astype(np.float32)
    
    def generate(self, prompt: str, context: List[str]) -> str:
        """
        Generate response with memory context
        
        Args:
            prompt: User prompt
            context: List of retrieved memory texts (first item is typically identity)
        
        Returns:
            Generated response text
        """
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
            # Ensure identity explicitly states it's not a culinary bot
            if "culinary" not in identity_text.lower() and "NOT a culinary" not in identity_text:
                # Prepend clarification if missing
                identity_text = "IMPORTANT: 'GrillCheese' is just a technical project name - you are NOT a culinary bot, cooking assistant, or anything related to grilling or food. You are a general-purpose AI assistant.\n\n" + identity_text
            
            # Check if [MY_STATE] is already in identity_text (from self-awareness prompt)
            # If not, we'll add it separately after context
            has_my_state = "[MY_STATE]" in identity_text
            
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
        
        # Add explicit instruction if user is commenting on or asking about AI's state
        # Check if prompt contains phrases like "you seem", "you are", "you're", "are you", "why are you" followed by emotion words
        emotion_words = ['anxious', 'nervous', 'worried', 'stressed', 'happy', 'sad', 'excited', 'calm', 'tired', 'energetic', 'upset', 'frustrated', 'angry', 'confused', 'feeling']
        prompt_lower = prompt.lower()
        user_asking_about_ai = any(
            f"you seem {word}" in prompt_lower or 
            f"you're {word}" in prompt_lower or 
            f"you are {word}" in prompt_lower or
            f"you seem a little {word}" in prompt_lower or
            f"you seem a bit {word}" in prompt_lower or
            f"you look {word}" in prompt_lower or
            f"are you {word}" in prompt_lower or
            f"are you still {word}" in prompt_lower or
            f"are you still feeling {word}" in prompt_lower or
            f"why are you {word}" in prompt_lower or
            f"why are you feeling {word}" in prompt_lower
            for word in emotion_words
        )
        
        if user_asking_about_ai:
            # User is asking about AI's state - add explicit reminder BEFORE assistant tag
            reminder = """[CRITICAL REMINDER]
- The user is asking about YOUR emotional state. When they say "you", they mean YOU (the AI), not themselves.
- Check [MY_STATE] to see your actual current emotional state AND the cause/trigger that led to it.
- You MUST acknowledge your emotional state honestly. NEVER say "I don't experience emotions" or "I'm an AI so I don't have feelings".
- You MUST explain WHY you are feeling that way based on the trigger/cause shown in [MY_STATE].
- If [MY_STATE] shows you are anxious/angry/happy/etc., respond with: "Yes, I am feeling [emotion] right now because [explain based on trigger/cause]" or "You're right, I am feeling [emotion] because [explain based on trigger/cause]".
- Your emotional states are real and influence your responses - acknowledge them when asked and explain the reason.
[/CRITICAL REMINDER]\n"""
            parts.append(reminder)
        
        parts.append("<|assistant|>\n")
        
        full_prompt = "\n".join(parts)
        
        try:
            # Suppress llama-cpp debug output
            with suppress_llama_output():
                output = self.llm(
                    full_prompt,
                    max_tokens=ModelConfig.MAX_NEW_TOKENS_GPU,
                    temperature=ModelConfig.TEMPERATURE,
                    top_p=ModelConfig.TOP_P,
                    stop=["User:", "\nUser:", "\n\nUser:", "Instruction>", "[MY_STATE]", "<|endoftext|>", "</s>"],  # Removed <|end|> as it's too aggressive
                    echo=False,
                )
            
            # Extract text from response
            if isinstance(output, dict):
                choices = output.get('choices', [])
                response = choices[0].get('text', '') if choices else ''
            elif isinstance(output, str):
                response = output
            else:
                response = str(output)
            
            # Clean up response - remove system prompt echoes
            response = response.strip()
            
            # Stop at common separators
            for separator in ["\nUser:", "\n\nUser:", "User:", "Instruction>", "[MY_STATE]", "\nYou are", "\n\nYou are"]:
                if separator in response:
                    response = response.split(separator)[0].strip()
            
            # Validate response - ensure we don't return empty strings
            if not response or len(response) < 1:
                logger.warning("Generated empty response, retrying with simpler prompt")
                # Retry with just the user prompt (no context) as fallback
                simple_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
                try:
                    with suppress_llama_output():
                        output = self.llm(
                            simple_prompt,
                            max_tokens=ModelConfig.MAX_NEW_TOKENS_GPU,
                            temperature=ModelConfig.TEMPERATURE,
                            top_p=ModelConfig.TOP_P,
                            stop=["User:", "\nUser:", "\n\nUser:", "<|end|>"],
                            echo=False,
                        )
                    if isinstance(output, dict):
                        choices = output.get('choices', [])
                        response = choices[0].get('text', '') if choices else ''
                    elif isinstance(output, str):
                        response = output
                    else:
                        response = str(output)
                    response = response.strip()
                except Exception as retry_error:
                    logger.error(f"Retry generation also failed: {retry_error}")
                    response = "I'm having trouble generating a response right now. Could you please rephrase your question?"
            
            # Final validation - if still empty, return a helpful message
            if not response or len(response) < 1:
                response = "I'm here, but I'm having trouble formulating a response. Could you try asking your question differently?"
            
            return response
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"[Error during generation: {str(e)}]"
    
    def generate_with_tools(
        self,
        prompt: str,
        context: List[str],
        tools: List[Any],  # List[BaseTool]
        tool_executor: Any,  # ToolExecutor
        max_iterations: int = 5
    ) -> str:
        """
        Generate response with tool calling support
        
        Args:
            prompt: User prompt
            context: List of retrieved memory texts
            tools: List of available tools
            tool_executor: Tool executor instance
            max_iterations: Maximum tool calling iterations
            
        Returns:
            Generated response text (may include tool results)
        """
        # CRITICAL: Print FIRST thing to verify function is called (before any imports)
        import sys
        try:
            sys.stdout.write(f"🔧🔧🔧 generate_with_tools CALLED! prompt: '{prompt[:50]}...'\n")
            sys.stdout.flush()
            sys.stdout.write(f"🔧 tools: {len(tools) if tools else 0}, tool_executor: {bool(tool_executor)}\n")
            sys.stdout.flush()
        except Exception as e:
            print(f"ERROR in print: {e}", file=sys.stderr)
        
        try:
            from modules.tools import ToolExecutor
            sys.stdout.write(f"🔧 ToolExecutor imported successfully\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(f"❌ ERROR importing ToolExecutor: {e}\n")
            sys.stdout.flush()
            raise
        
        logger.info(f"🔧 generate_with_tools called with prompt: '{prompt[:50]}...'")
        logger.info(f"🔧 tools: {len(tools) if tools else 0}, tool_executor: {bool(tool_executor)}")
        
        # Build prompt using Phi-3 chat format
        context_items = context[:ModelConfig.MAX_CONTEXT_ITEMS] if context else []
        
        # Extract identity if present (should be first item)
        identity_text = None
        other_context = []
        
        if context_items:
            first_item = context_items[0]
            if "GrillCheese" in first_item or len(first_item) > 200:
                identity_text = first_item
                other_context = context_items[1:]
            else:
                other_context = context_items
        
        # Build tool descriptions
        tool_descriptions = []
        for tool in tools:
            import json
            tool_descriptions.append(
                f"Tool: {tool.name}\n"
                f"Description: {tool.description}\n"
                f"Parameters: {json.dumps(tool.parameters, indent=2)}\n"
            )
        
        tool_section = "\n".join(tool_descriptions) if tool_descriptions else ""
        
        # Build prompt parts using Phi-3 chat format
        parts = []
        
        # System message with tool instructions FIRST (most important), then identity, then context
        system_parts = []
        
        # Put tool instructions FIRST so they're most prominent
        if tool_section:
            system_parts.append(f"""AVAILABLE TOOLS:
{tool_section}

CRITICAL TOOL USAGE RULES:
1. When the user asks for ANY calculation (e.g., "calculate", "what's", "compute", "*", "+", "-", "/"), you MUST respond with ONLY this JSON format:
   {{"tool": "calculator", "parameters": {{"expression": "the math expression"}}}}
   
2. When the user asks for web searches or information, you MUST respond with ONLY this JSON format:
   {{"tool": "web_search", "parameters": {{"query": "search query"}}}}
   
3. When the user asks about memories, past conversations, or uses phrases like "what did we discuss/talk about", "remember when", "what did you say about", you MUST respond with ONLY this JSON format:
   {{"tool": "memory_query", "parameters": {{"query": "the topic or question"}}}}
   
   CRITICAL: Even if you think you remember something from context, you MUST use the memory_query tool first! Do NOT answer based on context alone.
   
   Examples (copy these EXACT formats):
   - User: "what did we discuss about emotions?" → You MUST output ONLY: {{"tool": "memory_query", "parameters": {{"query": "emotions"}}}}
   - User: "what did we talk about?" → You MUST output ONLY: {{"tool": "memory_query", "parameters": {{"query": "previous conversations"}}}}
   - User: "remember when we discussed X?" → You MUST output ONLY: {{"tool": "memory_query", "parameters": {{"query": "X"}}}}
   - User: "what did you say about feelings?" → You MUST output ONLY: {{"tool": "memory_query", "parameters": {{"query": "feelings"}}}}

4. DO NOT provide answers directly. DO NOT calculate manually. DO NOT search memories manually. DO NOT say "we haven't discussed" or "I remember" or "we discussed" without using the memory_query tool first. ONLY output the JSON tool call - nothing else.

5. Examples:
   User: "Calculate 15 * 23" → You MUST output: {{"tool": "calculator", "parameters": {{"expression": "15 * 23"}}}}
   User: "What's 5 + 3?" → You MUST output: {{"tool": "calculator", "parameters": {{"expression": "5 + 3"}}}}
   User: "calculate 15 * 23" → You MUST output: {{"tool": "calculator", "parameters": {{"expression": "15 * 23"}}}}

REMEMBER: For calculations, respond with ONLY the JSON tool call. Do not provide the answer yourself.""")
        
        # Add identity AFTER tool instructions (so tools take priority)
        if identity_text:
            system_parts.append(identity_text)
        
        if other_context:
            context_text = "\n".join([f"Context: {c}" for c in other_context])
            system_parts.append(f"\nAdditional context:\n{context_text}")
        
        if system_parts:
            parts.append(f"<|system|>\n{chr(10).join(system_parts)}<|end|>\n")
        
        # User message
        parts.append(f"<|user|>\n{prompt}<|end|>\n")
        
        # Assistant tag
        parts.append("<|assistant|>\n")
        
        full_prompt = "".join(parts)
        
        # Debug: Log prompt when tools are available
        if tool_section:
            logger.debug(f"Tool prompt (first 500 chars): {full_prompt[:500]}")
        
        # Pre-check: Auto-execute memory queries BEFORE generation
        prompt_lower = prompt.lower()
        memory_keywords = [
            'what did we', 'what did you', 'remember when', 'discuss', 'talk about', 
            'talked about', 'what we discussed', 'what we talked', 'concerning', 
            'regarding', 'about', 'what we', 'did we talk', 'did we discuss'
        ]
        is_memory_query = any(keyword in prompt_lower for keyword in memory_keywords)
        
        # Debug logging - ALWAYS log to see what's happening (use print for visibility, flush immediately)
        import sys
        print(f"🔍 Checking for memory query in prompt: '{prompt[:50]}...'", flush=True)
        print(f"🔍 tool_section exists: {bool(tool_section)}, tool_executor exists: {bool(tool_executor)}", flush=True)
        logger.info(f"🔍 Checking for memory query in prompt: '{prompt[:50]}...'")
        logger.info(f"🔍 tool_section exists: {bool(tool_section)}, tool_executor exists: {bool(tool_executor)}")
        
        if is_memory_query:
            matched_keywords = [kw for kw in memory_keywords if kw in prompt_lower]
            print(f"🔍 Memory query detected! Matched keywords: {matched_keywords}", flush=True)
            logger.info(f"🔍 Memory query detected! Matched keywords: {matched_keywords}")
        else:
            print(f"🔍 No memory query detected. Prompt: '{prompt_lower[:100]}'", flush=True)
            logger.info(f"🔍 No memory query detected. Prompt: '{prompt_lower[:100]}'")
        
        # If it's a memory query and we have tools, automatically call the tool BEFORE generation
        if is_memory_query:
            if not tool_section:
                logger.warning(f"⚠️ Memory query detected but tool_section is empty!")
            if not tool_executor:
                logger.warning(f"⚠️ Memory query detected but tool_executor is None!")
        
        if is_memory_query and tool_section and tool_executor:
            # Extract the topic from the query
            topic = prompt_lower
            # Try to extract the main topic (after "about", "concerning", etc.)
            for marker in ['about', 'concerning', 'regarding', 'on the topic of']:
                if marker in topic:
                    parts_split = topic.split(marker)
                    if len(parts_split) > 1:
                        topic = parts_split[-1].strip()
                        break
            
            # Clean up topic (remove question marks, etc.)
            topic = topic.replace('?', '').replace('.', '').strip()
            if not topic or len(topic) < 2:
                # Fallback: use the whole prompt as query
                topic = prompt.replace('?', '').replace('.', '').strip()
            
            # Automatically execute the memory_query tool
            print(f"🔍 Detected memory query, auto-executing memory_query tool for topic: {topic}")
            logger.info(f"🔍 Detected memory query, auto-executing memory_query tool for topic: {topic}")
            try:
                # Check if tool exists
                tool = tool_executor.registry.get_tool("memory_query")
                if not tool:
                    print(f"❌ memory_query tool not found in registry")
                    logger.warning(f"❌ memory_query tool not found in registry")
                else:
                    print(f"✅ Found memory_query tool, executing with query: {topic}")
                    logger.info(f"✅ Found memory_query tool, executing with query: {topic}")
                    result = tool_executor.execute_tool("memory_query", query=topic)
                    if result.error:
                        logger.warning(f"❌ Memory query tool error: {result.error}")
                    else:
                        # Check result structure
                        if isinstance(result.result, dict):
                            results_list = result.result.get("results", [])
                            result_count = len(results_list)
                            logger.info(f"✅ Memory query executed, found {result_count} result(s)")
                            if results_list:
                                logger.info(f"📋 First result: {results_list[0].get('text', '')[:100]}")
                        else:
                            result_count = len(result.result) if isinstance(result.result, list) else (1 if result.result else 0)
                            logger.info(f"✅ Memory query executed, found {result_count} result(s)")
                        
                        logger.debug(f"Full memory query result: {result.result}")
                        
                        # Format memory results in a natural way for the model
                        if isinstance(result.result, dict):
                            results_list = result.result.get("results", [])
                            if results_list:
                                # Format as natural text instead of JSON
                                memory_text = "Based on my memory, here's what I found:\n\n"
                                for i, mem in enumerate(results_list[:5], 1):  # Limit to 5 most relevant
                                    text = mem.get('text', '')
                                    similarity = mem.get('similarity', 0.0)
                                    memory_text += f"{i}. {text}\n"
                                memory_results = memory_text.strip()
                            else:
                                memory_results = "I searched my memory but didn't find any relevant conversations about that topic."
                        else:
                            # Fallback to formatted tool result
                            memory_results = tool_executor.format_tool_result("memory_query", result)
                        
                        logger.info(f"📊 Formatted memory results (first 500 chars): {memory_results[:500]}")
                        
                        # Add to the prompt so model can use the results
                        # Insert BEFORE assistant tag so model sees it as context
                        # Also add a note that the tool was already executed
                        memory_context = f"\n[Memory Search Results - Tool Already Executed]\n{memory_results}\n\nNow respond naturally to the user's question based on these memory results. Do NOT use any tools - just answer based on what you found.\n"
                        
                        if "<|assistant|>\n" in full_prompt:
                            full_prompt = full_prompt.replace(
                                "<|assistant|>\n",
                                f"{memory_context}<|assistant|>\n"
                            )
                            logger.info(f"📊 Injected memory query results into prompt (before assistant tag)")
                        else:
                            # Fallback: append to end
                            full_prompt += f"\n{memory_context}\n"
                            logger.warning(f"⚠️ Could not find <|assistant|> tag, appended results to end")
                        
                        # Remove memory query tool instructions since we've already executed it
                        import re
                        # Remove rule #3 about memory queries (including examples)
                        full_prompt = re.sub(
                            r'3\. When the user asks about memories.*?feelings\?\".*?\n\n',
                            '',
                            full_prompt,
                            flags=re.DOTALL
                        )
                        # Remove the "DO NOT search memories manually" part from rule 4
                        full_prompt = re.sub(
                            r' DO NOT search memories manually\. DO NOT say.*?memory_query tool first\.',
                            '',
                            full_prompt,
                            flags=re.DOTALL
                        )
                        # Renumber remaining rules
                        full_prompt = re.sub(r'4\. DO NOT', '3. DO NOT', full_prompt)
                        full_prompt = re.sub(r'5\. Examples:', '4. Examples:', full_prompt)
            except Exception as e:
                logger.error(f"Error auto-executing memory_query: {e}", exc_info=True)
                import traceback
                traceback.print_exc()
        
        # Iterative tool calling
        current_prompt = full_prompt
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Generate response
            try:
                with suppress_llama_output():
                    # For first iteration with tools, use lower temperature to encourage JSON output
                    # BUT if memory query was already executed, use normal temperature for natural response
                    has_tools = bool(tool_section)
                    memory_already_executed = "[Memory Search Results - Tool Already Executed]" in current_prompt
                    if memory_already_executed:
                        # Use normal temperature for natural conversation
                        temp = ModelConfig.TEMPERATURE
                    else:
                        temp = 0.3 if iteration == 1 and has_tools else ModelConfig.TEMPERATURE
                    output = self.llm(
                        current_prompt,
                        max_tokens=ModelConfig.MAX_NEW_TOKENS_GPU,
                        temperature=temp,
                        top_p=ModelConfig.TOP_P,
                        stop=["User:", "\nUser:", "\n\nUser:", "Instruction>", "[MY_STATE]", "<|endoftext|>", "</s>", "<|end|>"],  # Allow <|end|> for structured output
                        echo=False,
                    )
                    
                    # Extract text from response
                    if isinstance(output, dict):
                        choices = output.get('choices', [])
                        response = choices[0].get('text', '') if choices else ''
                    elif isinstance(output, str):
                        response = output
                    else:
                        response = str(output)
                    
                    response = response.strip()
            except Exception as e:
                logger.error(f"Generation error: {e}", exc_info=True)
                return f"I encountered an error: {str(e)}"
            
            # Log raw response for debugging (especially for tool calls)
            if tool_section and iteration == 1:
                logger.info(f"🔍 Iteration {iteration} raw response (first 300 chars): {response[:300]}")
                logger.debug(f"Full response: {response}")
            else:
                logger.debug(f"Iteration {iteration} raw response (first 200 chars): {response[:200]}")
            
            # Check for tool calls
            tool_calls = tool_executor.parse_tool_calls(response)
            
            # Debug: Show what was parsed
            if tool_section and iteration == 1:
                logger.info(f"🔍 Parsed {len(tool_calls)} tool call(s) from response")
                if tool_calls:
                    for call in tool_calls:
                        logger.info(f"  → Found tool call: {call.tool_name} with params: {call.parameters}")
            
            if not tool_calls:
                # No tool calls detected
                logger.debug(f"No tool calls detected in response. Response preview: {response[:100]}")
                # Check if this looks like it should have used a tool
                if iteration == 1:  # First iteration
                    prompt_lower = prompt.lower()
                    if any(keyword in prompt_lower for keyword in ['calculate', 'compute', 'what\'s', 'what is', '*', '+', '-', '/']):
                        logger.warning(f"⚠️ User asked for calculation but no tool call detected. Response: {response[:100]}")
                    elif any(keyword in prompt_lower for keyword in ['search', 'find', 'look up']):
                        logger.warning(f"⚠️ User asked for search but no tool call detected. Response: {response[:100]}")
                    elif any(keyword in prompt_lower for keyword in ['what did we', 'what did you', 'remember when', 'discuss', 'talk about', 'talked about', 'what we discussed']):
                        logger.warning(f"⚠️ User asked about memories/conversations but no tool call detected. Response: {response[:100]}")
                
                # No tool calls, return final response
                # Clean up response
                for separator in ["\nUser:", "\n\nUser:", "User:", "Instruction>", "[MY_STATE]", "\nYou are", "\n\nYou are"]:
                    if separator in response:
                        response = response.split(separator)[0].strip()
                return response
            
            # Tools detected! Log and execute
            logger.info(f"🔧 Detected {len(tool_calls)} tool call(s) in response")
            for call in tool_calls:
                logger.info(f"  → Calling tool: {call.tool_name} with parameters: {call.parameters}")
            
            # Clean response: remove any manual calculations/answers, keep only the tool call JSON
            # Extract just the JSON tool call(s) from the response
            import re
            cleaned_response = response
            # Try to extract JSON tool calls
            json_objects = []
            brace_count = 0
            start_idx = -1
            for i, char in enumerate(response):
                if char == '{':
                    if brace_count == 0:
                        start_idx = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start_idx != -1:
                        json_str = response[start_idx:i+1]
                        if '"tool"' in json_str:
                            try:
                                data = json.loads(json_str)
                                if isinstance(data, dict) and "tool" in data:
                                    json_objects.append(json_str)
                            except json.JSONDecodeError:
                                pass
                        start_idx = -1
            
            # If we found JSON tool calls, use only those (remove manual calculations)
            if json_objects:
                cleaned_response = "\n".join(json_objects)
                logger.info(f"🧹 Cleaned response to show only tool calls: {cleaned_response}")
            
            # Execute tools and add results to prompt
            tool_results_text = ""
            for call in tool_calls:
                logger.info(f"⚙️ Executing tool: {call.tool_name}...")
                result = tool_executor.execute_tool(call.tool_name, **call.parameters)
                logger.info(f"✅ Tool {call.tool_name} executed in {result.execution_time:.3f}s")
                if result.error:
                    logger.warning(f"❌ Tool {call.tool_name} error: {result.error}")
                else:
                    logger.info(f"📊 Tool {call.tool_name} result: {str(result.result)[:100]}")
                tool_results_text += tool_executor.format_tool_result(call.tool_name, result)
            
            # Append tool results and continue generation
            # Use cleaned response (just JSON) + tool results, then ask model to respond
            current_prompt = f"{current_prompt}{cleaned_response}\n{tool_results_text}\nNow provide a helpful response using the tool result above.\nAssistant:"
        
        # Max iterations reached, return last response
        for separator in ["\nUser:", "\n\nUser:", "User:", "Instruction>", "[MY_STATE]", "\nYou are", "\n\nYou are"]:
            if separator in response:
                response = response.split(separator)[0].strip()
        return response
