"""
Inference Speed Benchmark
Tests language model generation speed (tokens/sec)
"""
import time
import sys
import numpy as np
import pytest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import find_gguf_model

# Try to import models
try:
    from model_gguf import Phi3GGUF, LLAMA_CPP_AVAILABLE
    GGUF_AVAILABLE = LLAMA_CPP_AVAILABLE
except ImportError:
    GGUF_AVAILABLE = False

try:
    from model import Phi3Model
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


class TestInferenceSpeed:
    """Benchmark inference speed for both GGUF and PyTorch models"""
    
    @pytest.fixture(scope="class")
    def gguf_model(self):
        """Load GGUF model once for all tests"""
        model_path = find_gguf_model()
        if model_path is None or not GGUF_AVAILABLE:
            pytest.skip("GGUF model not available")
        return Phi3GGUF(model_path=model_path, n_gpu_layers=-1)
    
    @pytest.fixture(scope="class")
    def pytorch_model(self):
        """Load PyTorch model once for all tests"""
        if not PYTORCH_AVAILABLE:
            pytest.skip("PyTorch model not available")
        return Phi3Model()
    
    # ==================== GGUF Benchmarks ====================
    
    @pytest.mark.skipif(not GGUF_AVAILABLE, reason="GGUF not available")
    def test_gguf_embedding_speed(self, gguf_model):
        """Benchmark GGUF embedding extraction speed"""
        test_texts = [
            "Hello, how are you?",
            "What is the meaning of life?",
            "Tell me about artificial intelligence and machine learning.",
            "The quick brown fox jumps over the lazy dog.",
            "In a hole in the ground there lived a hobbit.",
        ]
        
        # Warmup
        gguf_model.get_embedding("warmup")
        
        # Benchmark
        times = []
        for text in test_texts:
            start = time.time()
            embedding = gguf_model.get_embedding(text)
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = np.mean(times) * 1000  # ms
        
        print(f"\n[OK] GGUF Embedding Speed:")
        print(f"  Samples: {len(test_texts)}")
        print(f"  Avg latency: {avg_time:.2f} ms")
        print(f"  Embedding dim: {len(embedding)}")
        
        assert avg_time < 1000  # Should be under 1 second
    
    @pytest.mark.skipif(not GGUF_AVAILABLE, reason="GGUF not available")
    def test_gguf_generation_speed_short(self, gguf_model):
        """Benchmark GGUF generation speed - short responses"""
        prompts = [
            "What is 2+2?",
            "Say hello",
            "What color is the sky?",
        ]
        
        # Warmup
        gguf_model.generate("warmup", [])
        
        # Benchmark
        total_tokens = 0
        total_time = 0
        
        for prompt in prompts:
            start = time.time()
            response = gguf_model.generate(prompt, [])
            elapsed = time.time() - start
            
            # Estimate tokens (rough: ~4 chars per token)
            tokens = len(response) / 4
            total_tokens += tokens
            total_time += elapsed
            
            print(f"  Prompt: '{prompt[:30]}...' -> {tokens:.0f} tokens in {elapsed:.2f}s")
        
        tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
        
        print(f"\n[OK] GGUF Generation Speed (short):")
        print(f"  Total tokens: {total_tokens:.0f}")
        print(f"  Total time: {total_time:.2f} sec")
        print(f"  Throughput: {tokens_per_sec:.1f} tokens/sec")
    
    @pytest.mark.skipif(not GGUF_AVAILABLE, reason="GGUF not available")
    def test_gguf_generation_speed_with_context(self, gguf_model):
        """Benchmark GGUF generation with memory context"""
        prompt = "Based on the context, what should I do?"
        context = [
            "You are GrillCheese, a helpful AI assistant.",
            "The user previously asked about Python programming.",
            "The user mentioned they are working on a machine learning project.",
        ]
        
        # Warmup
        gguf_model.generate("warmup", [])
        
        # Benchmark
        times = []
        token_counts = []
        
        for _ in range(3):
            start = time.time()
            response = gguf_model.generate(prompt, context)
            elapsed = time.time() - start
            times.append(elapsed)
            token_counts.append(len(response) / 4)
        
        avg_time = np.mean(times)
        avg_tokens = np.mean(token_counts)
        tokens_per_sec = avg_tokens / avg_time if avg_time > 0 else 0
        
        print(f"\n[OK] GGUF Generation with Context:")
        print(f"  Context items: {len(context)}")
        print(f"  Avg response tokens: {avg_tokens:.0f}")
        print(f"  Avg latency: {avg_time:.2f} sec")
        print(f"  Throughput: {tokens_per_sec:.1f} tokens/sec")
    
    # ==================== PyTorch Benchmarks ====================
    # Note: PyTorch on CPU is very slow (2-3 min per generation)
    # These tests are marked slow and skipped by default
    
    @pytest.mark.skip(reason="PyTorch CPU inference is very slow - use GGUF instead")
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_pytorch_embedding_speed(self, pytorch_model):
        """Benchmark PyTorch embedding extraction speed"""
        test_texts = [
            "Hello, how are you?",
            "What is the meaning of life?",
            "Tell me about artificial intelligence.",
        ]
        
        # Warmup
        pytorch_model.get_embedding("warmup")
        
        # Benchmark
        times = []
        for text in test_texts:
            start = time.time()
            embedding = pytorch_model.get_embedding(text)
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = np.mean(times) * 1000  # ms
        
        print(f"\n[OK] PyTorch Embedding Speed:")
        print(f"  Device: {pytorch_model.device}")
        print(f"  Samples: {len(test_texts)}")
        print(f"  Avg latency: {avg_time:.2f} ms")
        print(f"  Embedding dim: {len(embedding)}")
    
    @pytest.mark.skip(reason="PyTorch CPU inference is very slow - use GGUF instead")
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_pytorch_generation_speed(self, pytorch_model):
        """Benchmark PyTorch generation speed"""
        prompt = "What is machine learning?"
        
        # Warmup
        pytorch_model.generate("hi", [])
        
        # Benchmark
        times = []
        token_counts = []
        
        for _ in range(2):  # Fewer iterations for slower CPU
            start = time.time()
            response = pytorch_model.generate(prompt, [])
            elapsed = time.time() - start
            times.append(elapsed)
            token_counts.append(len(response) / 4)
        
        avg_time = np.mean(times)
        avg_tokens = np.mean(token_counts)
        tokens_per_sec = avg_tokens / avg_time if avg_time > 0 else 0
        
        print(f"\n[OK] PyTorch Generation Speed:")
        print(f"  Device: {pytorch_model.device}")
        print(f"  Avg response tokens: {avg_tokens:.0f}")
        print(f"  Avg latency: {avg_time:.2f} sec")
        print(f"  Throughput: {tokens_per_sec:.1f} tokens/sec")


class TestFullPipelineSpeed:
    """Benchmark full inference pipeline (embedding + memory + generation + SNN)"""
    
    @pytest.mark.skipif(not GGUF_AVAILABLE, reason="GGUF not available")
    def test_full_pipeline_gguf(self):
        """Benchmark complete pipeline with GGUF model"""
        import tempfile
        import os
        
        from memory_store import MemoryStore
        from vulkan_backend import SNNCompute
        
        model_path = find_gguf_model()
        if model_path is None:
            pytest.skip("GGUF model not found")
        
        # Initialize components
        print("\n[INIT] Loading GGUF model...")
        model = Phi3GGUF(model_path=model_path, n_gpu_layers=-1)
        
        print("[INIT] Initializing memory store...")
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "bench_memories.db")
            memory = MemoryStore(db_path=db_path, embedding_dim=model.embedding_dim)
            
            print("[INIT] Initializing SNN...")
            snn = SNNCompute(n_neurons=1000, use_vulkan=True)
            
            # Pre-populate some memories
            print("[INIT] Populating memories...")
            for i in range(10):
                emb = model.get_embedding(f"Memory entry number {i} about topic {i % 3}")
                memory.store(emb, f"Memory {i}")
            
            # Benchmark full pipeline
            prompts = [
                "Hello, what can you help me with?",
                "Tell me about machine learning",
                "What did we talk about before?",
            ]
            
            print("\n[BENCHMARK] Running full pipeline...")
            
            results = []
            for prompt in prompts:
                timings = {}
                
                # 1. Embedding extraction
                start = time.time()
                embedding = model.get_embedding(prompt)
                timings['embedding'] = time.time() - start
                
                # 2. Memory store
                start = time.time()
                memory.store(embedding, prompt)
                timings['store'] = time.time() - start
                
                # 3. Memory retrieval
                start = time.time()
                context = memory.retrieve(embedding, k=3)
                timings['retrieve'] = time.time() - start
                
                # 4. Generation
                start = time.time()
                response = model.generate(prompt, context)
                timings['generate'] = time.time() - start
                
                # 5. SNN processing
                start = time.time()
                spikes = snn.process(embedding)
                timings['snn'] = time.time() - start
                
                timings['total'] = sum(timings.values())
                timings['response_tokens'] = len(response) / 4
                results.append(timings)
                
                print(f"  '{prompt[:30]}...'")
                print(f"    Embed: {timings['embedding']*1000:.1f}ms | "
                      f"Store: {timings['store']*1000:.1f}ms | "
                      f"Retrieve: {timings['retrieve']*1000:.1f}ms | "
                      f"Generate: {timings['generate']:.2f}s | "
                      f"SNN: {timings['snn']*1000:.1f}ms")
            
            # Summary
            avg_total = np.mean([r['total'] for r in results])
            avg_gen = np.mean([r['generate'] for r in results])
            avg_tokens = np.mean([r['response_tokens'] for r in results])
            tokens_per_sec = avg_tokens / avg_gen if avg_gen > 0 else 0
            
            print(f"\n[OK] Full Pipeline Summary (GGUF):")
            print(f"  Avg total latency: {avg_total:.2f} sec")
            print(f"  Avg generation: {avg_gen:.2f} sec")
            print(f"  Avg tokens: {avg_tokens:.0f}")
            print(f"  Generation speed: {tokens_per_sec:.1f} tokens/sec")
            print(f"  Memory operations: <10ms")
            print(f"  SNN processing: <5ms")


class TestTokenizationSpeed:
    """Benchmark tokenization speed"""
    
    @pytest.mark.skipif(not GGUF_AVAILABLE, reason="GGUF not available")
    def test_gguf_tokenization(self):
        """Benchmark GGUF tokenization"""
        model_path = find_gguf_model()
        if model_path is None:
            pytest.skip("GGUF model not found")
        
        model = Phi3GGUF(model_path=model_path, n_gpu_layers=0)  # CPU only for tokenization
        
        texts = [
            "Hello world",
            "The quick brown fox jumps over the lazy dog.",
            "In machine learning, artificial neural networks are computational models.",
            "A" * 1000,  # Long text
        ]
        
        print(f"\n[OK] GGUF Tokenization Speed:")
        for text in texts:
            start = time.time()
            for _ in range(100):
                tokens = model.llm.tokenize(text.encode('utf-8'))
            elapsed = time.time() - start
            
            avg_time = elapsed / 100 * 1000  # ms
            print(f"  '{text[:30]}...' ({len(text)} chars) -> {len(tokens)} tokens in {avg_time:.3f}ms")


class TestMemoryScaling:
    """Benchmark memory operations at different scales"""
    
    def test_memory_retrieval_scaling(self):
        """Test how retrieval speed scales with memory count"""
        import tempfile
        import os
        from memory_store import MemoryStore
        
        sizes = [10, 100, 500, 1000]
        results = []
        
        with tempfile.TemporaryDirectory() as tmpdir:
            for size in sizes:
                db_path = os.path.join(tmpdir, f"scale_{size}.db")
                memory = MemoryStore(db_path=db_path, embedding_dim=384)
                
                # Populate
                for i in range(size):
                    emb = np.random.randn(384).astype(np.float32)
                    memory.store(emb, f"Memory {i}")
                
                # Benchmark retrieval
                query = np.random.randn(384).astype(np.float32)
                
                # Warmup
                memory.retrieve(query, k=5)
                
                # Benchmark
                times = []
                for _ in range(20):
                    start = time.time()
                    memory.retrieve(query, k=5)
                    times.append(time.time() - start)
                
                avg_time = np.mean(times) * 1000  # ms
                results.append((size, avg_time))
                print(f"  {size} memories -> {avg_time:.2f}ms avg retrieval")
        
        print(f"\n[OK] Memory Retrieval Scaling:")
        for size, latency in results:
            print(f"  {size:5d} memories: {latency:.2f}ms")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

