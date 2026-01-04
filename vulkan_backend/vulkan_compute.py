"""
Main VulkanCompute class that composes all operation modules.
"""

from .vulkan_core import VulkanCore
from .vulkan_pipelines import VulkanPipelines
from .vulkan_snn import VulkanSNN
from .vulkan_faiss import VulkanFAISS
from .vulkan_fnn import VulkanFNN
from .vulkan_attention import VulkanAttention
from .vulkan_memory import VulkanMemory
from .vulkan_cells import VulkanCells


class VulkanCompute:
    """Complete Vulkan compute backend with GPU dispatch"""
    
    def __init__(self, shader_dir: str = "shaders"):
        """Initialize Vulkan compute backend"""
        # Initialize core
        self.core = VulkanCore(shader_dir)
        
        # Initialize pipelines
        self.pipelines = VulkanPipelines(self.core)
        
        # Initialize operation modules
        self.snn = VulkanSNN(self.core, self.pipelines)
        self.faiss = VulkanFAISS(self.core, self.pipelines)
        self.fnn = VulkanFNN(self.core, self.pipelines, self.core.shaders)
        self.attention = VulkanAttention(self.core, self.pipelines, self.core.shaders)
        self.memory = VulkanMemory(self.core, self.pipelines, self.core.shaders, self.fnn)
        self.cells = VulkanCells(self.core, self.pipelines, self.core.shaders)
        
        # Expose core methods for backward compatibility
        self._create_buffer = self.core._create_buffer
        self._upload_buffer = self.core._upload_buffer
        self._download_buffer = self.core._download_buffer
        self._dispatch_compute = self.core._dispatch_compute
        self.device = self.core.device
        self.descriptor_pool = self.core.descriptor_pool
        self.shaders = self.core.shaders
    
    # SNN operations - delegate to snn module
    def lif_step(self, *args, **kwargs):
        return self.snn.lif_step(*args, **kwargs)
    
    def hebbian_learning(self, *args, **kwargs):
        return self.snn.hebbian_learning(*args, **kwargs)
    
    def stdp_learning(self, *args, **kwargs):
        return self.snn.stdp_learning(*args, **kwargs)
    
    # FAISS operations - delegate to faiss module
    def faiss_compute_distances(self, *args, **kwargs):
        """Compute pairwise distances between query and database vectors"""
        return self.faiss.compute_distances(*args, **kwargs)
    
    def faiss_topk(self, *args, **kwargs):
        """Select top-k smallest distances for each query"""
        return self.faiss.topk(*args, **kwargs)
    
    # FNN operations - delegate to fnn module
    def activation_relu(self, *args, **kwargs):
        """Apply ReLU activation: max(0, x)"""
        return self.fnn.activation_relu(*args, **kwargs)
    
    def activation_gelu(self, *args, **kwargs):
        """Apply GELU activation"""
        return self.fnn.activation_gelu(*args, **kwargs)
    
    def activation_silu(self, *args, **kwargs):
        """Apply SiLU (Swish) activation: x * sigmoid(x)"""
        return self.fnn.activation_silu(*args, **kwargs)
    
    def activation_softmax(self, *args, **kwargs):
        """Apply softmax activation"""
        return self.fnn.activation_softmax(*args, **kwargs)
    
    # Memory operations - delegate to memory module
    def memory_read(self, *args, **kwargs):
        """Retrieve memories using attention mechanism"""
        return self.memory.memory_read(*args, **kwargs)
    
    def memory_write(self, *args, **kwargs):
        """Write key-value pair to memory"""
        return self.memory.memory_write(*args, **kwargs)
    
    # Attention operations - delegate to attention module
    def attention_scores(self, *args, **kwargs):
        """Compute attention scores: Q @ K^T / sqrt(head_dim)"""
        return self.attention.attention_scores(*args, **kwargs)
    
    def attention_mask(self, *args, **kwargs):
        """Apply causal mask to attention scores"""
        return self.attention.attention_mask(*args, **kwargs)
    
    def attention_output(self, *args, **kwargs):
        """Compute attention output: weights @ values"""
        return self.attention.attention_output(*args, **kwargs)
    
    def attention_concat_heads(self, *args, **kwargs):
        """Concatenate attention heads"""
        return self.attention.attention_concat_heads(*args, **kwargs)
    
    # Cell operations - delegate to cells module
    def place_cell(self, *args, **kwargs):
        """Generate place cell firing rates based on agent position"""
        return self.cells.place_cell(*args, **kwargs)
    
    def time_cell(self, *args, **kwargs):
        """Generate time cell firing rates based on elapsed time"""
        return self.cells.time_cell(*args, **kwargs)
    
    # Embedder operations - for custom SentencePiece embedder
    def position_encoding(self, embeddings, batch_size, seq_len, hidden_dim, max_position=512):
        """Add positional encoding using position-encoding shader"""
        import struct
        import numpy as np
        from .base import VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        from vulkan import vkDestroyBuffer, vkFreeMemory
        
        buf_input, mem_input = self._create_buffer(
            embeddings.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        buf_output, mem_output = self._create_buffer(
            embeddings.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        
        self._upload_buffer(buf_input, mem_input, embeddings)
        
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'position-encoding', 2, push_constant_size=16
        )
        
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'position-encoding',
            [(buf_input, embeddings.nbytes), (buf_output, embeddings.nbytes)]
        )
        
        push_constants = struct.pack('IIII', batch_size, seq_len, hidden_dim, max_position)
        
        workgroups = (batch_size * seq_len * hidden_dim + 255) // 256
        self._dispatch_compute(pipeline, pipeline_layout, descriptor_set, 
                              workgroups, push_constants)
        
        output = self._download_buffer(mem_output, embeddings.nbytes, dtype=np.float32)
        
        vkDestroyBuffer(self.device, buf_input, None)
        vkDestroyBuffer(self.device, buf_output, None)
        vkFreeMemory(self.device, mem_input, None)
        vkFreeMemory(self.device, mem_output, None)
        
        return output
    
    def mean_pooling(self, embeddings, attention_mask, batch_size, seq_len, hidden_dim):
        """Mean pooling over sequence using mean-pooling shader"""
        import struct
        import numpy as np
        from .base import VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        from vulkan import vkDestroyBuffer, vkFreeMemory
        
        output_size = batch_size * hidden_dim
        
        buf_emb, mem_emb = self._create_buffer(
            embeddings.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        buf_mask, mem_mask = self._create_buffer(
            attention_mask.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        buf_output, mem_output = self._create_buffer(
            output_size * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        
        self._upload_buffer(buf_emb, mem_emb, embeddings)
        self._upload_buffer(buf_mask, mem_mask, attention_mask)
        
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'mean-pooling', 3, push_constant_size=12
        )
        
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'mean-pooling',
            [
                (buf_emb, embeddings.nbytes),
                (buf_mask, attention_mask.nbytes),
                (buf_output, output_size * 4)
            ]
        )
        
        push_constants = struct.pack('III', batch_size, seq_len, hidden_dim)
        
        workgroups = (batch_size * hidden_dim + 255) // 256
        self._dispatch_compute(pipeline, pipeline_layout, descriptor_set,
                              workgroups, push_constants)
        
        output = self._download_buffer(mem_output, output_size * 4, dtype=np.float32)
        
        vkDestroyBuffer(self.device, buf_emb, None)
        vkDestroyBuffer(self.device, buf_mask, None)
        vkDestroyBuffer(self.device, buf_output, None)
        vkFreeMemory(self.device, mem_emb, None)
        vkFreeMemory(self.device, mem_mask, None)
        vkFreeMemory(self.device, mem_output, None)
        
        return output
    
    def l2_normalize(self, embeddings, batch_size, hidden_dim):
        """L2 normalize embeddings using l2-normalize shader"""
        import struct
        import numpy as np
        from .base import VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        from vulkan import vkDestroyBuffer, vkFreeMemory
        
        buf_input, mem_input = self._create_buffer(
            embeddings.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        buf_output, mem_output = self._create_buffer(
            embeddings.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        
        self._upload_buffer(buf_input, mem_input, embeddings)
        
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'l2-normalize', 2, push_constant_size=8
        )
        
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'l2-normalize',
            [(buf_input, embeddings.nbytes), (buf_output, embeddings.nbytes)]
        )
        
        push_constants = struct.pack('II', batch_size, hidden_dim)
        
        workgroups = batch_size
        self._dispatch_compute(pipeline, pipeline_layout, descriptor_set,
                              workgroups, push_constants)
        
        output = self._download_buffer(mem_output, embeddings.nbytes, dtype=np.float32)
        
        vkDestroyBuffer(self.device, buf_input, None)
        vkDestroyBuffer(self.device, buf_output, None)
        vkFreeMemory(self.device, mem_input, None)
        vkFreeMemory(self.device, mem_output, None)
        
        return output
    
    def embedding_lookup(self, token_ids, embedding_table, vocab_size, hidden_dim):
        """Embedding lookup using existing embedding-lookup shader"""
        import struct
        import numpy as np
        from .base import VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        from vulkan import vkDestroyBuffer, vkFreeMemory
        
        output_size = len(token_ids) * hidden_dim
        
        buf_ids, mem_ids = self._create_buffer(
            token_ids.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        buf_table, mem_table = self._create_buffer(
            embedding_table.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        buf_output, mem_output = self._create_buffer(
            output_size * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        
        self._upload_buffer(buf_ids, mem_ids, token_ids)
        self._upload_buffer(buf_table, mem_table, embedding_table.flatten())
        
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'embedding-lookup', 3, push_constant_size=12
        )
        
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'embedding-lookup',
            [
                (buf_ids, token_ids.nbytes),
                (buf_table, embedding_table.nbytes),
                (buf_output, output_size * 4)
            ]
        )
        
        push_constants = struct.pack('III', len(token_ids), vocab_size, hidden_dim)
        
        workgroups = (output_size + 255) // 256
        self._dispatch_compute(pipeline, pipeline_layout, descriptor_set,
                              workgroups, push_constants)
        
        output = self._download_buffer(mem_output, output_size * 4, dtype=np.float32)
        
        vkDestroyBuffer(self.device, buf_ids, None)
        vkDestroyBuffer(self.device, buf_table, None)
        vkDestroyBuffer(self.device, buf_output, None)
        vkFreeMemory(self.device, mem_ids, None)
        vkFreeMemory(self.device, mem_table, None)
        vkFreeMemory(self.device, mem_output, None)
        
        return output
    
    def __del__(self):
        """Cleanup Vulkan resources"""
        if hasattr(self, 'pipelines'):
            self.pipelines.cleanup()
        if hasattr(self, 'core'):
            self.core.cleanup()

