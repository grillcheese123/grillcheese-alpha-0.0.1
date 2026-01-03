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
    
    def __del__(self):
        """Cleanup Vulkan resources"""
        if hasattr(self, 'pipelines'):
            self.pipelines.cleanup()
        if hasattr(self, 'core'):
            self.core.cleanup()

