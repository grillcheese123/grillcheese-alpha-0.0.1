"""
Main VulkanCompute class that composes all operation modules.
"""

from .vulkan_core import VulkanCore
from .vulkan_pipelines import VulkanPipelines
from .vulkan_snn import VulkanSNN
from .vulkan_faiss import VulkanFAISS
# Import other modules as they are created
# from .vulkan_fnn import VulkanFNN
# from .vulkan_attention import VulkanAttention
# from .vulkan_memory import VulkanMemory
# from .vulkan_cells import VulkanCells


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
        # self.fnn = VulkanFNN(self.core, self.pipelines)
        # self.attention = VulkanAttention(self.core, self.pipelines)
        # self.memory = VulkanMemory(self.core, self.pipelines)
        # self.cells = VulkanCells(self.core, self.pipelines)
        
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
    
    # TODO: Delegate other operations to their respective modules
    # For now, methods will be added as modules are created
    
    def __del__(self):
        """Cleanup Vulkan resources"""
        if hasattr(self, 'pipelines'):
            self.pipelines.cleanup()
        if hasattr(self, 'core'):
            self.core.cleanup()

