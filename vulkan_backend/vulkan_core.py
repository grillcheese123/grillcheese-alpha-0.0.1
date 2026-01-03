"""
Core Vulkan initialization, buffer management, and dispatch operations.
"""

import numpy as np
import ctypes
from pathlib import Path
from .base import VULKAN_AVAILABLE

if VULKAN_AVAILABLE:
    from vulkan import *


class VulkanCore:
    """Core Vulkan operations: initialization, buffers, and dispatch"""
    
    def __init__(self, shader_dir: str = "shaders"):
        if not VULKAN_AVAILABLE:
            raise RuntimeError("Vulkan not available")
        
        self.shader_dir = Path(shader_dir)
        self.shaders = self._load_shaders()
        print(f"[OK] Loaded {len(self.shaders)} SPIR-V shaders")
        
        # Initialize Vulkan
        self._init_vulkan()
    
    def _load_shaders(self):
        """Load all .spv files"""
        shaders = {}
        spv_dir = Path(self.shader_dir) / "spv"
        if not spv_dir.exists():
            spv_dir = Path(self.shader_dir)
        
        for spv_file in spv_dir.glob("*.spv"):
            name = spv_file.stem
            with open(spv_file, 'rb') as f:
                shaders[name] = f.read()
        return shaders
    
    def _init_vulkan(self):
        """Initialize Vulkan instance, device, queue"""
        # Create instance
        app_info = VkApplicationInfo(
            sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="GrillCheese",
            applicationVersion=VK_MAKE_VERSION(1, 0, 0),
            pEngineName="SNN-Compute",
            engineVersion=VK_MAKE_VERSION(1, 0, 0),
            apiVersion=VK_API_VERSION_1_0
        )
        
        create_info = VkInstanceCreateInfo(
            sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationInfo=app_info
        )
        
        self.instance = vkCreateInstance(create_info, None)
        
        # Select GPU (prefer AMD)
        physical_devices = vkEnumeratePhysicalDevices(self.instance)
        self.physical_device = self._select_gpu(physical_devices)
        
        # Get device properties
        props = vkGetPhysicalDeviceProperties(self.physical_device)
        print(f"[OK] Using GPU: {props.deviceName}")
        
        # Find compute queue family
        queue_families = vkGetPhysicalDeviceQueueFamilyProperties(self.physical_device)
        compute_queue_family = None
        for i, family in enumerate(queue_families):
            if family.queueFlags & VK_QUEUE_COMPUTE_BIT:
                compute_queue_family = i
                break
        
        if compute_queue_family is None:
            raise RuntimeError("No compute queue found")
        
        # Create logical device
        queue_create_info = VkDeviceQueueCreateInfo(
            sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=compute_queue_family,
            queueCount=1,
            pQueuePriorities=[1.0]
        )
        
        device_create_info = VkDeviceCreateInfo(
            sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue_create_info]
        )
        
        self.device = vkCreateDevice(self.physical_device, device_create_info, None)
        self.queue = vkGetDeviceQueue(self.device, compute_queue_family, 0)
        self.compute_queue_family = compute_queue_family
        
        # Create command pool
        pool_info = VkCommandPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex=compute_queue_family,
            flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
        )
        self.command_pool = vkCreateCommandPool(self.device, pool_info, None)
        
        # Create descriptor pool (large enough for all shaders)
        pool_sizes = [
            VkDescriptorPoolSize(
                type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1000
            )
        ]
        
        pool_info = VkDescriptorPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            flags=VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            maxSets=500,
            poolSizeCount=len(pool_sizes),
            pPoolSizes=pool_sizes
        )
        
        self.descriptor_pool = vkCreateDescriptorPool(self.device, pool_info, None)
        
        print("[OK] Vulkan device initialized")
    
    def _select_gpu(self, devices):
        """Select GPU, prefer AMD"""
        for device in devices:
            props = vkGetPhysicalDeviceProperties(device)
            name = props.deviceName
            if 'AMD' in name or 'Radeon' in name:
                return device
        return devices[0]
    
    def _create_buffer(self, size: int, usage: int):
        """Create Vulkan buffer and allocate memory"""
        buffer_info = VkBufferCreateInfo(
            sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=size,
            usage=usage,
            sharingMode=VK_SHARING_MODE_EXCLUSIVE
        )
        
        buffer = vkCreateBuffer(self.device, buffer_info, None)
        
        # Get memory requirements
        mem_req = vkGetBufferMemoryRequirements(self.device, buffer)
        
        # Allocate memory
        mem_props = vkGetPhysicalDeviceMemoryProperties(self.physical_device)
        mem_type_index = self._find_memory_type(
            mem_req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            mem_props
        )
        
        alloc_info = VkMemoryAllocateInfo(
            sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=mem_req.size,
            memoryTypeIndex=mem_type_index
        )
        
        memory = vkAllocateMemory(self.device, alloc_info, None)
        vkBindBufferMemory(self.device, buffer, memory, 0)
        
        return buffer, memory
    
    def _find_memory_type(self, type_filter, properties, mem_props):
        """Find suitable memory type"""
        for i in range(mem_props.memoryTypeCount):
            if (type_filter & (1 << i)) and \
               (mem_props.memoryTypes[i].propertyFlags & properties) == properties:
                return i
        raise RuntimeError("Failed to find suitable memory type")
    
    def _upload_buffer(self, buffer, memory, data: np.ndarray):
        """Upload numpy array to GPU buffer"""
        data_ptr = vkMapMemory(self.device, memory, 0, data.nbytes, 0)
        memview = memoryview(data_ptr)
        memview[:data.nbytes] = data.tobytes()
        vkUnmapMemory(self.device, memory)
    
    def _download_buffer(self, memory, size: int, dtype=np.float32) -> np.ndarray:
        """Download GPU buffer to numpy array"""
        data_ptr = vkMapMemory(self.device, memory, 0, size, 0)
        memview = memoryview(data_ptr)
        element_size = np.dtype(dtype).itemsize
        count = size // element_size
        result = np.frombuffer(memview, dtype=dtype, count=count).copy()
        vkUnmapMemory(self.device, memory)
        return result
    
    def _dispatch_compute(self, pipeline, pipeline_layout, descriptor_set, 
                         workgroup_x: int, push_constants: bytes = None, 
                         workgroup_y: int = 1, workgroup_z: int = 1):
        """Dispatch compute shader"""
        # Allocate command buffer
        alloc_info = VkCommandBufferAllocateInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.command_pool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1
        )
        
        command_buffer = vkAllocateCommandBuffers(self.device, alloc_info)[0]
        
        # Begin command buffer
        begin_info = VkCommandBufferBeginInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        )
        vkBeginCommandBuffer(command_buffer, begin_info)
        
        # Bind pipeline and descriptor set
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)
        vkCmdBindDescriptorSets(
            command_buffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout,
            0, 1, [descriptor_set],
            0, None
        )
        
        # Push constants if provided
        if push_constants:
            push_buf = ctypes.create_string_buffer(push_constants)
            vkCmdPushConstants(
                command_buffer,
                pipeline_layout,
                VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                len(push_constants),
                ctypes.addressof(push_buf)
            )
        
        # Dispatch
        vkCmdDispatch(command_buffer, workgroup_x, workgroup_y, workgroup_z)
        
        vkEndCommandBuffer(command_buffer)
        
        # Submit
        submit_info = VkSubmitInfo(
            sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[command_buffer]
        )
        
        vkQueueSubmit(self.queue, 1, [submit_info], None)
        vkQueueWaitIdle(self.queue)
        
        # Free command buffer
        vkFreeCommandBuffers(self.device, self.command_pool, 1, [command_buffer])
    
    def cleanup(self):
        """Cleanup Vulkan resources"""
        if hasattr(self, 'device') and self.device:
            if hasattr(self, 'descriptor_pool'):
                vkDestroyDescriptorPool(self.device, self.descriptor_pool, None)
            if hasattr(self, 'command_pool'):
                vkDestroyCommandPool(self.device, self.command_pool, None)
            vkDestroyDevice(self.device, None)
        
        if hasattr(self, 'instance') and self.instance:
            vkDestroyInstance(self.instance, None)

