import numpy as np
from pathlib import Path
import ctypes
import struct

try:
    from vulkan import *
    VULKAN_AVAILABLE = True
except ImportError:
    VULKAN_AVAILABLE = False


class VulkanCompute:
    """Complete Vulkan compute backend with GPU dispatch"""
    
    def __init__(self, shader_dir: str = "shaders"):
        if not VULKAN_AVAILABLE:
            raise RuntimeError("Vulkan not available")
        
        self.shader_dir = Path(shader_dir)
        self.shaders = self._load_shaders()
        print(f"[OK] Loaded {len(self.shaders)} SPIR-V shaders")
        
        # Initialize Vulkan
        self._init_vulkan()
        
        # Shader pipelines cache
        self.pipelines = {}
        self.descriptor_set_layouts = {}
        self.pipeline_layouts = {}
    
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
        # Use VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT to allow freeing sets
        pool_sizes = [
            VkDescriptorPoolSize(
                type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1000  # Increased for multiple descriptor sets
            )
        ]
        
        pool_info = VkDescriptorPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            flags=VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,  # Allow freeing sets
            maxSets=500,  # Increased pool size
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
        # vkMapMemory returns a CFFI buffer object - use memoryview for direct copy
        memview = memoryview(data_ptr)
        memview[:data.nbytes] = data.tobytes()
        vkUnmapMemory(self.device, memory)
    
    def _download_buffer(self, memory, size: int, dtype=np.float32) -> np.ndarray:
        """Download GPU buffer to numpy array"""
        data_ptr = vkMapMemory(self.device, memory, 0, size, 0)
        # vkMapMemory returns a CFFI buffer object - use memoryview to get bytes
        memview = memoryview(data_ptr)
        # Calculate number of elements based on size and dtype
        element_size = np.dtype(dtype).itemsize
        count = size // element_size
        result = np.frombuffer(memview, dtype=dtype, count=count).copy()
        vkUnmapMemory(self.device, memory)
        return result
    
    def _create_descriptor_set_layout(self, num_buffers: int):
        """Create descriptor set layout for storage buffers"""
        bindings = []
        for i in range(num_buffers):
            bindings.append(VkDescriptorSetLayoutBinding(
                binding=i,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=VK_SHADER_STAGE_COMPUTE_BIT
            ))
        
        layout_info = VkDescriptorSetLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=len(bindings),
            pBindings=bindings
        )
        
        return vkCreateDescriptorSetLayout(self.device, layout_info, None)
    
    def _create_pipeline_layout(self, descriptor_set_layout, push_constant_size: int = 0):
        """Create pipeline layout"""
        push_constant_range = None
        if push_constant_size > 0:
            push_constant_range = VkPushConstantRange(
                stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
                offset=0,
                size=push_constant_size
            )
        
        layout_info = VkPipelineLayoutCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=[descriptor_set_layout],
            pushConstantRangeCount=1 if push_constant_range else 0,
            pPushConstantRanges=[push_constant_range] if push_constant_range else None
        )
        
        return vkCreatePipelineLayout(self.device, layout_info, None)
    
    def _create_compute_pipeline(self, shader_code: bytes, pipeline_layout):
        """Create compute pipeline"""
        # Create shader module
        shader_info = VkShaderModuleCreateInfo(
            sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(shader_code),
            pCode=shader_code
        )
        shader_module = vkCreateShaderModule(self.device, shader_info, None)
        
        # Create compute pipeline
        stage_info = VkPipelineShaderStageCreateInfo(
            sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=VK_SHADER_STAGE_COMPUTE_BIT,
            module=shader_module,
            pName="main"
        )
        
        pipeline_info = VkComputePipelineCreateInfo(
            sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=stage_info,
            layout=pipeline_layout
        )
        
        pipeline = vkCreateComputePipelines(self.device, None, 1, [pipeline_info], None)[0]
        
        # Can destroy shader module after pipeline creation
        vkDestroyShaderModule(self.device, shader_module, None)
        
        return pipeline
    
    def _create_descriptor_set(self, layout, buffers: list):
        """Create and populate descriptor set"""
        alloc_info = VkDescriptorSetAllocateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=self.descriptor_pool,
            descriptorSetCount=1,
            pSetLayouts=[layout]
        )
        
        descriptor_set = vkAllocateDescriptorSets(self.device, alloc_info)[0]
        
        # Update descriptor set with buffer bindings
        writes = []
        for i, (buffer, size) in enumerate(buffers):
            buffer_info = VkDescriptorBufferInfo(
                buffer=buffer,
                offset=0,
                range=size
            )
            
            write = VkWriteDescriptorSet(
                sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=descriptor_set,
                dstBinding=i,
                dstArrayElement=0,
                descriptorCount=1,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[buffer_info]
            )
            writes.append(write)
        
        vkUpdateDescriptorSets(self.device, len(writes), writes, 0, None)
        
        return descriptor_set
    
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
            # Create ctypes buffer from bytes and pass its address
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
    
    def lif_step(self, input_current, membrane, refractory, 
                 dt=0.001, tau_mem=20.0, v_thresh=1.0):
        """Run LIF shader on GPU"""
        n = len(input_current)
        
        # Create buffers
        buf_in, mem_in = self._create_buffer(
            input_current.nbytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        buf_mem, mem_mem = self._create_buffer(
            membrane.nbytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        buf_ref, mem_ref = self._create_buffer(
            refractory.nbytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        buf_out, mem_out = self._create_buffer(
            n * 4,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        
        # Upload data
        self._upload_buffer(buf_in, mem_in, input_current.astype(np.float32))
        self._upload_buffer(buf_mem, mem_mem, membrane.astype(np.float32))
        self._upload_buffer(buf_ref, mem_ref, refractory.astype(np.float32))
        
        # Create or get cached pipeline
        if 'lif-neuron' not in self.pipelines:
            desc_layout = self._create_descriptor_set_layout(4)  # 4 buffers
            pipe_layout = self._create_pipeline_layout(desc_layout, push_constant_size=32)
            pipeline = self._create_compute_pipeline(
                self.shaders['lif-neuron'],
                pipe_layout
            )
            
            self.descriptor_set_layouts['lif-neuron'] = desc_layout
            self.pipeline_layouts['lif-neuron'] = pipe_layout
            self.pipelines['lif-neuron'] = pipeline
        
        # Create descriptor set
        descriptor_set = self._create_descriptor_set(
            self.descriptor_set_layouts['lif-neuron'],
            [
                (buf_in, input_current.nbytes),
                (buf_mem, membrane.nbytes),
                (buf_ref, refractory.nbytes),
                (buf_out, n * 4)
            ]
        )
        
        # Pack push constants (matches shader layout)
        push_constants = struct.pack(
            'Ifffffff',
            n,                # n_neurons
            dt,               # dt
            tau_mem,          # tau_mem
            0.0,              # V_rest
            0.0,              # V_reset
            v_thresh,         # V_thresh
            1.0,              # R_mem
            2.0               # t_refrac_period
        )
        
        # Dispatch (256 threads per workgroup from shader)
        workgroups = (n + 255) // 256
        self._dispatch_compute(
            self.pipelines['lif-neuron'],
            self.pipeline_layouts['lif-neuron'],
            descriptor_set,
            workgroups,
            push_constants
        )
        
        # Download results
        membrane_out = self._download_buffer(mem_mem, membrane.nbytes)
        refractory_out = self._download_buffer(mem_ref, refractory.nbytes)
        spikes_out = self._download_buffer(mem_out, n * 4)
        
        # Free descriptor set (after command buffer has finished)
        vkFreeDescriptorSets(self.device, self.descriptor_pool, 1, [descriptor_set])
        
        # Cleanup
        vkDestroyBuffer(self.device, buf_in, None)
        vkDestroyBuffer(self.device, buf_mem, None)
        vkDestroyBuffer(self.device, buf_ref, None)
        vkDestroyBuffer(self.device, buf_out, None)
        vkFreeMemory(self.device, mem_in, None)
        vkFreeMemory(self.device, mem_mem, None)
        vkFreeMemory(self.device, mem_ref, None)
        vkFreeMemory(self.device, mem_out, None)
        
        return membrane_out, refractory_out, spikes_out
    
    def hebbian_learning(self, pre_activations, post_activations, weights,
                         learning_rate=0.01, weight_decay=0.0):
        """
        Apply Hebbian learning rule: ΔW = η * <pre * post> - λ * W
        
        Args:
            pre_activations: (batch, time, pre_dim) array
            post_activations: (batch, time, post_dim) array
            weights: (post_dim, pre_dim) array to update
            learning_rate: Learning rate η
            weight_decay: Weight decay coefficient λ
        
        Returns:
            Updated weights array
        """
        batch_size, time_steps, pre_dim = pre_activations.shape
        _, _, post_dim = post_activations.shape
        
        # Flatten activations
        pre_flat = pre_activations.astype(np.float32).flatten()
        post_flat = post_activations.astype(np.float32).flatten()
        weights_flat = weights.astype(np.float32).flatten()
        
        # Create buffers
        buf_pre, mem_pre = self._create_buffer(pre_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_post, mem_post = self._create_buffer(post_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_weights, mem_weights = self._create_buffer(weights_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self._upload_buffer(buf_pre, mem_pre, pre_flat)
        self._upload_buffer(buf_post, mem_post, post_flat)
        self._upload_buffer(buf_weights, mem_weights, weights_flat)
        
        # Create or get cached pipeline
        if 'hebbian-learning' not in self.pipelines:
            desc_layout = self._create_descriptor_set_layout(3)
            pipe_layout = self._create_pipeline_layout(desc_layout, push_constant_size=32)
            pipeline = self._create_compute_pipeline(
                self.shaders['hebbian-learning'],
                pipe_layout
            )
            
            self.descriptor_set_layouts['hebbian-learning'] = desc_layout
            self.pipeline_layouts['hebbian-learning'] = pipe_layout
            self.pipelines['hebbian-learning'] = pipeline
        
        # Create descriptor set
        descriptor_set = self._create_descriptor_set(
            self.descriptor_set_layouts['hebbian-learning'],
            [
                (buf_pre, pre_flat.nbytes),
                (buf_post, post_flat.nbytes),
                (buf_weights, weights_flat.nbytes)
            ]
        )
        
        # Pack push constants
        push_constants = struct.pack(
            'IIIIff',
            batch_size,
            time_steps,
            pre_dim,
            post_dim,
            learning_rate,
            weight_decay
        )
        
        # Dispatch (16x16 workgroup from shader)
        # Shader uses post_idx = gl_GlobalInvocationID.y, pre_idx = gl_GlobalInvocationID.x
        workgroups_x = (pre_dim + 15) // 16
        workgroups_y = (post_dim + 15) // 16
        self._dispatch_compute(
            self.pipelines['hebbian-learning'],
            self.pipeline_layouts['hebbian-learning'],
            descriptor_set,
            workgroups_x,
            push_constants,
            workgroups_y,
            1
        )
        
        # Download results
        weights_out = self._download_buffer(mem_weights, weights_flat.nbytes, dtype=np.float32)
        weights_out = weights_out[:post_dim * pre_dim].reshape(post_dim, pre_dim)
        
        # Free descriptor set
        vkFreeDescriptorSets(self.device, self.descriptor_pool, 1, [descriptor_set])
        
        # Cleanup
        vkDestroyBuffer(self.device, buf_pre, None)
        vkDestroyBuffer(self.device, buf_post, None)
        vkDestroyBuffer(self.device, buf_weights, None)
        vkFreeMemory(self.device, mem_pre, None)
        vkFreeMemory(self.device, mem_post, None)
        vkFreeMemory(self.device, mem_weights, None)
        
        return weights_out
    
    def stdp_learning(self, pre_activations, post_activations, weights,
                      pre_trace, post_trace,
                      lr_potentiation=0.01, lr_depression=0.01, trace_decay=0.9):
        """
        Apply STDP learning rule with eligibility traces
        
        Args:
            pre_activations: (batch, time, pre_dim) array
            post_activations: (batch, time, post_dim) array
            weights: (post_dim, pre_dim) array to update
            pre_trace: (batch, pre_dim) array - eligibility traces
            post_trace: (batch, post_dim) array - eligibility traces
            lr_potentiation: LTP learning rate
            lr_depression: LTD learning rate
            trace_decay: Exponential trace decay factor
        
        Returns:
            Updated weights array, updated pre_trace, updated post_trace
        """
        batch_size, time_steps, pre_dim = pre_activations.shape
        _, _, post_dim = post_activations.shape
        
        # Flatten activations
        pre_flat = pre_activations.astype(np.float32).flatten()
        post_flat = post_activations.astype(np.float32).flatten()
        weights_flat = weights.astype(np.float32).flatten()
        pre_trace_flat = pre_trace.astype(np.float32).flatten()
        post_trace_flat = post_trace.astype(np.float32).flatten()
        
        # Create buffers
        buf_pre, mem_pre = self._create_buffer(pre_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_post, mem_post = self._create_buffer(post_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_weights, mem_weights = self._create_buffer(weights_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_pre_trace, mem_pre_trace = self._create_buffer(pre_trace_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_post_trace, mem_post_trace = self._create_buffer(post_trace_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self._upload_buffer(buf_pre, mem_pre, pre_flat)
        self._upload_buffer(buf_post, mem_post, post_flat)
        self._upload_buffer(buf_weights, mem_weights, weights_flat)
        self._upload_buffer(buf_pre_trace, mem_pre_trace, pre_trace_flat)
        self._upload_buffer(buf_post_trace, mem_post_trace, post_trace_flat)
        
        # Create or get cached pipeline
        if 'stdp-learning' not in self.pipelines:
            desc_layout = self._create_descriptor_set_layout(5)
            pipe_layout = self._create_pipeline_layout(desc_layout, push_constant_size=32)
            pipeline = self._create_compute_pipeline(
                self.shaders['stdp-learning'],
                pipe_layout
            )
            
            self.descriptor_set_layouts['stdp-learning'] = desc_layout
            self.pipeline_layouts['stdp-learning'] = pipe_layout
            self.pipelines['stdp-learning'] = pipeline
        
        # Create descriptor set
        descriptor_set = self._create_descriptor_set(
            self.descriptor_set_layouts['stdp-learning'],
            [
                (buf_pre, pre_flat.nbytes),
                (buf_post, post_flat.nbytes),
                (buf_weights, weights_flat.nbytes),
                (buf_pre_trace, pre_trace_flat.nbytes),
                (buf_post_trace, post_trace_flat.nbytes)
            ]
        )
        
        # Pack push constants (uint uint uint uint float float float uint = 32 bytes)
        push_constants = struct.pack(
            'IIIIfffI',
            batch_size,
            time_steps,
            pre_dim,
            post_dim,
            lr_potentiation,
            lr_depression,
            trace_decay,
            0  # pass_type: 0 = update traces
        )
        
        # Dispatch pass 1: Update traces (16x16 workgroup)
        # For trace update: row (y) < batch_size, col (x) < max(pre_dim, post_dim)
        workgroups_x = (max(pre_dim, post_dim) + 15) // 16
        workgroups_y = (batch_size + 15) // 16
        self._dispatch_compute(
            self.pipelines['stdp-learning'],
            self.pipeline_layouts['stdp-learning'],
            descriptor_set,
            workgroups_x,
            push_constants,
            workgroups_y,
            1
        )
        
        # Download updated traces
        pre_trace_out = self._download_buffer(mem_pre_trace, pre_trace_flat.nbytes, dtype=np.float32)
        post_trace_out = self._download_buffer(mem_post_trace, post_trace_flat.nbytes, dtype=np.float32)
        pre_trace_out = pre_trace_out[:batch_size * pre_dim].reshape(batch_size, pre_dim)
        post_trace_out = post_trace_out[:batch_size * post_dim].reshape(batch_size, post_dim)
        
        # Upload updated traces back for pass 2
        self._upload_buffer(buf_pre_trace, mem_pre_trace, pre_trace_out.flatten())
        self._upload_buffer(buf_post_trace, mem_post_trace, post_trace_out.flatten())
        
        # Pack push constants for pass 2
        push_constants = struct.pack(
            'IIIIfffI',
            batch_size,
            time_steps,
            pre_dim,
            post_dim,
            lr_potentiation,
            lr_depression,
            trace_decay,
            1  # pass_type: 1 = update weights
        )
        
        # Dispatch pass 2: Update weights (16x16 workgroup)
        workgroups_x = (pre_dim + 15) // 16
        workgroups_y = (post_dim + 15) // 16
        self._dispatch_compute(
            self.pipelines['stdp-learning'],
            self.pipeline_layouts['stdp-learning'],
            descriptor_set,
            workgroups_x,
            push_constants,
            workgroups_y,
            1
        )
        
        # Download results
        weights_out = self._download_buffer(mem_weights, weights_flat.nbytes, dtype=np.float32)
        weights_out = weights_out[:post_dim * pre_dim].reshape(post_dim, pre_dim)
        pre_trace_out = self._download_buffer(mem_pre_trace, pre_trace_flat.nbytes, dtype=np.float32)
        post_trace_out = self._download_buffer(mem_post_trace, post_trace_flat.nbytes, dtype=np.float32)
        pre_trace_out = pre_trace_out[:batch_size * pre_dim].reshape(batch_size, pre_dim)
        post_trace_out = post_trace_out[:batch_size * post_dim].reshape(batch_size, post_dim)
        
        # Free descriptor set
        vkFreeDescriptorSets(self.device, self.descriptor_pool, 1, [descriptor_set])
        
        # Cleanup
        vkDestroyBuffer(self.device, buf_pre, None)
        vkDestroyBuffer(self.device, buf_post, None)
        vkDestroyBuffer(self.device, buf_weights, None)
        vkDestroyBuffer(self.device, buf_pre_trace, None)
        vkDestroyBuffer(self.device, buf_post_trace, None)
        vkFreeMemory(self.device, mem_pre, None)
        vkFreeMemory(self.device, mem_post, None)
        vkFreeMemory(self.device, mem_weights, None)
        vkFreeMemory(self.device, mem_pre_trace, None)
        vkFreeMemory(self.device, mem_post_trace, None)
        
        return weights_out, pre_trace_out, post_trace_out
    
    def activation_relu(self, input_data):
        """Apply ReLU activation: max(0, x)"""
        data = input_data.astype(np.float32).flatten()
        total_elements = len(data)
        
        # Create buffers
        buf_in, mem_in = self._create_buffer(data.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_out, mem_out = self._create_buffer(data.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self._upload_buffer(buf_in, mem_in, data)
        
        # Create or get cached pipeline
        if 'activation-relu' not in self.pipelines:
            desc_layout = self._create_descriptor_set_layout(2)
            pipe_layout = self._create_pipeline_layout(desc_layout, push_constant_size=4)
            pipeline = self._create_compute_pipeline(
                self.shaders['activation-relu'],
                pipe_layout
            )
            
            self.descriptor_set_layouts['activation-relu'] = desc_layout
            self.pipeline_layouts['activation-relu'] = pipe_layout
            self.pipelines['activation-relu'] = pipeline
        
        # Create descriptor set
        descriptor_set = self._create_descriptor_set(
            self.descriptor_set_layouts['activation-relu'],
            [(buf_in, data.nbytes), (buf_out, data.nbytes)]
        )
        
        # Pack push constants
        push_constants = struct.pack('I', total_elements)
        
        # Dispatch
        workgroups = (total_elements + 255) // 256
        self._dispatch_compute(
            self.pipelines['activation-relu'],
            self.pipeline_layouts['activation-relu'],
            descriptor_set,
            workgroups,
            push_constants
        )
        
        # Download results
        result = self._download_buffer(mem_out, data.nbytes, dtype=np.float32)
        result = result[:total_elements]
        
        # Free descriptor set
        vkFreeDescriptorSets(self.device, self.descriptor_pool, 1, [descriptor_set])
        
        # Cleanup
        vkDestroyBuffer(self.device, buf_in, None)
        vkDestroyBuffer(self.device, buf_out, None)
        vkFreeMemory(self.device, mem_in, None)
        vkFreeMemory(self.device, mem_out, None)
        
        return result.reshape(input_data.shape) if input_data.ndim > 1 else result
    
    def activation_gelu(self, input_data):
        """Apply GELU activation"""
        data = input_data.astype(np.float32).flatten()
        total_elements = len(data)
        
        # Create buffers
        buf_in, mem_in = self._create_buffer(data.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_out, mem_out = self._create_buffer(data.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self._upload_buffer(buf_in, mem_in, data)
        
        # Create or get cached pipeline
        if 'activation-gelu' not in self.pipelines:
            desc_layout = self._create_descriptor_set_layout(2)
            pipe_layout = self._create_pipeline_layout(desc_layout, push_constant_size=4)
            pipeline = self._create_compute_pipeline(
                self.shaders['activation-gelu'],
                pipe_layout
            )
            
            self.descriptor_set_layouts['activation-gelu'] = desc_layout
            self.pipeline_layouts['activation-gelu'] = pipe_layout
            self.pipelines['activation-gelu'] = pipeline
        
        # Create descriptor set
        descriptor_set = self._create_descriptor_set(
            self.descriptor_set_layouts['activation-gelu'],
            [(buf_in, data.nbytes), (buf_out, data.nbytes)]
        )
        
        # Pack push constants
        push_constants = struct.pack('I', total_elements)
        
        # Dispatch
        workgroups = (total_elements + 255) // 256
        self._dispatch_compute(
            self.pipelines['activation-gelu'],
            self.pipeline_layouts['activation-gelu'],
            descriptor_set,
            workgroups,
            push_constants
        )
        
        # Download results
        result = self._download_buffer(mem_out, data.nbytes, dtype=np.float32)
        result = result[:total_elements]
        
        # Free descriptor set
        vkFreeDescriptorSets(self.device, self.descriptor_pool, 1, [descriptor_set])
        
        # Cleanup
        vkDestroyBuffer(self.device, buf_in, None)
        vkDestroyBuffer(self.device, buf_out, None)
        vkFreeMemory(self.device, mem_in, None)
        vkFreeMemory(self.device, mem_out, None)
        
        return result.reshape(input_data.shape) if input_data.ndim > 1 else result
    
    def activation_silu(self, input_data):
        """Apply SiLU (Swish) activation: x * sigmoid(x)"""
        data = input_data.astype(np.float32).flatten()
        total_elements = len(data)
        
        # Create buffers
        buf_in, mem_in = self._create_buffer(data.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_out, mem_out = self._create_buffer(data.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self._upload_buffer(buf_in, mem_in, data)
        
        # Create or get cached pipeline
        if 'activation-silu' not in self.pipelines:
            desc_layout = self._create_descriptor_set_layout(2)
            pipe_layout = self._create_pipeline_layout(desc_layout, push_constant_size=4)
            pipeline = self._create_compute_pipeline(
                self.shaders['activation-silu'],
                pipe_layout
            )
            
            self.descriptor_set_layouts['activation-silu'] = desc_layout
            self.pipeline_layouts['activation-silu'] = pipe_layout
            self.pipelines['activation-silu'] = pipeline
        
        # Create descriptor set
        descriptor_set = self._create_descriptor_set(
            self.descriptor_set_layouts['activation-silu'],
            [(buf_in, data.nbytes), (buf_out, data.nbytes)]
        )
        
        # Pack push constants
        push_constants = struct.pack('I', total_elements)
        
        # Dispatch
        workgroups = (total_elements + 255) // 256
        self._dispatch_compute(
            self.pipelines['activation-silu'],
            self.pipeline_layouts['activation-silu'],
            descriptor_set,
            workgroups,
            push_constants
        )
        
        # Download results
        result = self._download_buffer(mem_out, data.nbytes, dtype=np.float32)
        result = result[:total_elements]
        
        # Free descriptor set
        vkFreeDescriptorSets(self.device, self.descriptor_pool, 1, [descriptor_set])
        
        # Cleanup
        vkDestroyBuffer(self.device, buf_in, None)
        vkDestroyBuffer(self.device, buf_out, None)
        vkFreeMemory(self.device, mem_in, None)
        vkFreeMemory(self.device, mem_out, None)
        
        return result.reshape(input_data.shape) if input_data.ndim > 1 else result
    
    def activation_softmax(self, input_data, axis=-1):
        """
        Apply softmax activation along specified axis
        
        Args:
            input_data: Input array (can be 1D, 2D, or 3D)
            axis: Axis to normalize over (default: -1, last dimension)
        
        Returns:
            Softmax output with same shape as input
        """
        data = input_data.astype(np.float32)
        original_shape = data.shape
        
        # Handle different input shapes
        if data.ndim == 1:
            batch_size, seq_len, features = 1, 1, len(data)
            data = data.reshape(1, 1, -1)
        elif data.ndim == 2:
            batch_size, seq_len, features = data.shape[0], 1, data.shape[1]
            data = data.reshape(data.shape[0], 1, -1)
        else:
            batch_size, seq_len, features = data.shape
        
        data_flat = data.flatten()
        
        # Create buffers
        buf_in, mem_in = self._create_buffer(data_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_out, mem_out = self._create_buffer(data_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_max, mem_max = self._create_buffer(batch_size * seq_len * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_sum, mem_sum = self._create_buffer(batch_size * seq_len * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self._upload_buffer(buf_in, mem_in, data_flat)
        
        # Create or get cached pipeline
        if 'activation-softmax' not in self.pipelines:
            desc_layout = self._create_descriptor_set_layout(4)
            pipe_layout = self._create_pipeline_layout(desc_layout, push_constant_size=24)
            pipeline = self._create_compute_pipeline(
                self.shaders['activation-softmax'],
                pipe_layout
            )
            
            self.descriptor_set_layouts['activation-softmax'] = desc_layout
            self.pipeline_layouts['activation-softmax'] = pipe_layout
            self.pipelines['activation-softmax'] = pipeline
        
        # Create descriptor set
        descriptor_set = self._create_descriptor_set(
            self.descriptor_set_layouts['activation-softmax'],
            [
                (buf_in, data_flat.nbytes),
                (buf_out, data_flat.nbytes),
                (buf_max, batch_size * seq_len * 4),
                (buf_sum, batch_size * seq_len * 4)
            ]
        )
        
        # Pack push constants
        push_constants = struct.pack('IIIII', batch_size, seq_len, features, 0, features)
        
        # Pass 1: Compute max
        workgroups = ((batch_size * seq_len) + 255) // 256
        self._dispatch_compute(
            self.pipelines['activation-softmax'],
            self.pipeline_layouts['activation-softmax'],
            descriptor_set,
            workgroups,
            push_constants
        )
        
        # Pass 2: Compute sum of exponentials
        push_constants = struct.pack('IIIII', batch_size, seq_len, features, 1, features)
        self._dispatch_compute(
            self.pipelines['activation-softmax'],
            self.pipeline_layouts['activation-softmax'],
            descriptor_set,
            workgroups,
            push_constants
        )
        
        # Pass 3: Normalize
        push_constants = struct.pack('IIIII', batch_size, seq_len, features, 2, features)
        workgroups = (len(data_flat) + 255) // 256
        self._dispatch_compute(
            self.pipelines['activation-softmax'],
            self.pipeline_layouts['activation-softmax'],
            descriptor_set,
            workgroups,
            push_constants
        )
        
        # Download results
        result = self._download_buffer(mem_out, data_flat.nbytes, dtype=np.float32)
        result = result[:len(data_flat)].reshape(original_shape)
        
        # Free descriptor set
        vkFreeDescriptorSets(self.device, self.descriptor_pool, 1, [descriptor_set])
        
        # Cleanup
        vkDestroyBuffer(self.device, buf_in, None)
        vkDestroyBuffer(self.device, buf_out, None)
        vkDestroyBuffer(self.device, buf_max, None)
        vkDestroyBuffer(self.device, buf_sum, None)
        vkFreeMemory(self.device, mem_in, None)
        vkFreeMemory(self.device, mem_out, None)
        vkFreeMemory(self.device, mem_max, None)
        vkFreeMemory(self.device, mem_sum, None)
        
        return result
    
    def layernorm(self, input_data, gamma, beta, eps=1e-5):
        """
        Apply Layer Normalization: (x - mean) / sqrt(var + eps) * gamma + beta
        
        Args:
            input_data: Input array (batch, seq_len, features)
            gamma: Scale parameters (features,)
            beta: Shift parameters (features,)
            eps: Small constant for numerical stability
        
        Returns:
            Normalized output with same shape as input
        """
        data = input_data.astype(np.float32)
        gamma_arr = gamma.astype(np.float32).flatten()
        beta_arr = beta.astype(np.float32).flatten()
        
        if data.ndim == 2:
            batch_size, seq_len, features = data.shape[0], 1, data.shape[1]
            data = data.reshape(data.shape[0], 1, -1)
        else:
            batch_size, seq_len, features = data.shape
        
        data_flat = data.flatten()
        
        # Create buffers
        buf_in, mem_in = self._create_buffer(data_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_out, mem_out = self._create_buffer(data_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_gamma, mem_gamma = self._create_buffer(gamma_arr.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_beta, mem_beta = self._create_buffer(beta_arr.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_mean, mem_mean = self._create_buffer(batch_size * seq_len * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_var, mem_var = self._create_buffer(batch_size * seq_len * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self._upload_buffer(buf_in, mem_in, data_flat)
        self._upload_buffer(buf_gamma, mem_gamma, gamma_arr)
        self._upload_buffer(buf_beta, mem_beta, beta_arr)
        
        # Create or get cached pipeline
        if 'fnn-layernorm' not in self.pipelines:
            desc_layout = self._create_descriptor_set_layout(6)
            pipe_layout = self._create_pipeline_layout(desc_layout, push_constant_size=20)
            pipeline = self._create_compute_pipeline(
                self.shaders['fnn-layernorm'],
                pipe_layout
            )
            
            self.descriptor_set_layouts['fnn-layernorm'] = desc_layout
            self.pipeline_layouts['fnn-layernorm'] = pipe_layout
            self.pipelines['fnn-layernorm'] = pipeline
        
        # Create descriptor set
        descriptor_set = self._create_descriptor_set(
            self.descriptor_set_layouts['fnn-layernorm'],
            [
                (buf_in, data_flat.nbytes),
                (buf_out, data_flat.nbytes),
                (buf_gamma, gamma_arr.nbytes),
                (buf_beta, beta_arr.nbytes),
                (buf_mean, batch_size * seq_len * 4),
                (buf_var, batch_size * seq_len * 4)
            ]
        )
        
        # Pass 1: Compute mean
        push_constants = struct.pack('IIIfI', batch_size, seq_len, features, eps, 0)
        workgroups = ((batch_size * seq_len) + 255) // 256
        self._dispatch_compute(
            self.pipelines['fnn-layernorm'],
            self.pipeline_layouts['fnn-layernorm'],
            descriptor_set,
            workgroups,
            push_constants
        )
        
        # Pass 2: Compute variance
        push_constants = struct.pack('IIIfI', batch_size, seq_len, features, eps, 1)
        self._dispatch_compute(
            self.pipelines['fnn-layernorm'],
            self.pipeline_layouts['fnn-layernorm'],
            descriptor_set,
            workgroups,
            push_constants
        )
        
        # Pass 3: Normalize
        push_constants = struct.pack('IIIfI', batch_size, seq_len, features, eps, 2)
        workgroups = (len(data_flat) + 255) // 256
        self._dispatch_compute(
            self.pipelines['fnn-layernorm'],
            self.pipeline_layouts['fnn-layernorm'],
            descriptor_set,
            workgroups,
            push_constants
        )
        
        # Download results
        result = self._download_buffer(mem_out, data_flat.nbytes, dtype=np.float32)
        result = result[:len(data_flat)].reshape(data.shape)
        
        # Free descriptor set
        vkFreeDescriptorSets(self.device, self.descriptor_pool, 1, [descriptor_set])
        
        # Cleanup
        vkDestroyBuffer(self.device, buf_in, None)
        vkDestroyBuffer(self.device, buf_out, None)
        vkDestroyBuffer(self.device, buf_gamma, None)
        vkDestroyBuffer(self.device, buf_beta, None)
        vkDestroyBuffer(self.device, buf_mean, None)
        vkDestroyBuffer(self.device, buf_var, None)
        vkFreeMemory(self.device, mem_in, None)
        vkFreeMemory(self.device, mem_out, None)
        vkFreeMemory(self.device, mem_gamma, None)
        vkFreeMemory(self.device, mem_beta, None)
        vkFreeMemory(self.device, mem_mean, None)
        vkFreeMemory(self.device, mem_var, None)
        
        return result
    
    def linear(self, input_data, weights, bias=None):
        """
        Apply linear transformation: output = input @ weights^T + bias
        
        Args:
            input_data: Input array (batch, seq_len, input_dim) or (batch * seq_len, input_dim)
            weights: Weight matrix (output_dim, input_dim)
            bias: Optional bias vector (output_dim,)
        
        Returns:
            Output array (batch, seq_len, output_dim)
        """
        data = input_data.astype(np.float32)
        weights_arr = weights.astype(np.float32)
        
        # Handle input shape
        if data.ndim == 2:
            batch_seq = data.shape[0]
            input_dim = data.shape[1]
            batch_size = batch_seq
            seq_len = 1
        else:
            batch_size, seq_len, input_dim = data.shape
            batch_seq = batch_size * seq_len
            data = data.reshape(batch_seq, input_dim)
        
        output_dim = weights_arr.shape[0]
        data_flat = data.flatten()
        weights_flat = weights_arr.flatten()
        
        # Create buffers
        buf_in, mem_in = self._create_buffer(data_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_weights, mem_weights = self._create_buffer(weights_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_out, mem_out = self._create_buffer(batch_seq * output_dim * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self._upload_buffer(buf_in, mem_in, data_flat)
        self._upload_buffer(buf_weights, mem_weights, weights_flat)
        
        buffers = [(buf_in, data_flat.nbytes), (buf_weights, weights_flat.nbytes), (buf_out, batch_seq * output_dim * 4)]
        
        if bias is not None:
            bias_arr = bias.astype(np.float32).flatten()
            buf_bias, mem_bias = self._create_buffer(bias_arr.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
            self._upload_buffer(buf_bias, mem_bias, bias_arr)
            buffers.append((buf_bias, bias_arr.nbytes))
            has_bias = 1
        else:
            buf_bias = None
            mem_bias = None
            has_bias = 0
        
        # Create or get cached pipeline
        if 'fnn-linear' not in self.pipelines:
            desc_layout = self._create_descriptor_set_layout(len(buffers))
            pipe_layout = self._create_pipeline_layout(desc_layout, push_constant_size=16)
            pipeline = self._create_compute_pipeline(
                self.shaders['fnn-linear'],
                pipe_layout
            )
            
            self.descriptor_set_layouts['fnn-linear'] = desc_layout
            self.pipeline_layouts['fnn-linear'] = pipe_layout
            self.pipelines['fnn-linear'] = pipeline
        
        # Create descriptor set
        descriptor_set = self._create_descriptor_set(
            self.descriptor_set_layouts['fnn-linear'],
            buffers
        )
        
        # Pack push constants
        push_constants = struct.pack('IIII', batch_seq, input_dim, output_dim, has_bias)
        
        # Dispatch (2D: batch_seq x output_dim)
        workgroups_x = (output_dim + 15) // 16
        workgroups_y = (batch_seq + 15) // 16
        self._dispatch_compute(
            self.pipelines['fnn-linear'],
            self.pipeline_layouts['fnn-linear'],
            descriptor_set,
            workgroups_x,
            push_constants=push_constants,
            workgroup_y=workgroups_y
        )
        
        # Download results
        result = self._download_buffer(mem_out, batch_seq * output_dim * 4, dtype=np.float32)
        result = result[:batch_seq * output_dim].reshape(batch_seq, output_dim)
        
        # Free descriptor set
        vkFreeDescriptorSets(self.device, self.descriptor_pool, 1, [descriptor_set])
        
        # Cleanup
        vkDestroyBuffer(self.device, buf_in, None)
        vkDestroyBuffer(self.device, buf_weights, None)
        vkDestroyBuffer(self.device, buf_out, None)
        vkFreeMemory(self.device, mem_in, None)
        vkFreeMemory(self.device, mem_weights, None)
        vkFreeMemory(self.device, mem_out, None)
        if buf_bias is not None:
            vkDestroyBuffer(self.device, buf_bias, None)
            vkFreeMemory(self.device, mem_bias, None)
        
        # Reshape to (batch, seq_len, output_dim) if needed
        if input_data.ndim == 3:
            result = result.reshape(batch_size, seq_len, output_dim)
        
        return result
    
    def attention_scores(self, queries, keys, num_heads, head_dim, scale=None):
        """
        Compute attention scores: Q @ K^T / sqrt(head_dim)
        
        Args:
            queries: Query tensor (batch, seq_len, num_heads, head_dim) or (batch, seq_len, num_heads * head_dim)
            keys: Key tensor (batch, seq_len, num_heads, head_dim) or (batch, seq_len, num_heads * head_dim)
            num_heads: Number of attention heads
            head_dim: Dimension of each head
            scale: Optional scaling factor (default: 1/sqrt(head_dim))
        
        Returns:
            Attention scores (batch, num_heads, seq_len, seq_len)
        """
        q = queries.astype(np.float32)
        k = keys.astype(np.float32)
        
        if scale is None:
            scale = 1.0 / np.sqrt(head_dim)
        
        # Handle flattened head dimension
        if q.ndim == 3:
            batch_size, seq_len, _ = q.shape
            q = q.reshape(batch_size, seq_len, num_heads, head_dim)
            k = k.reshape(batch_size, seq_len, num_heads, head_dim)
        else:
            batch_size, seq_len, num_heads, head_dim = q.shape
        
        q_flat = q.flatten()
        k_flat = k.flatten()
        
        # Create buffers
        buf_q, mem_q = self._create_buffer(q_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_k, mem_k = self._create_buffer(k_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_scores, mem_scores = self._create_buffer(batch_size * num_heads * seq_len * seq_len * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self._upload_buffer(buf_q, mem_q, q_flat)
        self._upload_buffer(buf_k, mem_k, k_flat)
        
        # Create or get cached pipeline
        if 'attention-scores' not in self.pipelines:
            desc_layout = self._create_descriptor_set_layout(4)
            pipe_layout = self._create_pipeline_layout(desc_layout, push_constant_size=24)
            pipeline = self._create_compute_pipeline(
                self.shaders['attention-scores'],
                pipe_layout
            )
            
            self.descriptor_set_layouts['attention-scores'] = desc_layout
            self.pipeline_layouts['attention-scores'] = pipe_layout
            self.pipelines['attention-scores'] = pipeline
        
        # Create dummy V buffer (not used in scores computation, but required by shader)
        buf_v_dummy, mem_v_dummy = self._create_buffer(q_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        self._upload_buffer(buf_v_dummy, mem_v_dummy, q_flat)  # Use Q data as placeholder
        
        # Create descriptor set
        descriptor_set = self._create_descriptor_set(
            self.descriptor_set_layouts['attention-scores'],
            [
                (buf_q, q_flat.nbytes),
                (buf_k, k_flat.nbytes),
                (buf_v_dummy, q_flat.nbytes),
                (buf_scores, batch_size * num_heads * seq_len * seq_len * 4)
            ]
        )
        
        # Pack push constants
        push_constants = struct.pack('IIIIfI', batch_size, seq_len, num_heads, head_dim, scale, 0)
        
        # Dispatch (2D: total_positions x seq_len)
        workgroups_x = (seq_len + 15) // 16
        workgroups_y = ((batch_size * num_heads * seq_len) + 15) // 16
        self._dispatch_compute(
            self.pipelines['attention-scores'],
            self.pipeline_layouts['attention-scores'],
            descriptor_set,
            workgroups_x,
            push_constants=push_constants,
            workgroup_y=workgroups_y
        )
        
        # Download results
        result = self._download_buffer(mem_scores, batch_size * num_heads * seq_len * seq_len * 4, dtype=np.float32)
        result = result[:batch_size * num_heads * seq_len * seq_len].reshape(batch_size, num_heads, seq_len, seq_len)
        
        # Free descriptor set
        vkFreeDescriptorSets(self.device, self.descriptor_pool, 1, [descriptor_set])
        
        # Cleanup
        vkDestroyBuffer(self.device, buf_q, None)
        vkDestroyBuffer(self.device, buf_k, None)
        vkDestroyBuffer(self.device, buf_v_dummy, None)
        vkDestroyBuffer(self.device, buf_scores, None)
        vkFreeMemory(self.device, mem_q, None)
        vkFreeMemory(self.device, mem_k, None)
        vkFreeMemory(self.device, mem_v_dummy, None)
        vkFreeMemory(self.device, mem_scores, None)
        
        return result
    
    def attention_mask(self, attention_scores, use_causal=True, mask_value=-1e9):
        """
        Apply causal mask to attention scores
        
        Args:
            attention_scores: Attention scores (batch, num_heads, seq_len, seq_len)
            use_causal: Whether to apply causal masking
            mask_value: Value to use for masked positions
        
        Returns:
            Masked attention scores
        """
        scores = attention_scores.astype(np.float32)
        batch_size, num_heads, seq_len, _ = scores.shape
        
        scores_flat = scores.flatten()
        
        # Create buffers
        buf_scores, mem_scores = self._create_buffer(scores_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_mask, mem_mask = self._create_buffer(seq_len * seq_len * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Create causal mask (1 = allow, 0 = mask)
        mask = np.ones((seq_len, seq_len), dtype=np.float32)
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                mask[i, j] = 0.0
        
        # Upload data
        self._upload_buffer(buf_scores, mem_scores, scores_flat)
        self._upload_buffer(buf_mask, mem_mask, mask.flatten())
        
        # Create or get cached pipeline
        if 'attention-mask' not in self.pipelines:
            desc_layout = self._create_descriptor_set_layout(2)
            pipe_layout = self._create_pipeline_layout(desc_layout, push_constant_size=20)
            pipeline = self._create_compute_pipeline(
                self.shaders['attention-mask'],
                pipe_layout
            )
            
            self.descriptor_set_layouts['attention-mask'] = desc_layout
            self.pipeline_layouts['attention-mask'] = pipe_layout
            self.pipelines['attention-mask'] = pipeline
        
        # Create descriptor set
        descriptor_set = self._create_descriptor_set(
            self.descriptor_set_layouts['attention-mask'],
            [
                (buf_scores, scores_flat.nbytes),
                (buf_mask, seq_len * seq_len * 4)
            ]
        )
        
        # Pack push constants
        use_mask = 1 if use_causal else 0
        push_constants = struct.pack('IIIIf', batch_size, num_heads, seq_len, use_mask, mask_value)
        
        # Dispatch
        workgroups = ((batch_size * num_heads * seq_len * seq_len) + 255) // 256
        self._dispatch_compute(
            self.pipelines['attention-mask'],
            self.pipeline_layouts['attention-mask'],
            descriptor_set,
            workgroups,
            push_constants
        )
        
        # Download results
        result = self._download_buffer(mem_scores, scores_flat.nbytes, dtype=np.float32)
        result = result[:len(scores_flat)].reshape(scores.shape)
        
        # Free descriptor set
        vkFreeDescriptorSets(self.device, self.descriptor_pool, 1, [descriptor_set])
        
        # Cleanup
        vkDestroyBuffer(self.device, buf_scores, None)
        vkDestroyBuffer(self.device, buf_mask, None)
        vkFreeMemory(self.device, mem_scores, None)
        vkFreeMemory(self.device, mem_mask, None)
        
        return result
    
    def attention_output(self, attention_weights, values, num_heads, head_dim):
        """
        Compute attention output: attention_weights @ values
        
        Args:
            attention_weights: Attention weights after softmax (batch, num_heads, seq_len, seq_len)
            values: Value tensor (batch, seq_len, num_heads, head_dim) or (batch, seq_len, num_heads * head_dim)
            num_heads: Number of attention heads
            head_dim: Dimension of each head
        
        Returns:
            Attention output (batch, seq_len, num_heads, head_dim)
        """
        weights = attention_weights.astype(np.float32)
        v = values.astype(np.float32)
        
        batch_size, num_heads, seq_len, _ = weights.shape
        
        # Handle flattened head dimension
        if v.ndim == 3:
            v = v.reshape(batch_size, seq_len, num_heads, head_dim)
        else:
            batch_size, seq_len, num_heads, head_dim = v.shape
        
        weights_flat = weights.flatten()
        v_flat = v.flatten()
        
        # Create buffers
        buf_weights, mem_weights = self._create_buffer(weights_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_v, mem_v = self._create_buffer(v_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_out, mem_out = self._create_buffer(batch_size * seq_len * num_heads * head_dim * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self._upload_buffer(buf_weights, mem_weights, weights_flat)
        self._upload_buffer(buf_v, mem_v, v_flat)
        
        # Create or get cached pipeline
        if 'attention-output' not in self.pipelines:
            desc_layout = self._create_descriptor_set_layout(3)
            pipe_layout = self._create_pipeline_layout(desc_layout, push_constant_size=16)
            pipeline = self._create_compute_pipeline(
                self.shaders['attention-output'],
                pipe_layout
            )
            
            self.descriptor_set_layouts['attention-output'] = desc_layout
            self.pipeline_layouts['attention-output'] = pipe_layout
            self.pipelines['attention-output'] = pipeline
        
        # Create descriptor set
        descriptor_set = self._create_descriptor_set(
            self.descriptor_set_layouts['attention-output'],
            [
                (buf_weights, weights_flat.nbytes),
                (buf_v, v_flat.nbytes),
                (buf_out, batch_size * seq_len * num_heads * head_dim * 4)
            ]
        )
        
        # Pack push constants
        push_constants = struct.pack('IIII', batch_size, seq_len, num_heads, head_dim)
        
        # Dispatch (2D: total_positions x head_dim)
        workgroups_x = (head_dim + 15) // 16
        workgroups_y = ((batch_size * seq_len * num_heads) + 15) // 16
        self._dispatch_compute(
            self.pipelines['attention-output'],
            self.pipeline_layouts['attention-output'],
            descriptor_set,
            workgroups_x,
            push_constants=push_constants,
            workgroup_y=workgroups_y
        )
        
        # Download results
        result = self._download_buffer(mem_out, batch_size * seq_len * num_heads * head_dim * 4, dtype=np.float32)
        result = result[:batch_size * seq_len * num_heads * head_dim].reshape(batch_size, seq_len, num_heads, head_dim)
        
        # Free descriptor set
        vkFreeDescriptorSets(self.device, self.descriptor_pool, 1, [descriptor_set])
        
        # Cleanup
        vkDestroyBuffer(self.device, buf_weights, None)
        vkDestroyBuffer(self.device, buf_v, None)
        vkDestroyBuffer(self.device, buf_out, None)
        vkFreeMemory(self.device, mem_weights, None)
        vkFreeMemory(self.device, mem_v, None)
        vkFreeMemory(self.device, mem_out, None)
        
        return result
    
    def attention_concat_heads(self, attention_output):
        """
        Concatenate attention heads: (batch, seq_len, num_heads, head_dim) -> (batch, seq_len, num_heads * head_dim)
        
        Args:
            attention_output: Attention output (batch, seq_len, num_heads, head_dim)
        
        Returns:
            Concatenated output (batch, seq_len, num_heads * head_dim)
        """
        data = attention_output.astype(np.float32)
        batch_size, seq_len, num_heads, head_dim = data.shape
        
        data_flat = data.flatten()
        
        # Create buffers
        buf_in, mem_in = self._create_buffer(data_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_out, mem_out = self._create_buffer(batch_size * seq_len * num_heads * head_dim * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self._upload_buffer(buf_in, mem_in, data_flat)
        
        # Create or get cached pipeline
        if 'attention-concat-heads' not in self.pipelines:
            desc_layout = self._create_descriptor_set_layout(2)
            pipe_layout = self._create_pipeline_layout(desc_layout, push_constant_size=16)
            pipeline = self._create_compute_pipeline(
                self.shaders['attention-concat-heads'],
                pipe_layout
            )
            
            self.descriptor_set_layouts['attention-concat-heads'] = desc_layout
            self.pipeline_layouts['attention-concat-heads'] = pipe_layout
            self.pipelines['attention-concat-heads'] = pipeline
        
        # Create descriptor set
        descriptor_set = self._create_descriptor_set(
            self.descriptor_set_layouts['attention-concat-heads'],
            [
                (buf_in, data_flat.nbytes),
                (buf_out, batch_size * seq_len * num_heads * head_dim * 4)
            ]
        )
        
        # Pack push constants
        push_constants = struct.pack('IIII', batch_size, seq_len, num_heads, head_dim)
        
        # Dispatch
        workgroups = ((batch_size * seq_len * num_heads * head_dim) + 255) // 256
        self._dispatch_compute(
            self.pipelines['attention-concat-heads'],
            self.pipeline_layouts['attention-concat-heads'],
            descriptor_set,
            workgroups,
            push_constants
        )
        
        # Download results
        result = self._download_buffer(mem_out, batch_size * seq_len * num_heads * head_dim * 4, dtype=np.float32)
        result = result[:batch_size * seq_len * num_heads * head_dim].reshape(batch_size, seq_len, num_heads * head_dim)
        
        # Free descriptor set
        vkFreeDescriptorSets(self.device, self.descriptor_pool, 1, [descriptor_set])
        
        # Cleanup
        vkDestroyBuffer(self.device, buf_in, None)
        vkDestroyBuffer(self.device, buf_out, None)
        vkFreeMemory(self.device, mem_in, None)
        vkFreeMemory(self.device, mem_out, None)
        
        return result
    
    def memory_query_pooling(self, hidden_states, query_weights, query_bias):
        """
        Generate memory queries by mean pooling hidden states and applying linear projection
        
        Args:
            hidden_states: Hidden states (batch, seq_len, dim)
            query_weights: Query projection weights (dim, dim)
            query_bias: Query projection bias (dim,)
        
        Returns:
            Query vectors (batch, dim)
        """
        hidden = hidden_states.astype(np.float32)
        weights = query_weights.astype(np.float32)
        bias = query_bias.astype(np.float32)
        
        batch_size, seq_len, dim = hidden.shape
        
        hidden_flat = hidden.flatten()
        weights_flat = weights.flatten()
        bias_flat = bias.flatten()
        
        # Create buffers
        buf_hidden, mem_hidden = self._create_buffer(hidden_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_weights, mem_weights = self._create_buffer(weights_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_bias, mem_bias = self._create_buffer(bias_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_queries, mem_queries = self._create_buffer(batch_size * dim * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self._upload_buffer(buf_hidden, mem_hidden, hidden_flat)
        self._upload_buffer(buf_weights, mem_weights, weights_flat)
        self._upload_buffer(buf_bias, mem_bias, bias_flat)
        
        # Create or get cached pipeline
        if 'memory-query-pooling' not in self.pipelines:
            desc_layout = self._create_descriptor_set_layout(4)
            pipe_layout = self._create_pipeline_layout(desc_layout, push_constant_size=12)
            pipeline = self._create_compute_pipeline(
                self.shaders['memory-query-pooling'],
                pipe_layout
            )
            
            self.descriptor_set_layouts['memory-query-pooling'] = desc_layout
            self.pipeline_layouts['memory-query-pooling'] = pipe_layout
            self.pipelines['memory-query-pooling'] = pipeline
        
        # Create descriptor set
        descriptor_set = self._create_descriptor_set(
            self.descriptor_set_layouts['memory-query-pooling'],
            [
                (buf_hidden, hidden_flat.nbytes),
                (buf_weights, weights_flat.nbytes),
                (buf_bias, bias_flat.nbytes),
                (buf_queries, batch_size * dim * 4)
            ]
        )
        
        # Pack push constants
        push_constants = struct.pack('III', batch_size, seq_len, dim)
        
        # Dispatch
        workgroups = ((batch_size * dim) + 255) // 256
        self._dispatch_compute(
            self.pipelines['memory-query-pooling'],
            self.pipeline_layouts['memory-query-pooling'],
            descriptor_set,
            workgroups,
            push_constants
        )
        
        # Download results
        result = self._download_buffer(mem_queries, batch_size * dim * 4, dtype=np.float32)
        result = result[:batch_size * dim].reshape(batch_size, dim)
        
        # Free descriptor set
        vkFreeDescriptorSets(self.device, self.descriptor_pool, 1, [descriptor_set])
        
        # Cleanup
        vkDestroyBuffer(self.device, buf_hidden, None)
        vkDestroyBuffer(self.device, buf_weights, None)
        vkDestroyBuffer(self.device, buf_bias, None)
        vkDestroyBuffer(self.device, buf_queries, None)
        vkFreeMemory(self.device, mem_hidden, None)
        vkFreeMemory(self.device, mem_weights, None)
        vkFreeMemory(self.device, mem_bias, None)
        vkFreeMemory(self.device, mem_queries, None)
        
        return result
    
    def memory_read(self, queries, memory_keys, memory_values, temperature=None):
        """
        Retrieve memories using attention mechanism
        
        Args:
            queries: Query vectors (batch, key_dim)
            memory_keys: Memory keys (num_memories, key_dim)
            memory_values: Memory values (num_memories, value_dim)
            temperature: Temperature for softmax (default: sqrt(key_dim))
        
        Returns:
            Retrieved values (batch, value_dim)
        """
        q = queries.astype(np.float32)
        keys = memory_keys.astype(np.float32)
        values = memory_values.astype(np.float32)
        
        batch_size, key_dim = q.shape
        num_memories, _ = keys.shape
        _, value_dim = values.shape
        
        if temperature is None:
            temperature = np.sqrt(key_dim)
        
        q_flat = q.flatten()
        keys_flat = keys.flatten()
        values_flat = values.flatten()
        
        # Create buffers
        buf_q, mem_q = self._create_buffer(q_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_keys, mem_keys = self._create_buffer(keys_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_values, mem_values_buf = self._create_buffer(values_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_scores, mem_scores = self._create_buffer(batch_size * num_memories * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_out, mem_out = self._create_buffer(batch_size * value_dim * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self._upload_buffer(buf_q, mem_q, q_flat)
        self._upload_buffer(buf_keys, mem_keys, keys_flat)
        self._upload_buffer(buf_values, mem_values_buf, values_flat)
        
        # Create or get cached pipeline
        if 'memory-read' not in self.pipelines:
            desc_layout = self._create_descriptor_set_layout(5)
            pipe_layout = self._create_pipeline_layout(desc_layout, push_constant_size=24)
            pipeline = self._create_compute_pipeline(
                self.shaders['memory-read'],
                pipe_layout
            )
            
            self.descriptor_set_layouts['memory-read'] = desc_layout
            self.pipeline_layouts['memory-read'] = pipe_layout
            self.pipelines['memory-read'] = pipeline
        
        # Create descriptor set
        descriptor_set = self._create_descriptor_set(
            self.descriptor_set_layouts['memory-read'],
            [
                (buf_q, q_flat.nbytes),
                (buf_keys, keys_flat.nbytes),
                (buf_values, values_flat.nbytes),
                (buf_out, batch_size * value_dim * 4),
                (buf_scores, batch_size * num_memories * 4)
            ]
        )
        
        # Pass 1: Compute attention scores
        push_constants = struct.pack('IIIIfI', batch_size, num_memories, key_dim, value_dim, temperature, 0)
        workgroups = ((batch_size * num_memories) + 255) // 256
        self._dispatch_compute(
            self.pipelines['memory-read'],
            self.pipeline_layouts['memory-read'],
            descriptor_set,
            workgroups,
            push_constants
        )
        
        # Pass 2: Apply softmax (using activation-softmax shader)
        scores = self._download_buffer(mem_scores, batch_size * num_memories * 4, dtype=np.float32)
        scores = scores[:batch_size * num_memories].reshape(batch_size, num_memories)
        scores_softmax = self.activation_softmax(scores, axis=-1)
        
        # Upload softmax scores back
        scores_softmax_flat = scores_softmax.flatten()
        self._upload_buffer(buf_scores, mem_scores, scores_softmax_flat)
        
        # Pass 3: Weighted sum
        push_constants = struct.pack('IIIIfI', batch_size, num_memories, key_dim, value_dim, temperature, 2)
        workgroups = ((batch_size * value_dim) + 255) // 256
        self._dispatch_compute(
            self.pipelines['memory-read'],
            self.pipeline_layouts['memory-read'],
            descriptor_set,
            workgroups,
            push_constants
        )
        
        # Download results
        result = self._download_buffer(mem_out, batch_size * value_dim * 4, dtype=np.float32)
        result = result[:batch_size * value_dim].reshape(batch_size, value_dim)
        
        # Free descriptor set
        vkFreeDescriptorSets(self.device, self.descriptor_pool, 1, [descriptor_set])
        
        # Cleanup
        vkDestroyBuffer(self.device, buf_q, None)
        vkDestroyBuffer(self.device, buf_keys, None)
        vkDestroyBuffer(self.device, buf_values, None)
        vkDestroyBuffer(self.device, buf_scores, None)
        vkDestroyBuffer(self.device, buf_out, None)
        vkFreeMemory(self.device, mem_q, None)
        vkFreeMemory(self.device, mem_keys, None)
        vkFreeMemory(self.device, mem_values_buf, None)
        vkFreeMemory(self.device, mem_scores, None)
        vkFreeMemory(self.device, mem_out, None)
        
        return result
    
    def memory_inject_gate(self, hidden_states, memory_context, gate_weights, gate_bias, memory_proj_weights):
        """
        Inject memory context into hidden states using a gating mechanism
        
        Args:
            hidden_states: Hidden states (batch, seq_len, dim)
            memory_context: Memory context (batch, dim)
            gate_weights: Gate weights (dim, dim * 2) - for [hidden; memory] concatenation
            gate_bias: Gate bias (dim,)
            memory_proj_weights: Memory projection weights (dim, dim)
        
        Returns:
            Augmented hidden states (batch, seq_len, dim)
        """
        hidden = hidden_states.astype(np.float32)
        context = memory_context.astype(np.float32)
        gate_w = gate_weights.astype(np.float32)
        gate_b = gate_bias.astype(np.float32)
        mem_proj_w = memory_proj_weights.astype(np.float32)
        
        batch_size, seq_len, dim = hidden.shape
        
        hidden_flat = hidden.flatten()
        context_flat = context.flatten()
        gate_w_flat = gate_w.flatten()
        gate_b_flat = gate_b.flatten()
        mem_proj_w_flat = mem_proj_w.flatten()
        
        # Create buffers
        buf_hidden, mem_hidden = self._create_buffer(hidden_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_context, mem_context = self._create_buffer(context_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_gate_w, mem_gate_w = self._create_buffer(gate_w_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_gate_b, mem_gate_b = self._create_buffer(gate_b_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_mem_proj_w, mem_mem_proj_w = self._create_buffer(mem_proj_w_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_out, mem_out = self._create_buffer(hidden_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self._upload_buffer(buf_hidden, mem_hidden, hidden_flat)
        self._upload_buffer(buf_context, mem_context, context_flat)
        self._upload_buffer(buf_gate_w, mem_gate_w, gate_w_flat)
        self._upload_buffer(buf_gate_b, mem_gate_b, gate_b_flat)
        self._upload_buffer(buf_mem_proj_w, mem_mem_proj_w, mem_proj_w_flat)
        
        # Create or get cached pipeline
        if 'memory-inject-gate' not in self.pipelines:
            desc_layout = self._create_descriptor_set_layout(6)
            pipe_layout = self._create_pipeline_layout(desc_layout, push_constant_size=12)
            pipeline = self._create_compute_pipeline(
                self.shaders['memory-inject-gate'],
                pipe_layout
            )
            
            self.descriptor_set_layouts['memory-inject-gate'] = desc_layout
            self.pipeline_layouts['memory-inject-gate'] = pipe_layout
            self.pipelines['memory-inject-gate'] = pipeline
        
        # Create descriptor set
        descriptor_set = self._create_descriptor_set(
            self.descriptor_set_layouts['memory-inject-gate'],
            [
                (buf_hidden, hidden_flat.nbytes),
                (buf_context, context_flat.nbytes),
                (buf_gate_w, gate_w_flat.nbytes),
                (buf_gate_b, gate_b_flat.nbytes),
                (buf_mem_proj_w, mem_proj_w_flat.nbytes),
                (buf_out, hidden_flat.nbytes)
            ]
        )
        
        # Pack push constants
        push_constants = struct.pack('III', batch_size, seq_len, dim)
        
        # Dispatch
        workgroups = ((batch_size * seq_len * dim) + 255) // 256
        self._dispatch_compute(
            self.pipelines['memory-inject-gate'],
            self.pipeline_layouts['memory-inject-gate'],
            descriptor_set,
            workgroups,
            push_constants
        )
        
        # Download results
        result = self._download_buffer(mem_out, hidden_flat.nbytes, dtype=np.float32)
        result = result[:len(hidden_flat)].reshape(hidden.shape)
        
        # Free descriptor set
        vkFreeDescriptorSets(self.device, self.descriptor_pool, 1, [descriptor_set])
        
        # Cleanup
        vkDestroyBuffer(self.device, buf_hidden, None)
        vkDestroyBuffer(self.device, buf_context, None)
        vkDestroyBuffer(self.device, buf_gate_w, None)
        vkDestroyBuffer(self.device, buf_gate_b, None)
        vkDestroyBuffer(self.device, buf_mem_proj_w, None)
        vkDestroyBuffer(self.device, buf_out, None)
        vkFreeMemory(self.device, mem_hidden, None)
        vkFreeMemory(self.device, mem_context, None)
        vkFreeMemory(self.device, mem_gate_w, None)
        vkFreeMemory(self.device, mem_gate_b, None)
        vkFreeMemory(self.device, mem_mem_proj_w, None)
        vkFreeMemory(self.device, mem_out, None)
        
        return result
    
    def embedding_lookup(self, token_ids, embedding_table):
        """
        Lookup embeddings from token IDs
        
        Args:
            token_ids: Token IDs (batch, seq_len) as uint32 or (batch * seq_len,) as uint32
            embedding_table: Embedding table (vocab_size, embedding_dim)
        
        Returns:
            Embeddings (batch, seq_len, embedding_dim)
        """
        # Convert token_ids to uint32 if needed
        if token_ids.dtype != np.uint32:
            token_ids = token_ids.astype(np.uint32)
        
        # Handle input shape
        if token_ids.ndim == 2:
            batch_size, seq_len = token_ids.shape
            token_ids_flat = token_ids.flatten()
        else:
            batch_size = 1
            seq_len = len(token_ids)
            token_ids_flat = token_ids.flatten()
        
        vocab_size, embedding_dim = embedding_table.shape
        embedding_table_flat = embedding_table.astype(np.float32).flatten()
        
        # Create buffers
        buf_tokens, mem_tokens = self._create_buffer(token_ids_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_embeddings, mem_embeddings = self._create_buffer(embedding_table_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_output, mem_output = self._create_buffer(batch_size * seq_len * embedding_dim * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self._upload_buffer(buf_tokens, mem_tokens, token_ids_flat)
        self._upload_buffer(buf_embeddings, mem_embeddings, embedding_table_flat)
        
        # Create or get cached pipeline
        if 'embedding-lookup' not in self.pipelines:
            desc_layout = self._create_descriptor_set_layout(3)
            pipe_layout = self._create_pipeline_layout(desc_layout, push_constant_size=16)
            pipeline = self._create_compute_pipeline(
                self.shaders['embedding-lookup'],
                pipe_layout
            )
            
            self.descriptor_set_layouts['embedding-lookup'] = desc_layout
            self.pipeline_layouts['embedding-lookup'] = pipe_layout
            self.pipelines['embedding-lookup'] = pipeline
        
        # Create descriptor set
        descriptor_set = self._create_descriptor_set(
            self.descriptor_set_layouts['embedding-lookup'],
            [
                (buf_tokens, token_ids_flat.nbytes),
                (buf_embeddings, embedding_table_flat.nbytes),
                (buf_output, batch_size * seq_len * embedding_dim * 4)
            ]
        )
        
        # Pack push constants
        push_constants = struct.pack('IIII', batch_size, seq_len, vocab_size, embedding_dim)
        
        # Dispatch
        workgroups = ((batch_size * seq_len) + 255) // 256
        self._dispatch_compute(
            self.pipelines['embedding-lookup'],
            self.pipeline_layouts['embedding-lookup'],
            descriptor_set,
            workgroups,
            push_constants
        )
        
        # Download results
        result = self._download_buffer(mem_output, batch_size * seq_len * embedding_dim * 4, dtype=np.float32)
        result = result[:batch_size * seq_len * embedding_dim].reshape(batch_size, seq_len, embedding_dim)
        
        # Free descriptor set
        vkFreeDescriptorSets(self.device, self.descriptor_pool, 1, [descriptor_set])
        
        # Cleanup
        vkDestroyBuffer(self.device, buf_tokens, None)
        vkDestroyBuffer(self.device, buf_embeddings, None)
        vkDestroyBuffer(self.device, buf_output, None)
        vkFreeMemory(self.device, mem_tokens, None)
        vkFreeMemory(self.device, mem_embeddings, None)
        vkFreeMemory(self.device, mem_output, None)
        
        return result
    
    def memory_write(self, new_key, new_value, memory_keys, memory_values, write_index, write_mode=0, blend_factor=0.5):
        """
        Write key-value pair to memory
        
        Args:
            new_key: New key to write (key_dim,)
            new_value: New value to write (value_dim,)
            memory_keys: Memory keys buffer (num_memories, key_dim) - will be modified
            memory_values: Memory values buffer (num_memories, value_dim) - will be modified
            write_index: Index to write to
            write_mode: 0 = overwrite, 1 = blend
            blend_factor: For blend mode (default: 0.5)
        
        Returns:
            (updated_memory_keys, updated_memory_values)
        """
        key = new_key.astype(np.float32).flatten()
        value = new_value.astype(np.float32).flatten()
        keys = memory_keys.astype(np.float32).flatten()
        values = memory_values.astype(np.float32).flatten()
        
        key_dim = len(key)
        value_dim = len(value)
        num_memories, _ = memory_keys.shape
        
        # Create buffers
        buf_key, mem_key = self._create_buffer(key.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_value, mem_value = self._create_buffer(value.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_keys, mem_keys = self._create_buffer(keys.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_values, mem_values_buf = self._create_buffer(values.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data (read existing memory first)
        self._upload_buffer(buf_key, mem_key, key)
        self._upload_buffer(buf_value, mem_value, value)
        self._upload_buffer(buf_keys, mem_keys, keys)
        self._upload_buffer(buf_values, mem_values_buf, values)
        
        # Create or get cached pipeline
        if 'memory-write' not in self.pipelines:
            desc_layout = self._create_descriptor_set_layout(4)
            pipe_layout = self._create_pipeline_layout(desc_layout, push_constant_size=24)
            pipeline = self._create_compute_pipeline(
                self.shaders['memory-write'],
                pipe_layout
            )
            
            self.descriptor_set_layouts['memory-write'] = desc_layout
            self.pipeline_layouts['memory-write'] = pipe_layout
            self.pipelines['memory-write'] = pipeline
        
        # Create descriptor set
        descriptor_set = self._create_descriptor_set(
            self.descriptor_set_layouts['memory-write'],
            [
                (buf_key, key.nbytes),
                (buf_value, value.nbytes),
                (buf_keys, keys.nbytes),
                (buf_values, values.nbytes)
            ]
        )
        
        # Pack push constants
        push_constants = struct.pack('IIIIIf', num_memories, key_dim, value_dim, write_index, write_mode, blend_factor)
        
        # Dispatch (need enough threads to handle both key and value dimensions)
        max_dim = max(key_dim, value_dim)
        workgroups = (max_dim + 255) // 256
        self._dispatch_compute(
            self.pipelines['memory-write'],
            self.pipeline_layouts['memory-write'],
            descriptor_set,
            workgroups,
            push_constants
        )
        
        # Download updated memory
        updated_keys = self._download_buffer(mem_keys, keys.nbytes, dtype=np.float32)
        updated_keys = updated_keys[:len(keys)].reshape(num_memories, key_dim)
        updated_values = self._download_buffer(mem_values_buf, values.nbytes, dtype=np.float32)
        updated_values = updated_values[:len(values)].reshape(num_memories, value_dim)
        
        # Free descriptor set
        vkFreeDescriptorSets(self.device, self.descriptor_pool, 1, [descriptor_set])
        
        # Cleanup
        vkDestroyBuffer(self.device, buf_key, None)
        vkDestroyBuffer(self.device, buf_value, None)
        vkDestroyBuffer(self.device, buf_keys, None)
        vkDestroyBuffer(self.device, buf_values, None)
        vkFreeMemory(self.device, mem_key, None)
        vkFreeMemory(self.device, mem_value, None)
        vkFreeMemory(self.device, mem_keys, None)
        vkFreeMemory(self.device, mem_values_buf, None)
        
        return updated_keys, updated_values
    
    def dropout(self, input_data, dropout_prob=0.1, is_training=True, seed=None):
        """
        Apply dropout regularization
        
        Args:
            input_data: Input tensor
            dropout_prob: Probability of dropping (default: 0.1)
            is_training: Whether in training mode (default: True)
            seed: Random seed for reproducibility (optional)
        
        Returns:
            Output tensor with dropout applied
        """
        data = input_data.astype(np.float32)
        original_shape = data.shape
        data_flat = data.flatten()
        total_elements = len(data_flat)
        
        # Generate random mask (on CPU for now)
        if seed is not None:
            np.random.seed(seed)
        random_mask = np.random.rand(total_elements).astype(np.float32)
        
        # Create buffers
        buf_input, mem_input = self._create_buffer(data_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_random, mem_random = self._create_buffer(random_mask.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_output, mem_output = self._create_buffer(data_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self._upload_buffer(buf_input, mem_input, data_flat)
        self._upload_buffer(buf_random, mem_random, random_mask)
        
        # Create or get cached pipeline
        if 'fnn-dropout' not in self.pipelines:
            desc_layout = self._create_descriptor_set_layout(3)
            pipe_layout = self._create_pipeline_layout(desc_layout, push_constant_size=16)
            pipeline = self._create_compute_pipeline(
                self.shaders['fnn-dropout'],
                pipe_layout
            )
            
            self.descriptor_set_layouts['fnn-dropout'] = desc_layout
            self.pipeline_layouts['fnn-dropout'] = pipe_layout
            self.pipelines['fnn-dropout'] = pipeline
        
        # Create descriptor set
        descriptor_set = self._create_descriptor_set(
            self.descriptor_set_layouts['fnn-dropout'],
            [
                (buf_input, data_flat.nbytes),
                (buf_random, random_mask.nbytes),
                (buf_output, data_flat.nbytes)
            ]
        )
        
        # Pack push constants
        is_train = 1 if is_training else 0
        push_constants = struct.pack('IfI', total_elements, dropout_prob, is_train)
        
        # Dispatch
        workgroups = (total_elements + 255) // 256
        self._dispatch_compute(
            self.pipelines['fnn-dropout'],
            self.pipeline_layouts['fnn-dropout'],
            descriptor_set,
            workgroups,
            push_constants
        )
        
        # Download results
        result = self._download_buffer(mem_output, data_flat.nbytes, dtype=np.float32)
        result = result[:total_elements].reshape(original_shape)
        
        # Free descriptor set
        vkFreeDescriptorSets(self.device, self.descriptor_pool, 1, [descriptor_set])
        
        # Cleanup
        vkDestroyBuffer(self.device, buf_input, None)
        vkDestroyBuffer(self.device, buf_random, None)
        vkDestroyBuffer(self.device, buf_output, None)
        vkFreeMemory(self.device, mem_input, None)
        vkFreeMemory(self.device, mem_random, None)
        vkFreeMemory(self.device, mem_output, None)
        
        return result
    
    def place_cell(self, agent_position, field_centers, field_width=1.0, max_rate=20.0, baseline_rate=0.1, spatial_dims=2):
        """
        Generate place cell firing rates based on agent position
        
        Args:
            agent_position: Agent position (spatial_dims,) - single position
            field_centers: Place field centers (n_neurons, spatial_dims)
            field_width: Width of place fields (default: 1.0)
            max_rate: Maximum firing rate in Hz (default: 20.0)
            baseline_rate: Baseline firing rate in Hz (default: 0.1)
            spatial_dims: 2 for 2D, 3 for 3D (default: 2)
        
        Returns:
            Firing rates (n_neurons,)
        """
        pos = agent_position.astype(np.float32).flatten()
        centers = field_centers.astype(np.float32)
        
        if len(pos) != spatial_dims:
            raise ValueError(f"agent_position must have {spatial_dims} elements for {spatial_dims}D space")
        
        n_neurons, _ = centers.shape
        centers_flat = centers.flatten()
        
        # Create buffers
        buf_pos, mem_pos = self._create_buffer(pos.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_centers, mem_centers = self._create_buffer(centers_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_rates, mem_rates = self._create_buffer(n_neurons * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self._upload_buffer(buf_pos, mem_pos, pos)
        self._upload_buffer(buf_centers, mem_centers, centers_flat)
        
        # Create or get cached pipeline
        if 'place-cell' not in self.pipelines:
            desc_layout = self._create_descriptor_set_layout(3)
            pipe_layout = self._create_pipeline_layout(desc_layout, push_constant_size=24)
            pipeline = self._create_compute_pipeline(
                self.shaders['place-cell'],
                pipe_layout
            )
            
            self.descriptor_set_layouts['place-cell'] = desc_layout
            self.pipeline_layouts['place-cell'] = pipe_layout
            self.pipelines['place-cell'] = pipeline
        
        # Create descriptor set
        descriptor_set = self._create_descriptor_set(
            self.descriptor_set_layouts['place-cell'],
            [
                (buf_pos, pos.nbytes),
                (buf_centers, centers_flat.nbytes),
                (buf_rates, n_neurons * 4)
            ]
        )
        
        # Pack push constants
        push_constants = struct.pack('IIfff', n_neurons, spatial_dims, field_width, max_rate, baseline_rate)
        
        # Dispatch
        workgroups = (n_neurons + 255) // 256
        self._dispatch_compute(
            self.pipelines['place-cell'],
            self.pipeline_layouts['place-cell'],
            descriptor_set,
            workgroups,
            push_constants
        )
        
        # Download results
        result = self._download_buffer(mem_rates, n_neurons * 4, dtype=np.float32)
        result = result[:n_neurons]
        
        # Free descriptor set
        vkFreeDescriptorSets(self.device, self.descriptor_pool, 1, [descriptor_set])
        
        # Cleanup
        vkDestroyBuffer(self.device, buf_pos, None)
        vkDestroyBuffer(self.device, buf_centers, None)
        vkDestroyBuffer(self.device, buf_rates, None)
        vkFreeMemory(self.device, mem_pos, None)
        vkFreeMemory(self.device, mem_centers, None)
        vkFreeMemory(self.device, mem_rates, None)
        
        return result
    
    def time_cell(self, current_time, preferred_times, time_constant=1.0, max_rate=20.0, baseline_rate=0.1):
        """
        Generate time cell firing rates based on elapsed time
        
        Args:
            current_time: Current normalized time (0-1)
            preferred_times: Preferred firing times for each cell (n_neurons,)
            time_constant: Time field width (default: 1.0)
            max_rate: Maximum firing rate in Hz (default: 20.0)
            baseline_rate: Baseline firing rate in Hz (default: 0.1)
        
        Returns:
            Firing rates (n_neurons,)
        """
        time_arr = np.array([current_time], dtype=np.float32)
        prefs = preferred_times.astype(np.float32).flatten()
        n_neurons = len(prefs)
        
        # Create buffers
        buf_time, mem_time = self._create_buffer(time_arr.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_prefs, mem_prefs = self._create_buffer(prefs.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_rates, mem_rates = self._create_buffer(n_neurons * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_mem, mem_mem = self._create_buffer(n_neurons * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Initialize memory buffer
        self._upload_buffer(buf_mem, mem_mem, np.zeros(n_neurons, dtype=np.float32))
        
        # Upload data
        self._upload_buffer(buf_time, mem_time, time_arr)
        self._upload_buffer(buf_prefs, mem_prefs, prefs)
        
        # Create or get cached pipeline
        if 'time-cell' not in self.pipelines:
            desc_layout = self._create_descriptor_set_layout(4)
            pipe_layout = self._create_pipeline_layout(desc_layout, push_constant_size=20)
            pipeline = self._create_compute_pipeline(
                self.shaders['time-cell'],
                pipe_layout
            )
            
            self.descriptor_set_layouts['time-cell'] = desc_layout
            self.pipeline_layouts['time-cell'] = pipe_layout
            self.pipelines['time-cell'] = pipeline
        
        # Create descriptor set
        descriptor_set = self._create_descriptor_set(
            self.descriptor_set_layouts['time-cell'],
            [
                (buf_time, time_arr.nbytes),
                (buf_prefs, prefs.nbytes),
                (buf_rates, n_neurons * 4),
                (buf_mem, n_neurons * 4)
            ]
        )
        
        # Pack push constants
        push_constants = struct.pack('Ifff', n_neurons, time_constant, max_rate, baseline_rate)
        
        # Dispatch
        workgroups = (n_neurons + 255) // 256
        self._dispatch_compute(
            self.pipelines['time-cell'],
            self.pipeline_layouts['time-cell'],
            descriptor_set,
            workgroups,
            push_constants
        )
        
        # Download results
        result = self._download_buffer(mem_rates, n_neurons * 4, dtype=np.float32)
        result = result[:n_neurons]
        
        # Free descriptor set
        vkFreeDescriptorSets(self.device, self.descriptor_pool, 1, [descriptor_set])
        
        # Cleanup
        vkDestroyBuffer(self.device, buf_time, None)
        vkDestroyBuffer(self.device, buf_prefs, None)
        vkDestroyBuffer(self.device, buf_rates, None)
        vkDestroyBuffer(self.device, buf_mem, None)
        vkFreeMemory(self.device, mem_time, None)
        vkFreeMemory(self.device, mem_prefs, None)
        vkFreeMemory(self.device, mem_rates, None)
        vkFreeMemory(self.device, mem_mem, None)
        
        return result
    
    def __del__(self):
        """Cleanup Vulkan resources"""
        if hasattr(self, 'device') and self.device:
            # Destroy pipelines
            for pipeline in self.pipelines.values():
                vkDestroyPipeline(self.device, pipeline, None)
            
            # Destroy pipeline layouts
            for layout in self.pipeline_layouts.values():
                vkDestroyPipelineLayout(self.device, layout, None)
            
            # Destroy descriptor set layouts
            for layout in self.descriptor_set_layouts.values():
                vkDestroyDescriptorSetLayout(self.device, layout, None)
            
            # Destroy descriptor pool
            if hasattr(self, 'descriptor_pool'):
                vkDestroyDescriptorPool(self.device, self.descriptor_pool, None)
            
            # Destroy command pool
            if hasattr(self, 'command_pool'):
                vkDestroyCommandPool(self.device, self.command_pool, None)
            
            # Destroy device
            vkDestroyDevice(self.device, None)
        
        # Destroy instance
        if hasattr(self, 'instance') and self.instance:
            vkDestroyInstance(self.instance, None)

    def faiss_compute_distances(self, queries, database, distance_type='cosine'):
        """
        Compute pairwise distances between query and database vectors.
        
        Args:
            queries: Query vectors (num_queries, dim) or (dim,)
            database: Database vectors (num_database, dim)
            distance_type: Distance metric - 'l2', 'cosine', or 'dot'
            
        Returns:
            Distance matrix (num_queries, num_database)
        """
        queries = queries.astype(np.float32)
        database = database.astype(np.float32)
        
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        
        num_queries, dim = queries.shape
        num_database = database.shape[0]
        
        distance_map = {'l2': 0, 'cosine': 1, 'dot': 2}
        dist_type_int = distance_map.get(distance_type, 1)
        
        queries_flat = queries.flatten()
        database_flat = database.flatten()
        
        buf_queries, mem_queries = self._create_buffer(
            queries_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        buf_database, mem_database = self._create_buffer(
            database_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        buf_distances, mem_distances = self._create_buffer(
            num_queries * num_database * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        
        self._upload_buffer(buf_queries, mem_queries, queries_flat)
        self._upload_buffer(buf_database, mem_database, database_flat)
        
        if 'faiss-distance' not in self.pipelines:
            desc_layout = self._create_descriptor_set_layout(3)
            pipe_layout = self._create_pipeline_layout(desc_layout, push_constant_size=16)
            pipeline = self._create_compute_pipeline(
                self.shaders['faiss-distance'],
                pipe_layout
            )
            
            self.descriptor_set_layouts['faiss-distance'] = desc_layout
            self.pipeline_layouts['faiss-distance'] = pipe_layout
            self.pipelines['faiss-distance'] = pipeline
        
        descriptor_set = self._create_descriptor_set(
            self.descriptor_set_layouts['faiss-distance'],
            [
                (buf_queries, queries_flat.nbytes),
                (buf_database, database_flat.nbytes),
                (buf_distances, num_queries * num_database * 4)
            ]
        )
        
        push_constants = struct.pack('IIII', num_queries, num_database, dim, dist_type_int)
        
        workgroups_x = (num_database + 15) // 16
        workgroups_y = (num_queries + 15) // 16
        self._dispatch_compute(
            self.pipelines['faiss-distance'],
            self.pipeline_layouts['faiss-distance'],
            descriptor_set,
            workgroups_x,
            push_constants,
            workgroups_y,
            1
        )
        
        distances = self._download_buffer(
            mem_distances, num_queries * num_database * 4, dtype=np.float32
        )
        distances = distances[:num_queries * num_database].reshape(num_queries, num_database)
        
        vkFreeDescriptorSets(self.device, self.descriptor_pool, 1, [descriptor_set])
        
        vkDestroyBuffer(self.device, buf_queries, None)
        vkDestroyBuffer(self.device, buf_database, None)
        vkDestroyBuffer(self.device, buf_distances, None)
        vkFreeMemory(self.device, mem_queries, None)
        vkFreeMemory(self.device, mem_database, None)
        vkFreeMemory(self.device, mem_distances, None)
        
        return distances

    def faiss_topk(self, distances, k):
        """
        Select top-k smallest distances for each query.
        
        Args:
            distances: Distance matrix (num_queries, num_database)
            k: Number of nearest neighbors to return
            
        Returns:
            Tuple of (indices, distances) arrays, each (num_queries, k)
        """
        distances = distances.astype(np.float32)
        num_queries, num_database = distances.shape
        
        k = min(k, num_database)
        
        distances_flat = distances.flatten()
        db_indices = np.arange(num_database, dtype=np.uint32)
        
        buf_distances, mem_distances = self._create_buffer(
            distances_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        buf_db_indices, mem_db_indices = self._create_buffer(
            db_indices.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        buf_topk_indices, mem_topk_indices = self._create_buffer(
            num_queries * k * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        buf_topk_distances, mem_topk_distances = self._create_buffer(
            num_queries * k * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        
        self._upload_buffer(buf_distances, mem_distances, distances_flat)
        self._upload_buffer(buf_db_indices, mem_db_indices, db_indices)
        
        if 'faiss-topk' not in self.pipelines:
            desc_layout = self._create_descriptor_set_layout(4)
            pipe_layout = self._create_pipeline_layout(desc_layout, push_constant_size=12)
            pipeline = self._create_compute_pipeline(
                self.shaders['faiss-topk'],
                pipe_layout
            )
            
            self.descriptor_set_layouts['faiss-topk'] = desc_layout
            self.pipeline_layouts['faiss-topk'] = pipe_layout
            self.pipelines['faiss-topk'] = pipeline
        
        descriptor_set = self._create_descriptor_set(
            self.descriptor_set_layouts['faiss-topk'],
            [
                (buf_distances, distances_flat.nbytes),
                (buf_db_indices, db_indices.nbytes),
                (buf_topk_indices, num_queries * k * 4),
                (buf_topk_distances, num_queries * k * 4)
            ]
        )
        
        push_constants = struct.pack('III', num_queries, num_database, k)
        
        workgroups = (num_queries + 255) // 256
        self._dispatch_compute(
            self.pipelines['faiss-topk'],
            self.pipeline_layouts['faiss-topk'],
            descriptor_set,
            workgroups,
            push_constants
        )
        
        topk_indices = self._download_buffer(
            mem_topk_indices, num_queries * k * 4, dtype=np.uint32
        )
        topk_distances = self._download_buffer(
            mem_topk_distances, num_queries * k * 4, dtype=np.float32
        )
        
        topk_indices = topk_indices[:num_queries * k].reshape(num_queries, k)
        topk_distances = topk_distances[:num_queries * k].reshape(num_queries, k)
        
        vkFreeDescriptorSets(self.device, self.descriptor_pool, 1, [descriptor_set])
        
        vkDestroyBuffer(self.device, buf_distances, None)
        vkDestroyBuffer(self.device, buf_db_indices, None)
        vkDestroyBuffer(self.device, buf_topk_indices, None)
        vkDestroyBuffer(self.device, buf_topk_distances, None)
        vkFreeMemory(self.device, mem_distances, None)
        vkFreeMemory(self.device, mem_db_indices, None)
        vkFreeMemory(self.device, mem_topk_indices, None)
        vkFreeMemory(self.device, mem_topk_distances, None)
        
        return topk_indices, topk_distances


# Import config for SNN parameters
try:
    from config import SNNConfig, LogConfig
    _config_available = True
except ImportError:
    _config_available = False

import logging
logging.basicConfig(level="INFO")
_snn_logger = logging.getLogger(__name__)


class SNNCompute:
    """
    High-level SNN interface for spike computation
    
    Provides a clean API for processing embeddings through a spiking neural network.
    Uses Vulkan GPU acceleration when available, with CPU fallback.
    """
    
    # Default parameters (overridden by config if available)
    DEFAULT_N_NEURONS = 1000
    DEFAULT_DT = 0.01
    DEFAULT_TAU_MEM = 5.0
    DEFAULT_V_THRESH = 0.5
    DEFAULT_TIMESTEPS = 50
    DEFAULT_INPUT_SCALE = 20.0
    
    def __init__(self, n_neurons: int = None, use_vulkan: bool = True):
        """
        Initialize SNN compute engine
        
        Args:
            n_neurons: Number of neurons (default from config or 1000)
            use_vulkan: Whether to use GPU acceleration
        """
        # Get parameters from config if available
        if _config_available:
            self.n_neurons = n_neurons or SNNConfig.N_NEURONS
            self.dt = SNNConfig.DT
            self.tau_mem = SNNConfig.TAU_MEM
            self.v_thresh = SNNConfig.V_THRESH
            self.timesteps = SNNConfig.TIMESTEPS
            self.input_scale = SNNConfig.INPUT_SCALE
        else:
            self.n_neurons = n_neurons or self.DEFAULT_N_NEURONS
            self.dt = self.DEFAULT_DT
            self.tau_mem = self.DEFAULT_TAU_MEM
            self.v_thresh = self.DEFAULT_V_THRESH
            self.timesteps = self.DEFAULT_TIMESTEPS
            self.input_scale = self.DEFAULT_INPUT_SCALE
        
        # Initialize GPU backend
        self.use_vulkan = False
        self.backend = None
        
        if use_vulkan and VULKAN_AVAILABLE:
            try:
                self.backend = VulkanCompute()
                self.use_vulkan = True
                _snn_logger.info("[OK] GPU compute enabled")
            except Exception as e:
                _snn_logger.warning(f"[!] GPU init failed: {e}, using CPU fallback")
        
        # Neuron state
        self.membrane = np.zeros(self.n_neurons, dtype=np.float32)
        self.refractory = np.zeros(self.n_neurons, dtype=np.float32)
    
    def reset(self) -> None:
        """Reset neuron state (membrane potential and refractory periods)"""
        self.membrane.fill(0)
        self.refractory.fill(0)
    
    def forward(self, input_current: np.ndarray) -> np.ndarray:
        """
        Run one timestep of LIF dynamics
        
        Args:
            input_current: Input current for each neuron (n_neurons,)
            
        Returns:
            Spike output array (n_neurons,) - 1.0 where spike occurred, 0.0 otherwise
        """
        if self.use_vulkan and self.backend is not None:
            self.membrane, self.refractory, spikes = self.backend.lif_step(
                input_current, self.membrane, self.refractory,
                dt=self.dt,
                tau_mem=self.tau_mem,
                v_thresh=self.v_thresh
            )
            return spikes
        else:
            return self._cpu_forward(input_current)
    
    def _cpu_forward(self, input_current: np.ndarray) -> np.ndarray:
        """CPU fallback for LIF dynamics"""
        decay = np.exp(-self.dt / self.tau_mem)
        self.membrane = self.membrane * decay + input_current * self.dt
        spikes = (self.membrane >= self.v_thresh).astype(np.float32)
        self.membrane = self.membrane * (1 - spikes)  # Reset after spike
        return spikes
    
    def process(self, embedding: np.ndarray) -> dict:
        """
        Process embedding through SNN pipeline
        
        Converts embedding to input current, runs through LIF neurons for
        multiple timesteps, and returns spike metrics for visualization.
        
        Args:
            embedding: Input embedding vector (any length, will be padded/truncated)
        
        Returns:
            Dictionary with:
                - 'spike_activity': Total spike count across all timesteps
                - 'spikes': Binary spike pattern (n_neurons,)
                - 'firing_rate': Average firing rate (spikes per neuron per timestep)
        """
        # Prepare input current
        input_current = self._prepare_input(embedding)
        
        # Run simulation
        total_spikes, spike_pattern = self._run_simulation(input_current)
        
        # Compute firing rate
        firing_rate = total_spikes / (self.n_neurons * self.timesteps)
        
        return {
            'spike_activity': total_spikes,
            'spikes': spike_pattern,
            'firing_rate': firing_rate
        }
    
    def _prepare_input(self, embedding: np.ndarray) -> np.ndarray:
        """Prepare embedding as input current for neurons"""
        # Pad or truncate to n_neurons
        input_current = embedding.astype(np.float32)[:self.n_neurons]
        if len(input_current) < self.n_neurons:
            input_current = np.pad(input_current, (0, self.n_neurons - len(input_current)))
        
        # Scale to appropriate range for LIF neurons
        # Use absolute value and scale for consistent activity
        return np.abs(input_current) * self.input_scale
    
    def _run_simulation(self, input_current: np.ndarray) -> tuple:
        """
        Run LIF simulation for multiple timesteps
        
        Returns:
            Tuple of (total_spike_count, spike_pattern)
        """
        # Reset membrane for fresh simulation
        self.reset()
        
        # Scale down for accumulation (LIF integrates over time)
        scaled_input = input_current / 4.0
        
        total_spikes = 0.0
        spike_pattern = np.zeros(self.n_neurons, dtype=np.float32)
        
        for _ in range(self.timesteps):
            spikes = self.forward(scaled_input)
            total_spikes += float(spikes.sum())
            spike_pattern = np.maximum(spike_pattern, spikes)
        
        # If no spikes from LIF, use threshold-based fallback for visualization
        if total_spikes == 0:
            threshold_spikes = (input_current >= self.v_thresh).astype(np.float32)
            total_spikes = float(threshold_spikes.sum())
            spike_pattern = threshold_spikes
        
        return total_spikes, spike_pattern