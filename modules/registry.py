"""
Plugin registry for GrillCheese plugin system

Manages all loaded plugins and provides access to active plugins.
Uses singleton pattern to ensure single registry instance.
"""
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from config import ModuleConfig
from .base import (
    BaseMemoryBackend,
    BaseModelProvider,
    BaseProcessingHook,
    BaseAPIExtension,
    BaseTool
)
from .loader import load_all_modules

logger = logging.getLogger(__name__)


class ModuleRegistry:
    """
    Central registry for all plugins.
    
    Singleton pattern ensures single registry instance across the application.
    """
    _instance: Optional['ModuleRegistry'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize registry (only called once due to singleton)."""
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        
        # Plugin storage
        self.memory_backends: Dict[str, BaseMemoryBackend] = {}
        self.model_providers: Dict[str, BaseModelProvider] = {}
        self.processing_hooks: List[BaseProcessingHook] = []
        self.api_extensions: List[BaseAPIExtension] = []
        self.tools: Dict[str, BaseTool] = {}
        
        # Active plugins
        self._active_memory_backend: Optional[str] = None
        self._active_model_provider: Optional[str] = None
        
        # Configuration
        self.config: Dict[str, Any] = {}
    
    async def load_all_modules(
        self,
        modules_dir: Optional[Path] = None,
        config_path: Optional[Path] = None
    ) -> None:
        """
        Load all modules from the modules directory.
        
        Args:
            modules_dir: Directory to scan (defaults to ModuleConfig.MODULES_DIR)
            config_path: Path to config file (defaults to ModuleConfig.MODULES_CONFIG_FILE)
        """
        # Load modules
        loaded_modules = load_all_modules(modules_dir, config_path)
        
        # Load configuration
        from .loader import load_config
        self.config = load_config(config_path)
        
        # Register memory backends
        for module_info in loaded_modules['memory_backends']:
            try:
                instance = module_info['class']()
                class_name = module_info['name']
                name = class_name.lower()
                self.register_memory_backend(name, instance)
                
                # Register common aliases for easier matching
                if 'sqlitefaiss' in name or 'sqlite_faiss' in name:
                    self.register_memory_backend('sqlite_faiss', instance)
                elif 'faiss' in name:
                    self.register_memory_backend('faiss', instance)
                
                logger.info(f"Registered memory backend: {name}")
            except Exception as e:
                logger.error(f"Failed to register memory backend {module_info['name']}: {e}")
        
        # Register model providers
        for module_info in loaded_modules['model_providers']:
            try:
                instance = module_info['class']()
                class_name = module_info['name']
                name = class_name.lower()
                self.register_model_provider(name, instance)
                
                # Register common aliases for easier matching
                if 'gguf' in name:
                    self.register_model_provider('gguf', instance)
                elif 'pytorch' in name or 'torch' in name:
                    self.register_model_provider('pytorch', instance)
                    self.register_model_provider('torch', instance)
                
                logger.info(f"Registered model provider: {name}")
            except Exception as e:
                logger.error(f"Failed to register model provider {module_info['name']}: {e}")
        
        # Register processing hooks
        for module_info in loaded_modules['processing_hooks']:
            try:
                instance = module_info['class']()
                self.register_processing_hook(instance)
                logger.info(f"Registered processing hook: {module_info['name']}")
            except Exception as e:
                logger.error(f"Failed to register processing hook {module_info['name']}: {e}")
        
        # Register API extensions
        for module_info in loaded_modules['api_extensions']:
            try:
                instance = module_info['class']()
                self.register_api_extension(instance)
                logger.info(f"Registered API extension: {module_info['name']}")
            except Exception as e:
                logger.error(f"Failed to register API extension {module_info['name']}: {e}")
        
        # Register tools
        for module_info in loaded_modules['tools']:
            try:
                instance = module_info['class']()
                self.register_tool(instance)
                logger.info(f"Registered tool: {instance.name}")
            except Exception as e:
                logger.error(f"Failed to register tool {module_info['name']}: {e}")
        
        # Set active plugins from config
        defaults = self.config.get('defaults', {})
        if defaults.get('memory_backend'):
            # Try exact match first, then flexible matching
            backend_name = defaults['memory_backend'].lower()
            if backend_name in self.memory_backends:
                self.set_active_memory_backend(backend_name)
            else:
                # Try flexible matching (e.g., "sqlite_faiss" matches "sqlitefaissbackend")
                backend_name_normalized = backend_name.replace('_', '').replace('-', '')
                for registered_name in self.memory_backends.keys():
                    if backend_name_normalized in registered_name or registered_name in backend_name_normalized:
                        self.set_active_memory_backend(registered_name)
                        break
        
        if defaults.get('model_provider'):
            # Try exact match first, then flexible matching
            provider_name = defaults['model_provider'].lower()
            if provider_name in self.model_providers:
                self.set_active_model_provider(provider_name)
            else:
                # Try flexible matching (e.g., "gguf" matches "ggufmodelprovider")
                provider_name_normalized = provider_name.replace('_', '').replace('-', '')
                for registered_name in self.model_providers.keys():
                    if provider_name_normalized in registered_name or registered_name in provider_name_normalized:
                        self.set_active_model_provider(registered_name)
                        break
    
    def register_memory_backend(self, name: str, backend: BaseMemoryBackend) -> None:
        """
        Register a memory backend.
        
        Args:
            name: Backend name (lowercase)
            backend: Backend instance
        """
        self.memory_backends[name.lower()] = backend
        logger.debug(f"Registered memory backend: {name}")
    
    def register_model_provider(self, name: str, provider: BaseModelProvider) -> None:
        """
        Register a model provider.
        
        Args:
            name: Provider name (lowercase)
            provider: Provider instance
        """
        self.model_providers[name.lower()] = provider
        logger.debug(f"Registered model provider: {name}")
    
    def register_processing_hook(self, hook: BaseProcessingHook) -> None:
        """
        Register a processing hook (appended to chain).
        
        Args:
            hook: Hook instance
        """
        self.processing_hooks.append(hook)
        logger.debug(f"Registered processing hook: {type(hook).__name__}")
    
    def register_api_extension(self, extension: BaseAPIExtension) -> None:
        """
        Register an API extension.
        
        Args:
            extension: Extension instance
        """
        self.api_extensions.append(extension)
        logger.debug(f"Registered API extension: {type(extension).__name__}")
    
    def register_tool(self, tool: BaseTool) -> None:
        """
        Register an AI tool.
        
        Args:
            tool: Tool instance
        """
        self.tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")
    
    def get_tools(self) -> List[BaseTool]:
        """
        Get all registered tools.
        
        Returns:
            List of tool instances
        """
        return list(self.tools.values())
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        Get a specific tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool instance or None if not found
        """
        return self.tools.get(name)
    
    def set_active_memory_backend(self, name: str) -> bool:
        """
        Set the active memory backend.
        
        Args:
            name: Backend name
            
        Returns:
            True if successful, False if backend not found
        """
        name_lower = name.lower()
        if name_lower in self.memory_backends:
            self._active_memory_backend = name_lower
            logger.info(f"Active memory backend set to: {name}")
            return True
        else:
            logger.warning(f"Memory backend not found: {name}")
            return False
    
    def set_active_model_provider(self, name: str) -> bool:
        """
        Set the active model provider.
        
        Args:
            name: Provider name
            
        Returns:
            True if successful, False if provider not found
        """
        name_lower = name.lower()
        if name_lower in self.model_providers:
            self._active_model_provider = name_lower
            logger.info(f"Active model provider set to: {name}")
            return True
        else:
            logger.warning(f"Model provider not found: {name}")
            return False
    
    def get_active_memory_backend(self) -> Optional[BaseMemoryBackend]:
        """
        Get the active memory backend.
        
        Returns:
            Active backend instance or None if not set
        """
        if self._active_memory_backend:
            return self.memory_backends.get(self._active_memory_backend)
        return None
    
    def get_active_model_provider(self) -> Optional[BaseModelProvider]:
        """
        Get the active model provider.
        
        Returns:
            Active provider instance or None if not set
        """
        if self._active_model_provider:
            return self.model_providers.get(self._active_model_provider)
        return None

