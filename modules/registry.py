"""
Plugin registry - Central singleton for managing loaded plugins
"""
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from modules.base import (
    BaseMemoryBackend,
    BaseModelProvider,
    BaseProcessingHook,
    BaseAPIExtension,
    BaseTool
)
from modules.loader import load_all_modules
from config import BASE_DIR

logger = logging.getLogger(__name__)


class ModuleRegistry:
    """
    Singleton registry for managing all plugins
    
    Provides access to:
    - Memory backends
    - Model providers
    - Processing hooks
    - API extensions
    - Tools
    """
    
    _instance: Optional['ModuleRegistry'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.memory_backends: Dict[str, BaseMemoryBackend] = {}
        self.model_providers: Dict[str, BaseModelProvider] = {}
        self.processing_hooks: List[BaseProcessingHook] = []
        self.api_extensions: List[BaseAPIExtension] = []
        self.tools: Dict[str, BaseTool] = {}
        
        # Active plugins (set by configuration)
        self._active_memory_backend: Optional[str] = None
        self._active_model_provider: Optional[str] = None
        
        self._initialized = True
    
    def register_memory_backend(
        self,
        name: str,
        backend: BaseMemoryBackend
    ) -> None:
        """Register a memory backend"""
        self.memory_backends[name] = backend
        logger.debug(f"Registered memory backend: {name}")
    
    def register_model_provider(
        self,
        name: str,
        provider: BaseModelProvider
    ) -> None:
        """Register a model provider"""
        self.model_providers[name] = provider
        logger.debug(f"Registered model provider: {name}")
    
    def register_processing_hook(
        self,
        hook: BaseProcessingHook
    ) -> None:
        """Register a processing hook (appends to chain)"""
        self.processing_hooks.append(hook)
        # Sort by priority
        self.processing_hooks.sort(key=lambda h: h.priority)
        logger.debug(f"Registered processing hook: {hook.name}")
    
    def register_api_extension(
        self,
        extension: BaseAPIExtension
    ) -> None:
        """Register an API extension"""
        self.api_extensions.append(extension)
        logger.debug(f"Registered API extension: {extension.name}")
    
    def register_tool(
        self,
        tool: BaseTool
    ) -> None:
        """Register an AI tool"""
        if tool.name in self.tools:
            logger.warning(f"Tool '{tool.name}' already registered, overwriting")
        
        # Pass registry to tools that need it (e.g., MemoryQueryTool)
        if hasattr(tool, '__class__') and tool.__class__.__name__ == 'MemoryQueryTool':
            if hasattr(tool, '__init__'):
                import inspect
                sig = inspect.signature(tool.__init__)
                if 'registry' in sig.parameters:
                    tool.registry = self
        
        self.tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")
    
    def get_tools(self) -> List[BaseTool]:
        """Get all registered tools"""
        return list(self.tools.values())
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a specific tool by name"""
        return self.tools.get(name)
    
    def get_active_memory_backend(self) -> Optional[BaseMemoryBackend]:
        """
        Get the active memory backend (configured default)
        
        Returns:
            Active memory backend instance or None
        """
        if self._active_memory_backend:
            return self.memory_backends.get(self._active_memory_backend)
        
        # Try to find by name match
        if self.memory_backends:
            # Return first available if no default set
            return list(self.memory_backends.values())[0]
        
        return None
    
    def get_active_model_provider(self) -> Optional[BaseModelProvider]:
        """
        Get the active model provider (configured default)
        
        Returns:
            Active model provider instance or None
        """
        if self._active_model_provider:
            return self.model_providers.get(self._active_model_provider)
        
        # Try to find by name match
        if self.model_providers:
            # Return first available if no default set
            return list(self.model_providers.values())[0]
        
        return None
    
    def set_active_memory_backend(self, name: str) -> bool:
        """
        Set the active memory backend
        
        Args:
            name: Backend name
            
        Returns:
            True if set successfully, False if backend not found
        """
        if name in self.memory_backends:
            self._active_memory_backend = name
            logger.info(f"Set active memory backend: {name}")
            return True
        
        # Try case-insensitive match
        for backend_name in self.memory_backends.keys():
            if backend_name.lower() == name.lower():
                self._active_memory_backend = backend_name
                logger.info(f"Set active memory backend: {backend_name}")
                return True
        
        logger.warning(f"Memory backend '{name}' not found")
        return False
    
    def set_active_model_provider(self, name: str) -> bool:
        """
        Set the active model provider
        
        Args:
            name: Provider name
            
        Returns:
            True if set successfully, False if provider not found
        """
        if name in self.model_providers:
            self._active_model_provider = name
            logger.info(f"Set active model provider: {name}")
            return True
        
        # Try case-insensitive match
        for provider_name in self.model_providers.keys():
            if provider_name.lower() == name.lower():
                self._active_model_provider = provider_name
                logger.info(f"Set active model provider: {provider_name}")
                return True
        
        logger.warning(f"Model provider '{name}' not found")
        return False
    
    def load_all_modules(
        self,
        modules_dir: Optional[Path] = None,
        config_path: Optional[Path] = None
    ) -> None:
        """
        Discover and load all modules
        
        Args:
            modules_dir: Directory to scan (default: BASE_DIR / "modules")
            config_path: Path to modules_config.json (default: modules_dir / "modules_config.json")
        """
        if modules_dir is None:
            modules_dir = BASE_DIR / "modules"
        
        if config_path is None:
            config_path = modules_dir / "modules_config.json"
        
        # Load all modules
        loaded = load_all_modules(modules_dir, config_path)
        
        # Register memory backends
        for backend in loaded.get("memory_backends", []):
            name = backend.__class__.__name__
            self.register_memory_backend(name, backend)
        
        # Register model providers
        for provider in loaded.get("model_providers", []):
            name = provider.__class__.__name__
            self.register_model_provider(name, provider)
        
        # Register processing hooks
        for hook in loaded.get("processing_hooks", []):
            self.register_processing_hook(hook)
        
        # Register API extensions
        for extension in loaded.get("api_extensions", []):
            self.register_api_extension(extension)
        
        # Register tools - pass registry to tools that need it (e.g., MemoryQueryTool)
        for tool in loaded.get("tools", []):
            # If tool needs registry, pass it
            if hasattr(tool, '__class__') and tool.__class__.__name__ == 'MemoryQueryTool':
                if hasattr(tool, '__init__'):
                    import inspect
                    sig = inspect.signature(tool.__init__)
                    if 'registry' in sig.parameters:
                        tool.registry = self
            self.register_tool(tool)
        
        # Set active plugins from config
        if config_path.exists():
            import json
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                defaults = config.get("defaults", {})
                
                # Set active memory backend
                if "memory_backend" in defaults:
                    self.set_active_memory_backend(defaults["memory_backend"])
                
                # Set active model provider
                if "model_provider" in defaults:
                    self.set_active_model_provider(defaults["model_provider"])
            
            except Exception as e:
                logger.warning(f"Failed to load defaults from config: {e}")
        
        logger.info(f"Module registry initialized: {len(self.memory_backends)} backends, "
                   f"{len(self.model_providers)} providers, {len(self.processing_hooks)} hooks, "
                   f"{len(self.api_extensions)} extensions, {len(self.tools)} tools")
