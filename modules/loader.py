"""
Module loader for plugin discovery and loading
Hybrid discovery: auto-scan directories with config override
"""
import json
import logging
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Type
import inspect

from modules.base import (
    BaseMemoryBackend,
    BaseModelProvider,
    BaseProcessingHook,
    BaseAPIExtension,
    BaseTool
)

logger = logging.getLogger(__name__)


def discover_modules(
    modules_dir: Path,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, List[Type]]:
    """
    Discover plugin classes in modules directory
    
    Args:
        modules_dir: Directory to scan for modules
        config: Optional configuration dict for filtering
        
    Returns:
        Dictionary mapping plugin types to lists of discovered classes:
        {
            "memory_backends": [...],
            "model_providers": [...],
            "processing_hooks": [...],
            "api_extensions": [...],
            "tools": [...]
        }
    """
    discovered = {
        "memory_backends": [],
        "model_providers": [],
        "processing_hooks": [],
        "api_extensions": [],
        "tools": []
    }
    
    if not modules_dir.exists():
        logger.warning(f"Modules directory does not exist: {modules_dir}")
        return discovered
    
    # Get scan directories from config
    scan_dirs = []
    if config:
        scan_dirs = config.get("scan_directories", [])
    
    # Default scan directories if not specified
    if not scan_dirs:
        scan_dirs = ["official", "core", "community", "marketplace"]
    
    # Scan each directory
    for scan_dir in scan_dirs:
        scan_path = modules_dir / scan_dir
        if not scan_path.exists():
            continue
        
        logger.debug(f"Scanning directory: {scan_path}")
        
        # Recursively find Python files
        for py_file in scan_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
            
            try:
                # Load module
                spec = importlib.util.spec_from_file_location(
                    f"module_{py_file.stem}",
                    py_file
                )
                if spec is None or spec.loader is None:
                    continue
                
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find plugin classes
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # Skip if not defined in this module
                    if obj.__module__ != module.__name__:
                        continue
                    
                    # Check for each plugin type
                    if issubclass(obj, BaseMemoryBackend) and obj != BaseMemoryBackend:
                        discovered["memory_backends"].append(obj)
                        logger.debug(f"Found memory backend: {name} in {py_file}")
                    
                    elif issubclass(obj, BaseModelProvider) and obj != BaseModelProvider:
                        discovered["model_providers"].append(obj)
                        logger.debug(f"Found model provider: {name} in {py_file}")
                    
                    elif issubclass(obj, BaseProcessingHook) and obj != BaseProcessingHook:
                        discovered["processing_hooks"].append(obj)
                        logger.debug(f"Found processing hook: {name} in {py_file}")
                    
                    elif issubclass(obj, BaseAPIExtension) and obj != BaseAPIExtension:
                        discovered["api_extensions"].append(obj)
                        logger.debug(f"Found API extension: {name} in {py_file}")
                    
                    elif issubclass(obj, BaseTool) and obj != BaseTool:
                        discovered["tools"].append(obj)
                        logger.debug(f"Found tool: {name} in {py_file}")
            
            except Exception as e:
                logger.warning(f"Failed to load module {py_file}: {e}", exc_info=True)
                continue
    
    return discovered


def filter_discovered_modules(
    discovered: Dict[str, List[Type]],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, List[Type]]:
    """
    Filter discovered modules based on configuration
    
    Args:
        discovered: Dictionary of discovered plugin classes
        config: Configuration dict with enabled_modules
        
    Returns:
        Filtered dictionary of plugin classes
    """
    if not config:
        return discovered
    
    enabled = config.get("enabled_modules", {})
    if not enabled:
        return discovered
    
    filtered = {
        "memory_backends": [],
        "model_providers": [],
        "processing_hooks": [],
        "api_extensions": [],
        "tools": []
    }
    
    # Filter each plugin type
    for plugin_type in ["memory_backends", "model_providers", "processing_hooks", "api_extensions", "tools"]:
        enabled_list = enabled.get(plugin_type, [])
        
        if not enabled_list:
            # If not specified, include all discovered
            filtered[plugin_type] = discovered.get(plugin_type, [])
            continue
        
        # Filter by name
        for cls in discovered.get(plugin_type, []):
            class_name = cls.__name__
            # Check if class name matches any enabled name (case-insensitive)
            if any(enabled_name.lower() == class_name.lower() for enabled_name in enabled_list):
                filtered[plugin_type].append(cls)
    
    return filtered


def load_module(
    plugin_class: Type,
    config: Optional[Dict[str, Any]] = None
) -> Optional[Any]:
    """
    Instantiate a plugin class
    
    Args:
        plugin_class: Plugin class to instantiate
        config: Optional configuration dict
        
    Returns:
        Plugin instance or None if instantiation fails
    """
    try:
        # Get module settings if available
        module_settings = {}
        if config:
            module_settings = config.get("module_settings", {}).get(plugin_class.__name__, {})
        
        # Check if module is disabled
        if module_settings.get("enabled", True) is False:
            logger.debug(f"Skipping disabled module: {plugin_class.__name__}")
            return None
        
        # Try to instantiate with config if constructor accepts it
        sig = inspect.signature(plugin_class.__init__)
        params = sig.parameters
        
        if len(params) > 1 and "config" in params:
            instance = plugin_class(config=module_settings)
        else:
            instance = plugin_class()
        
        logger.info(f"Loaded module: {plugin_class.__name__}")
        return instance
    
    except Exception as e:
        logger.error(f"Failed to instantiate {plugin_class.__name__}: {e}", exc_info=True)
        return None


def load_all_modules(
    modules_dir: Path,
    config_path: Optional[Path] = None
) -> Dict[str, List[Any]]:
    """
    Main entry point: discover and load all modules
    
    Args:
        modules_dir: Directory to scan for modules
        config_path: Optional path to modules_config.json
        
    Returns:
        Dictionary mapping plugin types to lists of instances:
        {
            "memory_backends": [...],
            "model_providers": [...],
            "processing_hooks": [...],
            "api_extensions": [...],
            "tools": [...]
        }
    """
    # Load configuration
    config = None
    if config_path and config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded module configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
    
    # Discover modules
    discovered = discover_modules(modules_dir, config)
    
    # Filter based on config
    filtered = filter_discovered_modules(discovered, config)
    
    # Load modules
    loaded = {
        "memory_backends": [],
        "model_providers": [],
        "processing_hooks": [],
        "api_extensions": [],
        "tools": []
    }
    
    for plugin_type in ["memory_backends", "model_providers", "processing_hooks", "api_extensions", "tools"]:
        for cls in filtered.get(plugin_type, []):
            instance = load_module(cls, config)
            if instance:
                loaded[plugin_type].append(instance)
    
    logger.info(f"Loaded {sum(len(v) for v in loaded.values())} modules")
    return loaded
