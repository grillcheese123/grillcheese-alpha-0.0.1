"""
Module loader for GrillCheese plugin system

Handles discovery and loading of plugins from the modules directory.
Supports hybrid discovery: auto-scan with optional config override.
"""
import json
import logging
import importlib.util
from pathlib import Path
from typing import List, Dict, Any, Optional, Type
import inspect

from config import ModuleConfig
from .base import (
    BaseMemoryBackend,
    BaseModelProvider,
    BaseProcessingHook,
    BaseAPIExtension,
    BaseTool
)

logger = logging.getLogger(__name__)


def discover_modules(modules_dir: Optional[Path] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Discover plugin classes in the modules directory.
    
    Args:
        modules_dir: Directory to scan (defaults to ModuleConfig.MODULES_DIR)
        config: Optional configuration dictionary for filtering directories
        
    Returns:
        Dictionary mapping plugin types to lists of discovered plugins:
        {
            'memory_backends': [...],
            'model_providers': [...],
            'processing_hooks': [...],
            'api_extensions': [...],
            'tools': [...]
        }
    """
    if modules_dir is None:
        modules_dir = ModuleConfig.MODULES_DIR
    
    if not modules_dir.exists():
        logger.warning(f"Modules directory does not exist: {modules_dir}")
        return {
            'memory_backends': [],
            'model_providers': [],
            'processing_hooks': [],
            'api_extensions': [],
            'tools': []
        }
    
    discovered = {
        'memory_backends': [],
        'model_providers': [],
        'processing_hooks': [],
        'api_extensions': [],
        'tools': []
    }
    
    # Recursively scan for Python files
    # Scan official, core, community, and marketplace directories (filtered by config if provided)
    scan_dirs = ['official', 'core', 'community', 'marketplace']
    
    # Filter directories based on config
    if config:
        scan_settings = config.get('scan_directories', {})
        scan_dirs = [d for d in scan_dirs if scan_settings.get(d, True)]
    
    for dir_name in scan_dirs:
        dir_path = modules_dir / dir_name
        if not dir_path.exists():
            continue
        
        for py_file in dir_path.rglob("*.py"):
            # Skip __init__.py and base.py
            if py_file.name in ('__init__.py', 'base.py', 'loader.py', 'registry.py', 'tools.py'):
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
                    # Skip base classes and imports
                    if obj.__module__ != module.__name__:
                        continue
                    
                    # Check for each plugin type
                    if issubclass(obj, BaseMemoryBackend) and obj != BaseMemoryBackend:
                        discovered['memory_backends'].append({
                            'name': getattr(obj, '__name__', name),
                            'class': obj,
                            'module': module,
                            'file': str(py_file)
                        })
                    elif issubclass(obj, BaseModelProvider) and obj != BaseModelProvider:
                        discovered['model_providers'].append({
                            'name': getattr(obj, '__name__', name),
                            'class': obj,
                            'module': module,
                            'file': str(py_file)
                        })
                    elif issubclass(obj, BaseProcessingHook) and obj != BaseProcessingHook:
                        discovered['processing_hooks'].append({
                            'name': getattr(obj, '__name__', name),
                            'class': obj,
                            'module': module,
                            'file': str(py_file)
                        })
                    elif issubclass(obj, BaseAPIExtension) and obj != BaseAPIExtension:
                        discovered['api_extensions'].append({
                            'name': getattr(obj, '__name__', name),
                            'class': obj,
                            'module': module,
                            'file': str(py_file)
                        })
                    elif issubclass(obj, BaseTool) and obj != BaseTool:
                        discovered['tools'].append({
                            'name': getattr(obj, '__name__', name),
                            'class': obj,
                            'module': module,
                            'file': str(py_file)
                        })
            
            except Exception as e:
                logger.warning(f"Failed to load module {py_file}: {e}")
                continue
    
    return discovered


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load module configuration from JSON file.
    
    Args:
        config_path: Path to config file (defaults to ModuleConfig.MODULES_CONFIG_FILE)
        
    Returns:
        Configuration dictionary with default values if file doesn't exist
    """
    if config_path is None:
        config_path = ModuleConfig.MODULES_CONFIG_FILE
    
    default_config = {
        "enabled_modules": {
            "memory_backends": [],
            "model_providers": [],
            "processing_hooks": [],
            "api_extensions": [],
            "tools": []
        },
        "defaults": {
            "memory_backend": ModuleConfig.DEFAULT_MEMORY_BACKEND,
            "model_provider": ModuleConfig.DEFAULT_MODEL_PROVIDER
        },
        "module_settings": {},
        "scan_directories": {
            "official": True,
            "core": True,
            "community": True,
            "marketplace": True
        }
    }
    
    if not config_path.exists():
        logger.info(f"Module config file not found, using defaults: {config_path}")
        return default_config
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Merge with defaults
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if sub_key not in config[key]:
                        config[key][sub_key] = sub_value
        
        return config
    
    except Exception as e:
        logger.error(f"Failed to load module config: {e}, using defaults")
        return default_config


def filter_discovered_modules(
    discovered: Dict[str, List[Dict[str, Any]]],
    config: Dict[str, Any]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Filter discovered modules based on configuration.
    
    Args:
        discovered: Discovered modules dictionary
        config: Configuration dictionary
        
    Returns:
        Filtered modules dictionary
    """
    enabled = config.get('enabled_modules', {})
    filtered = {
        'memory_backends': [],
        'model_providers': [],
        'processing_hooks': [],
        'api_extensions': [],
        'tools': []
    }
    
    for plugin_type in filtered.keys():
        enabled_list = enabled.get(plugin_type, [])
        discovered_list = discovered.get(plugin_type, [])
        
        if not enabled_list:
            # If empty list, enable all discovered modules
            filtered[plugin_type] = discovered_list
        else:
            # Filter by enabled names
            for module_info in discovered_list:
                module_name = module_info['name'].lower()
                if any(enabled_name.lower() in module_name or module_name in enabled_name.lower() 
                       for enabled_name in enabled_list):
                    filtered[plugin_type].append(module_info)
    
    return filtered


def load_all_modules(
    modules_dir: Optional[Path] = None,
    config_path: Optional[Path] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Main entry point: discover and load all modules.
    
    Args:
        modules_dir: Directory to scan (defaults to ModuleConfig.MODULES_DIR)
        config_path: Path to config file (defaults to ModuleConfig.MODULES_CONFIG_FILE)
        
    Returns:
        Dictionary of loaded modules ready for registration
    """
    # Load configuration
    config = load_config(config_path)
    
    # Discover modules
    if ModuleConfig.AUTO_DISCOVER:
        discovered = discover_modules(modules_dir, config)
    else:
        discovered = {
            'memory_backends': [],
            'model_providers': [],
            'processing_hooks': [],
            'api_extensions': [],
            'tools': []
        }
    
    # Filter based on config
    filtered = filter_discovered_modules(discovered, config)
    
    logger.info(f"Discovered modules: "
                f"{len(filtered['memory_backends'])} memory backends, "
                f"{len(filtered['model_providers'])} model providers, "
                f"{len(filtered['processing_hooks'])} processing hooks, "
                f"{len(filtered['api_extensions'])} API extensions, "
                f"{len(filtered['tools'])} tools")
    
    return filtered

