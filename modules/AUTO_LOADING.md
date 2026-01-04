# Automatic Module Loading

## Overview

Yes, modules are **automatically loaded** when the GrillCheese system starts. This happens in both the FastAPI server (`main.py`) and the CLI (`cli/cli.py`).

## How It Works

### 1. Server Startup (`main.py`)

When the FastAPI server starts, the `startup_event()` function automatically:

1. Creates a `ModuleRegistry` instance
2. Calls `await registry.load_all_modules()` which:
   - Scans the `modules/` directory (official, core, community, marketplace)
   - Discovers plugin classes
   - Instantiates and registers them
   - Sets active plugins based on config defaults

```python
@app.on_event("startup")
async def startup_event():
    # ...
    registry = ModuleRegistry()
    await registry.load_all_modules()
    # ...
```

### 2. CLI Startup (`cli/cli.py`)

Similarly, when the CLI starts:

1. Creates a `ModuleRegistry` instance
2. Calls `asyncio.run(registry.load_all_modules())`
3. Uses registered plugins or falls back to direct initialization

```python
registry = ModuleRegistry()
asyncio.run(registry.load_all_modules())
```

## What Gets Loaded

### Discovery Process

1. **Directory Scanning**: Scans `modules/official/`, `modules/core/`, `modules/community/`, and `modules/marketplace/`
2. **Plugin Detection**: Finds Python files containing classes that inherit from base interfaces
3. **Instantiation**: Creates instances of discovered plugins
4. **Registration**: Registers them in the ModuleRegistry
5. **Activation**: Sets active plugins based on `modules_config.json` defaults

### Default Behavior

- **Empty `enabled_modules` lists**: All discovered plugins are loaded
- **Config defaults**: `sqlite_faiss` backend and `gguf` provider are set as active
- **Alias matching**: Common aliases are registered (e.g., `gguf` → `GGUFModelProvider`)

## Configuration

### `modules_config.json`

```json
{
  "enabled_modules": {
    "memory_backends": [],      // Empty = load all
    "model_providers": [],      // Empty = load all
    "processing_hooks": [],      // Empty = load all
    "api_extensions": [],       // Empty = load all
    "tools": []                 // Empty = load all
  },
  "defaults": {
    "memory_backend": "sqlite_faiss",
    "model_provider": "gguf"
  },
  "scan_directories": {
    "official": true,
    "core": true,
    "community": true,
    "marketplace": true
  }
}
```

### Disabling Auto-Loading

To disable automatic loading:

1. Set `AUTO_DISCOVER = False` in `config.py` `ModuleConfig`
2. Or set `scan_directories` to `false` for specific directories
3. Or specify exact plugins in `enabled_modules` lists

## Fallback Behavior

If no plugins are found or loading fails:

- **Model Provider**: Falls back to direct `_init_model()` call
- **Memory Backend**: Falls back to direct `MemoryStore()` initialization
- **Tools**: Simply not available (system works without tools)
- **Hooks**: Simply not executed (system works without hooks)

This ensures the system always starts, even if plugins fail to load.

## Logging

Module loading is logged at INFO level:

```
Loading plugins and modules...
Registered memory backend: sqlitefaissbackend
Registered model provider: ggufmodelprovider
Registered tool: calculator
[OK] Plugins loaded
```

Errors are logged but don't stop the system:

```
Failed to register memory backend SQLiteFAISSBackend: <error>
```

## Verification

To verify modules are loading:

1. Check startup logs for "Registered ..." messages
2. Check `registry.get_tools()` returns tools
3. Check `registry.get_active_memory_backend()` returns a backend
4. Check `registry.get_active_model_provider()` returns a provider

## Manual Loading

You can also manually load modules:

```python
from modules.registry import ModuleRegistry

registry = ModuleRegistry()
await registry.load_all_modules()

# Or with custom paths
await registry.load_all_modules(
    modules_dir=Path("custom/modules"),
    config_path=Path("custom/config.json")
)
```

## Troubleshooting

### Modules Not Loading

1. **Check directory structure**: Ensure `modules/core/`, etc. exist
2. **Check plugin files**: Ensure Python files are in correct directories
3. **Check imports**: Ensure plugins can import base classes
4. **Check config**: Ensure `scan_directories` includes the directory
5. **Check logs**: Look for error messages during loading

### Wrong Plugin Active

1. **Check config defaults**: Verify `modules_config.json` defaults
2. **Check plugin names**: Ensure config names match registered names
3. **Check aliases**: Registry creates aliases for common names
4. **Manual override**: Use `registry.set_active_memory_backend(name)`

### Performance Issues

Module loading happens once at startup. If it's slow:

1. **Reduce scan directories**: Disable unused directories in config
2. **Specify enabled modules**: Use explicit lists instead of empty lists
3. **Check plugin initialization**: Some plugins may do expensive setup

