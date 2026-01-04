# Module Directory Structure

The GrillCheese plugin system is organized into four main directories:

## Directory Organization

### `official/`
**Purpose**: Officially maintained and supported plugins by the GrillCheese project team.

- These plugins are tested, documented, and maintained
- They receive regular updates and bug fixes
- Suitable for production use
- Examples: Official integrations with major services, certified plugins

### `core/`
**Purpose**: Core plugins that wrap existing GrillCheese functionality.

- Default implementations that provide the base functionality
- Wrappers around existing components (MemoryStore, models, etc.)
- Always enabled by default
- Examples:
  - `sqlite_faiss_backend.py`: Wraps MemoryStore
  - `gguf_model_provider.py`: Wraps Phi3GGUF
  - `pytorch_model_provider.py`: Wraps Phi3Model
  - `tools/`: Core tools (calculator, file_operations, system_info, etc.)

### `community/`
**Purpose**: Community-contributed plugins.

- Created and maintained by the community
- Not officially supported but available for use
- May require additional dependencies
- Use at your own discretion
- Examples: Community integrations, experimental features, user-contributed tools

### `marketplace/`
**Purpose**: Marketplace plugins available for download and installation.

- Plugins downloaded from the GrillCheese marketplace
- Can be installed/uninstalled on demand
- May come from official or community sources
- Automatically placed here when installed via marketplace
- Examples: Premium plugins, featured integrations, downloadable tools

## Configuration

You can control which directories are scanned in `modules_config.json`:

```json
{
  "scan_directories": {
    "official": true,
    "core": true,
    "community": true,
    "marketplace": true
  }
}
```

Set a directory to `false` to exclude it from scanning.

## Migration from `examples/`

The `examples/` directory still exists for reference, but new plugins should be placed in:
- `core/` for core functionality wrappers
- `official/` for official plugins
- `community/` for community contributions
- `marketplace/` for plugins installed via marketplace (auto-populated)

## Best Practices

1. **Core plugins**: Place in `core/` if they wrap existing functionality
2. **Official plugins**: Place in `official/` if maintained by the project team
3. **Community plugins**: Place in `community/` if contributed by users
4. **Documentation**: Include README.md in your plugin directory explaining usage
5. **Dependencies**: List all dependencies in your plugin's docstring or README

## Plugin Discovery

The loader automatically scans all enabled directories recursively. Plugins are discovered by:

1. Scanning enabled directories (`official/`, `core/`, `community/`, `marketplace/`)
2. Finding Python files (excluding `__init__.py`, `base.py`, etc.)
3. Detecting classes that inherit from base interfaces
4. Filtering based on `enabled_modules` in config
5. Registering with the ModuleRegistry

**Scan Order**: official → core → community → marketplace (first match takes precedence)

## Examples

### Adding a Core Plugin

Place your plugin in `modules/core/`:

```python
# modules/core/my_core_plugin.py
from modules.base import BaseTool
# ... implementation
```

### Adding an Official Plugin

Place your plugin in `modules/official/`:

```python
# modules/official/my_official_plugin.py
from modules.base import BaseAPIExtension
# ... implementation
```

### Adding a Community Plugin

Place your plugin in `modules/community/`:

```python
# modules/community/my_community_plugin.py
from modules.base import BaseTool
# ... implementation
```

### Marketplace Plugins

Marketplace plugins are automatically placed in `modules/marketplace/` when installed via the marketplace system. You typically don't manually add files here - they're installed through the marketplace interface.

However, if you want to manually install a marketplace plugin:

```python
# modules/marketplace/my_marketplace_plugin.py
from modules.base import BaseTool
# ... implementation
```

## Notes

- All directories support subdirectories (e.g., `core/tools/`)
- The loader recursively scans all subdirectories
- Plugin names should be unique across all directories
- If duplicate names exist, the first one found takes precedence (order: official → core → community → marketplace)
- Marketplace plugins are typically installed automatically via marketplace tools
- Marketplace directory can be excluded from scanning if you don't want marketplace plugins loaded

