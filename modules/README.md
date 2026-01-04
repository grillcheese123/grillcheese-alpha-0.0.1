# GrillCheese Plugin/Module System

The GrillCheese plugin system allows you to extend the AI assistant with custom functionality through a flexible plugin architecture.

## Overview

The plugin system supports five types of plugins:

1. **Memory Backends**: Alternative storage systems (e.g., PostgreSQL, ChromaDB, Pinecone)
2. **Model Providers**: Different LLM backends (e.g., OpenAI, Anthropic, local models)
3. **Processing Hooks**: Pre/post-processing middleware (e.g., prompt enhancement, response filtering)
4. **API Extensions**: Custom FastAPI routes and WebSocket handlers
5. **AI Tools**: MCP-style function calling tools (e.g., web search, file operations, database queries)

## Directory Structure

The modules directory is organized into four categories:

- **`official/`**: Officially maintained and supported plugins
- **`core/`**: Core plugins that wrap existing GrillCheese functionality (default implementations)
- **`community/`**: Community-contributed plugins
- **`marketplace/`**: Marketplace plugins available for download and installation

```
modules/
├── official/          # Official plugins
├── core/              # Core plugins (default implementations)
│   ├── tools/         # Core tools
│   └── ...
├── community/         # Community plugins
├── marketplace/       # Marketplace plugins (downloadable)
├── base.py            # Base interfaces
├── loader.py          # Module loader
├── registry.py        # Plugin registry
├── tools.py           # Tool execution framework
└── modules_config.json # Configuration
```

## Architecture

### Base Interfaces

All plugins must implement the appropriate base interface from `modules.base`:

- `BaseMemoryBackend`: Memory storage interface
- `BaseModelProvider`: LLM provider interface
- `BaseProcessingHook`: Processing middleware interface
- `BaseAPIExtension`: API extension interface
- `BaseTool`: AI tool interface

### Module Discovery

The system uses **hybrid discovery**:

1. **Auto-scan**: Automatically discovers plugins in the `modules/` directory
2. **Config override**: `modules_config.json` can enable/disable specific modules

### Plugin Registry

The `ModuleRegistry` singleton manages all loaded plugins and provides access to active plugins.

## Creating a Plugin

### Memory Backend Plugin

```python
from modules.base import BaseMemoryBackend
import numpy as np
from typing import List, Dict, Any, Optional

class MyMemoryBackend(BaseMemoryBackend):
    def __init__(self, **kwargs):
        # Initialize your backend
        pass
    
    def store(self, embedding: np.ndarray, text: str) -> None:
        # Store memory
        pass
    
    def retrieve(self, embedding: np.ndarray, k: int = 3) -> List[str]:
        # Retrieve k most similar memories
        pass
    
    def clear(self) -> None:
        # Clear all memories
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        # Return statistics
        pass
    
    def get_identity(self) -> Optional[str]:
        # Return identity text
        pass
    
    @property
    def embedding_dim(self) -> int:
        # Return embedding dimension
        return 384
```

### Model Provider Plugin

```python
from modules.base import BaseModelProvider
import numpy as np
from typing import List

class MyModelProvider(BaseModelProvider):
    def __init__(self, **kwargs):
        # Initialize your model
        pass
    
    def get_embedding(self, text: str) -> np.ndarray:
        # Extract embedding
        pass
    
    def generate(self, prompt: str, context: List[str]) -> str:
        # Generate response
        pass
    
    @property
    def embedding_dim(self) -> int:
        # Return embedding dimension
        return 384
```

### Processing Hook Plugin

```python
from modules.base import BaseProcessingHook
from typing import Dict, Any

class MyProcessingHook(BaseProcessingHook):
    async def pre_process(self, prompt: str, context: Dict[str, Any]) -> str:
        # Modify prompt before generation
        return prompt
    
    async def post_process(self, response_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        # Modify response after generation
        return response_data
    
    async def on_error(self, error: Exception, context: Dict[str, Any]) -> None:
        # Handle errors
        pass
```

### API Extension Plugin

```python
from modules.base import BaseAPIExtension
from fastapi import FastAPI

class MyAPIExtension(BaseAPIExtension):
    def register_routes(self, app: FastAPI) -> None:
        @app.get("/api/my-endpoint")
        async def my_endpoint():
            return {"message": "Hello from plugin"}
    
    def register_websockets(self, app: FastAPI) -> None:
        # Register WebSocket handlers
        pass
```

### AI Tool Plugin

```python
from modules.base import BaseTool
from typing import Any, Dict

class MyTool(BaseTool):
    @property
    def name(self) -> str:
        return "my_tool"
    
    @property
    def description(self) -> str:
        return "Description of what this tool does"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "Parameter description"
                }
            },
            "required": ["param1"]
        }
    
    async def execute(self, **kwargs: Any) -> Any:
        # Execute tool logic
        param1 = kwargs.get('param1')
        return {"result": f"Processed {param1}"}
```

## Tool Development Guide

### Tool Calling Flow

1. Model generates response with tool call JSON
2. Tool executor parses tool calls
3. Tools are executed (with validation)
4. Results are injected back into prompt
5. Model generates final response

### Tool Call Format

Tools are called using JSON format:

```json
{
  "tool": "tool_name",
  "parameters": {
    "param1": "value1"
  }
}
```

Multiple tools can be called:

```json
[
  {"tool": "tool1", "parameters": {...}},
  {"tool": "tool2", "parameters": {...}}
]
```

### Parameter Validation

Tools automatically validate parameters against their JSON schema. Invalid parameters result in an error response.

### Example Tools

See `modules/examples/tools/` for example implementations:

- `calculator_tool.py`: Mathematical calculations
- `file_operations_tool.py`: File read/write operations
- `system_info_tool.py`: System information
- `web_search_tool.py`: Web search (placeholder)
- `memory_query_tool.py`: Advanced memory querying

## Configuration

### modules_config.json

```json
{
  "enabled_modules": {
    "memory_backends": [],
    "model_providers": [],
    "processing_hooks": [],
    "api_extensions": [],
    "tools": []
  },
  "defaults": {
    "memory_backend": "sqlite_faiss",
    "model_provider": "gguf"
  },
  "module_settings": {},
  "scan_directories": {
    "official": true,
    "core": true,
    "community": true,
    "marketplace": true
  }
}
```

**Configuration Options**:

- `enabled_modules`: Lists of enabled plugins by type (empty = enable all discovered)
- `defaults`: Default plugins to use
- `module_settings`: Per-module settings
- `scan_directories`: Which directories to scan (official, core, community, marketplace)

### Empty Lists

If a plugin type has an empty list in `enabled_modules`, all discovered plugins of that type will be loaded.

## Example Plugins

See `modules/core/` for core plugin implementations that wrap existing functionality:

- `sqlite_faiss_backend.py`: Wraps existing MemoryStore
- `gguf_model_provider.py`: Wraps existing Phi3GGUF
- `pytorch_model_provider.py`: Wraps existing Phi3Model
- `prompt_enhancer_hook.py`: Example processing hook
- `custom_api_extension.py`: Example API extension
- `tools/`: Core tools (calculator, file_operations, system_info, etc.)

For reference examples, see `modules/examples/` (legacy location).

- `sqlite_faiss_backend.py`: Wraps existing MemoryStore
- `gguf_model_provider.py`: Wraps existing Phi3GGUF
- `pytorch_model_provider.py`: Wraps existing Phi3Model
- `prompt_enhancer_hook.py`: Example processing hook
- `custom_api_extension.py`: Example API extension

## Best Practices

1. **Error Handling**: Always handle errors gracefully and log them
2. **Type Hints**: Use proper type hints for better IDE support
3. **Documentation**: Document your plugin's purpose and usage
4. **Testing**: Test your plugins before deploying
5. **Backward Compatibility**: Maintain compatibility with existing APIs

## Integration

Plugins are automatically loaded on application startup. The registry provides access to:

- Active memory backend: `registry.get_active_memory_backend()`
- Active model provider: `registry.get_active_model_provider()`
- All tools: `registry.get_tools()`
- Processing hooks: `registry.processing_hooks`
- API extensions: `registry.api_extensions`

## Troubleshooting

### Plugin Not Loading

1. Check that your plugin class inherits from the correct base class
2. Verify the plugin is in the `modules/` directory
3. Check `modules_config.json` to ensure the plugin is enabled
4. Review logs for import errors

### Tool Not Working

1. Verify tool name matches exactly
2. Check parameter schema matches tool call
3. Review tool execution logs for errors
4. Ensure tool is registered in `modules_config.json`

### Import Errors

1. Ensure all dependencies are installed
2. Check Python path includes the modules directory
3. Verify relative imports are correct

## API Reference

See `modules/base.py` for complete interface definitions.

## License

Same as GrillCheese project.
