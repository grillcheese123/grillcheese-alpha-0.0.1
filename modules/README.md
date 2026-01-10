# GrillCheese Plugin/Module System

A flexible plugin system that allows extending GrillCheese AI with custom memory backends, model providers, processing hooks, API extensions, and AI tools.

## Overview

The module system uses **hybrid discovery**: it auto-scans the `modules/` directory and can be configured via `modules_config.json`. Modules are organized into categories:

- **`official/`**: Officially maintained modules
- **`core/`**: Core modules (wrappers around existing functionality)
- **`community/`**: Community-contributed modules
- **`marketplace/`**: Marketplace/distributed modules

## Architecture

### Base Interfaces (`modules/base.py`)

All plugins must implement one of these base classes:

- **`BaseMemoryBackend`**: Memory storage backends
- **`BaseModelProvider`**: LLM model providers
- **`BaseProcessingHook`**: Pre/post-processing middleware
- **`BaseAPIExtension`**: Custom FastAPI routes and WebSocket handlers
- **`BaseTool`**: AI tools (MCP-style function calling)

### Module Loader (`modules/loader.py`)

- **`discover_modules()`**: Recursively scans directories for plugin classes
- **`load_module()`**: Instantiates a plugin class
- **`load_all_modules()`**: Main entry point for loading all modules

### Plugin Registry (`modules/registry.py`)

Singleton `ModuleRegistry` manages all loaded plugins:

- `memory_backends`: Dict[str, BaseMemoryBackend]
- `model_providers`: Dict[str, BaseModelProvider]
- `processing_hooks`: List[BaseProcessingHook] (ordered by priority)
- `api_extensions`: List[BaseAPIExtension]
- `tools`: Dict[str, BaseTool]

### Tool System (`modules/tools.py`)

MCP-style tool execution framework:

- **`ToolExecutor`**: Executes tools and manages tool calls
- **`ToolCall`**: Represents a tool call request
- **`ToolResult`**: Represents a tool execution result

## Quick Start

### Creating a Plugin

1. **Choose a plugin type** (memory backend, model provider, hook, API extension, or tool)

2. **Create a Python file** in the appropriate directory (`core/`, `community/`, etc.)

3. **Implement the base interface**:

```python
from modules.base import BaseTool

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
                "param1": {"type": "string", "description": "..."}
            },
            "required": ["param1"]
        }
    
    def execute(self, **kwargs) -> Any:
        # Tool implementation
        return {"result": "..."}
```

4. **The module will be auto-discovered** on next startup

### Configuration

Edit `modules_config.json` to control module loading:

```json
{
  "scan_directories": ["official", "core", "community", "marketplace"],
  "enabled_modules": {
    "memory_backends": ["SqliteFaissBackend"],
    "model_providers": ["GGUFModelProvider"],
    "processing_hooks": ["PromptEnhancerHook"],
    "api_extensions": [],
    "tools": ["calculator", "web_search"]
  },
  "defaults": {
    "memory_backend": "SqliteFaissBackend",
    "model_provider": "GGUFModelProvider"
  },
  "module_settings": {
    "PromptEnhancerHook": {
      "enabled": true,
      "priority": 10
    }
  }
}
```

## Plugin Types

### Memory Backends

Implement `BaseMemoryBackend` to provide alternative storage systems:

```python
class MyMemoryBackend(BaseMemoryBackend):
    @property
    def embedding_dim(self) -> int:
        return 384
    
    def store(self, embedding, text, metadata=None) -> str:
        # Store memory, return memory_id
        pass
    
    def retrieve(self, query_embedding, k=3) -> List[Tuple[str, float]]:
        # Retrieve similar memories, return List[(text, similarity_score)]
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
```

### Model Providers

Implement `BaseModelProvider` to provide alternative LLM backends:

```python
class MyModelProvider(BaseModelProvider):
    @property
    def embedding_dim(self) -> int:
        return 384
    
    def get_embedding(self, text: str) -> np.ndarray:
        # Return embedding vector
        pass
    
    def generate(self, prompt: str, context: List[str]) -> str:
        # Generate response
        pass
    
    def generate_with_tools(self, prompt, context, tools, tool_executor, max_iterations=5) -> str:
        # Optional: implement tool calling support
        pass
```

### Processing Hooks

Implement `BaseProcessingHook` to add middleware:

```python
class MyHook(BaseProcessingHook):
    @property
    def name(self) -> str:
        return "my_hook"
    
    @property
    def priority(self) -> int:
        return 10  # Lower = earlier in chain
    
    async def pre_process(self, prompt: str, context: Dict[str, Any]) -> str:
        # Modify prompt before generation
        return prompt
    
    async def post_process(self, response_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        # Modify response after generation
        return response_data
```

### API Extensions

Implement `BaseAPIExtension` to add custom routes:

```python
class MyAPIExtension(BaseAPIExtension):
    @property
    def name(self) -> str:
        return "my_api"
    
    def register_routes(self, app: FastAPI) -> None:
        @app.get("/api/my-endpoint")
        async def my_endpoint():
            return {"data": "..."}
    
    def register_websockets(self, app: FastAPI) -> None:
        # Optional: register WebSocket handlers
        pass
```

### Tools

Implement `BaseTool` to add AI function calling capabilities:

```python
class MyTool(BaseTool):
    @property
    def name(self) -> str:
        return "my_tool"
    
    @property
    def description(self) -> str:
        return "What this tool does"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "param": {"type": "string", "description": "..."}
            },
            "required": ["param"]
        }
    
    def execute(self, **kwargs) -> Any:
        # Execute tool logic
        return {"result": "..."}
    
    async def execute_async(self, **kwargs) -> Any:
        # Optional: async execution
        return await self.execute(**kwargs)
```

## Tool Calling

Tools can be called by the model during generation. The model outputs tool calls in JSON format:

```json
{
  "tool": "calculator",
  "parameters": {
    "expression": "2 + 2"
  }
}
```

The `ToolExecutor` parses these calls, executes the tools, and injects results back into the prompt for continued generation.

## Examples

See `modules/core/` for example implementations:

- **`sqlite_faiss_backend.py`**: Wrapper around existing MemoryStore
- **`gguf_model_provider.py`**: Wrapper around Phi3GGUF
- **`pytorch_model_provider.py`**: Wrapper around Phi3Model
- **`prompt_enhancer_hook.py`**: Example processing hook
- **`custom_api_extension.py`**: Example API extension
- **`tools/`**: Example tools (calculator, file_operations, web_search, etc.)

## Integration

The module system is integrated into:

- **`main.py`**: FastAPI server initialization
- **`cli/cli.py`**: CLI initialization and processing

Modules are loaded automatically on startup. The system falls back to legacy initialization if modules are unavailable.

## Best Practices

1. **Error Handling**: Always handle errors gracefully and log them
2. **Type Hints**: Use type hints for better IDE support and documentation
3. **Documentation**: Add docstrings to all methods
4. **Testing**: Test your plugins before deploying
5. **Configuration**: Use `module_settings` in `modules_config.json` for plugin-specific settings
6. **Priority**: Set appropriate priorities for hooks (lower = earlier)
7. **Async Support**: Implement async methods for long-running operations

## Troubleshooting

### Module Not Loading

- Check that the class inherits from the correct base class
- Verify the class is defined in the module (not imported)
- Check `modules_config.json` for enabled/disabled modules
- Review logs for import errors

### Tool Not Executing

- Verify tool is registered in `modules_config.json`
- Check tool name matches exactly
- Ensure `generate_with_tools()` is implemented in model provider
- Review tool parameter validation

### Hook Not Running

- Check hook priority (lower = earlier)
- Verify hook is enabled in `module_settings`
- Review pre_process/post_process return values

## Contributing

When contributing plugins:

1. Place in appropriate directory (`core/`, `community/`, etc.)
2. Follow naming conventions (snake_case for files, PascalCase for classes)
3. Add docstrings and type hints
4. Test thoroughly
5. Update this README if adding new patterns

## License

See main project license.
