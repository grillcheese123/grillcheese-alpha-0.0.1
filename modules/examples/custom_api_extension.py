"""
Custom API Extension Example

Demonstrates how to add custom FastAPI routes and WebSocket handlers.
"""
from fastapi import FastAPI, WebSocket
from typing import Dict, Any

from modules.base import BaseAPIExtension


class CustomAPIExtension(BaseAPIExtension):
    """
    Example API extension that adds custom endpoints.
    """
    
    def register_routes(self, app: FastAPI) -> None:
        """
        Register custom FastAPI routes.
        
        Args:
            app: FastAPI application instance
        """
        @app.get("/api/custom/example")
        async def custom_example():
            """Example custom endpoint."""
            return {"message": "This is a custom API extension"}
        
        @app.get("/api/custom/status")
        async def custom_status():
            """Example status endpoint."""
            return {"status": "ok", "extension": "custom_api_extension"}
    
    def register_websockets(self, app: FastAPI) -> None:
        """
        Register custom WebSocket handlers.
        
        Args:
            app: FastAPI application instance
        """
        # Example: WebSocket handlers would be registered here
        # For now, this is a placeholder
        pass

