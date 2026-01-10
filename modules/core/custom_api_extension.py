"""
Custom API Extension Example
Demonstrates how to add custom FastAPI routes
"""
from fastapi import FastAPI
from typing import Dict, Any

from modules.base import BaseAPIExtension


class CustomAPIExtension(BaseAPIExtension):
    """
    Example API extension that adds custom routes
    """
    
    @property
    def name(self) -> str:
        return "custom_api"
    
    def register_routes(self, app: FastAPI) -> None:
        """
        Register custom FastAPI routes
        
        Args:
            app: FastAPI application instance
        """
        @app.get("/api/custom/health")
        async def custom_health():
            """Custom health check endpoint"""
            return {"status": "ok", "extension": "custom_api"}
        
        @app.get("/api/custom/info")
        async def custom_info():
            """Custom info endpoint"""
            return {
                "name": self.name,
                "description": "Example custom API extension"
            }
