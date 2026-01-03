"""
Tests for FastAPI main.py endpoints
"""
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestHealthEndpoint:
    """Tests for /health endpoint"""
    
    def test_health_endpoint_exists(self):
        """Health endpoint should be defined"""
        from main import app
        
        routes = [route.path for route in app.routes]
        assert "/health" in routes
    
    def test_health_returns_dict(self):
        """Health check should return expected structure"""
        from main import health_check
        import asyncio
        
        result = asyncio.get_event_loop().run_until_complete(health_check())
        
        assert isinstance(result, dict)
        assert 'status' in result
        assert 'model_loaded' in result
        assert 'memory_initialized' in result
        assert 'snn_initialized' in result


class TestStatsEndpoint:
    """Tests for /stats endpoint"""
    
    def test_stats_endpoint_exists(self):
        """Stats endpoint should be defined"""
        from main import app
        
        routes = [route.path for route in app.routes]
        assert "/stats" in routes


class TestWebSocketEndpoint:
    """Tests for /ws WebSocket endpoint"""
    
    def test_websocket_endpoint_exists(self):
        """WebSocket endpoint should be defined"""
        from main import app
        
        # WebSocket routes are handled differently
        has_ws = any(
            hasattr(route, 'path') and route.path == "/ws" 
            for route in app.routes
        )
        assert has_ws


class TestAppConfiguration:
    """Tests for app configuration"""
    
    def test_cors_middleware_enabled(self):
        """CORS middleware should be enabled"""
        from main import app
        
        middleware_classes = [m.cls.__name__ for m in app.user_middleware]
        assert 'CORSMiddleware' in middleware_classes
    
    def test_app_has_title(self):
        """App should have a title"""
        from main import app
        
        assert app.title is not None
        assert len(app.title) > 0


class TestInitModel:
    """Tests for model initialization logic"""
    
    def test_init_model_function_exists(self):
        """_init_model function should exist"""
        from main import _init_model
        
        assert callable(_init_model)


class TestProcessPrompt:
    """Tests for prompt processing logic"""
    
    def test_process_prompt_function_exists(self):
        """_process_prompt function should exist"""
        from main import _process_prompt
        
        assert callable(_process_prompt)

