"""
Web Search Tool
Searches the web for information
"""
from typing import Dict, Any

from modules.base import BaseTool


class WebSearchTool(BaseTool):
    """
    Tool for searching the web
    """
    
    @property
    def name(self) -> str:
        return "web_search"
    
    @property
    def description(self) -> str:
        return "Searches the web for information. Provide a search query to find relevant web pages."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 5)",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10
                }
            },
            "required": ["query"]
        }
    
    def execute(self, **kwargs) -> Any:
        """
        Search the web
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            Search results
        """
        query = kwargs.get("query", "")
        max_results = kwargs.get("max_results", 5)
        
        if not query:
            return {"error": "No query provided"}
        
        try:
            # Try to use duckduckgo-search if available
            try:
                from duckduckgo_search import DDGS
                ddgs = DDGS()
                results = []
                
                for result in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": result.get("title", ""),
                        "snippet": result.get("body", ""),
                        "url": result.get("href", "")
                    })
                
                return {
                    "query": query,
                    "results": results,
                    "count": len(results)
                }
            except ImportError:
                # Fallback: return mock results
                return {
                    "query": query,
                    "results": [
                        {
                            "title": f"Result {i+1} for '{query}'",
                            "snippet": f"This is a mock result for the query: {query}",
                            "url": f"https://example.com/result{i+1}"
                        }
                        for i in range(min(max_results, 3))
                    ],
                    "count": min(max_results, 3),
                    "note": "Web search library not installed. Install 'duckduckgo-search' for real results."
                }
        except Exception as e:
            return {
                "error": str(e),
                "query": query
            }
