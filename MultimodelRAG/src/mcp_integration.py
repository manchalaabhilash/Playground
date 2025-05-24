"""
MCP integration for advanced RAG techniques.
Implements Model-Client-Protocol routing for improved retrieval and response generation.
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional

class McpRouter:
    """
    MCP Router for advanced RAG techniques.
    Routes queries to appropriate models based on content type and complexity.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the MCP Router"""
        self.api_key = api_key or os.environ.get("LLM_API_KEY")
        self.model_preferences = {
            "text": {
                "hints": [
                    {"name": "claude-3-sonnet"},
                    {"name": "claude"}
                ],
                "intelligencePriority": 0.8,
                "speedPriority": 0.5,
                "costPriority": 0.3
            },
            "image": {
                "hints": [
                    {"name": "claude-3-opus"},
                    {"name": "gpt-4-vision"}
                ],
                "intelligencePriority": 0.9,
                "speedPriority": 0.3,
                "costPriority": 0.2
            },
            "mixed": {
                "hints": [
                    {"name": "claude-3-opus"},
                    {"name": "gpt-4-vision"}
                ],
                "intelligencePriority": 0.9,
                "speedPriority": 0.4,
                "costPriority": 0.2
            }
        }
    
    def determine_content_type(self, relevant_texts: List[Any], relevant_images: List[Any]) -> str:
        """Determine the content type based on retrieved documents"""
        has_text = len(relevant_texts) > 0
        has_images = len(relevant_images) > 0
        
        if has_text and has_images:
            return "mixed"
        elif has_images:
            return "image"
        else:
            return "text"
    
    def route_query(self, question: str, relevant_texts: List[Any], relevant_images: List[Any]) -> Dict[str, Any]:
        """Route the query to the appropriate model based on content type"""
        content_type = self.determine_content_type(relevant_texts, relevant_images)
        model_prefs = self.model_preferences[content_type]
        
        # Prepare the request for the MCP client
        request = {
            "messages": [{"role": "user", "content": question}],
            "modelPreferences": model_prefs,
            "context": {
                "texts": [doc.page_content for doc in relevant_texts] if relevant_texts else [],
                "images": [img["image_path"] for img in relevant_images] if relevant_images else []
            }
        }
        
        return request

class McpClient:
    """
    MCP Client for interacting with MCP-compatible AI services.
    Supports both local and remote MCP servers.
    """
    
    def __init__(self, server_url: Optional[str] = None):
        """Initialize the MCP Client"""
        self.server_url = server_url or os.environ.get("MCP_SERVER_URL", "http://localhost:8080")
        self.capabilities = self._get_server_capabilities()
    
    def _get_server_capabilities(self) -> Dict[str, Any]:
        """Get the capabilities of the MCP server"""
        try:
            response = requests.get(f"{self.server_url}/capabilities")
            return response.json()
        except Exception as e:
            print(f"Warning: Could not get MCP server capabilities: {str(e)}")
            # Return default capabilities
            return {
                "sampling": True,
                "resources": True,
                "tools": True,
                "prompts": True,
                "logging": True
            }
    
    def create_message(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Create a message using the MCP server"""
        try:
            response = requests.post(
                f"{self.server_url}/messages",
                json=request,
                headers={"Content-Type": "application/json"}
            )
            return response.json()
        except Exception as e:
            print(f"Error creating message with MCP server: {str(e)}")
            # Fallback to a simple response
            return {
                "content": {
                    "text": f"Error: Could not generate response using MCP server. {str(e)}"
                }
            }
    
    def supports_sampling(self) -> bool:
        """Check if the MCP server supports sampling"""
        return self.capabilities.get("sampling", False)
    
    def supports_resources(self) -> bool:
        """Check if the MCP server supports resources"""
        return self.capabilities.get("resources", False)

class McpRagOrchestrator:
    """
    MCP RAG Orchestrator for advanced RAG techniques.
    Orchestrates the RAG process using MCP routing and client capabilities.
    """
    
    def __init__(self):
        """Initialize the MCP RAG Orchestrator"""
        self.router = McpRouter()
        self.client = McpClient()
    
    def process_query(self, question: str, relevant_texts: List[Any], relevant_images: List[Any]) -> str:
        """Process a query using MCP routing and client capabilities"""
        # Route the query to the appropriate model
        request = self.router.route_query(question, relevant_texts, relevant_images)
        
        # Check if client supports resources for images
        if relevant_images and not self.client.supports_resources():
            print("Warning: MCP client does not support resources, images will not be included")
            # Remove images from the request
            request["context"]["images"] = []
        
        # Create the message using the MCP client
        response = self.client.create_message(request)
        
        # Extract the answer from the response
        answer = response.get("content", {}).get("text", "No response generated")
        
        return answer