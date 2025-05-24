"""
MCP Server implementation for the Multimodal RAG system.
Allows other applications to use the RAG system through the MCP protocol.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional

sys.path.append('.')
from src.multimodal_rag import MultimodalRAG
from src.config import MCP_SERVER_INFO

class McpSyncServer:
    """
    Synchronous MCP Server implementation.
    Provides MCP-compatible endpoints for the Multimodal RAG system.
    """
    
    def __init__(self, server_info=None, capabilities=None):
        """Initialize the MCP Server"""
        self.server_info = server_info or MCP_SERVER_INFO
        self.capabilities = capabilities or {
            "resources": True,
            "tools": True,
            "prompts": True,
            "logging": True,
            "sampling": True
        }
        self.tools = {}
        self.resources = {}
        self.prompts = {}
        self.rag_system = None
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("mcp-server")
    
    def initialize_rag(self, document_paths=None, image_paths=None):
        """Initialize the RAG system"""
        self.rag_system = MultimodalRAG(
            document_paths=document_paths,
            image_paths=image_paths,
            use_mcp=True
        )
        
        # Process documents and images
        num_text_chunks = self.rag_system.process_documents()
        num_images = self.rag_system.process_images()
        
        # Initialize vector database
        self.rag_system.initialize_vector_db()
        
        self.logger.info(f"RAG system initialized with {num_text_chunks} text chunks and {num_images} images")
        
        # Register RAG tool
        self.add_tool({
            "name": "rag-query",
            "description": "Query the RAG system with a question",
            "schema": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask the RAG system"
                    },
                    "use_rag": {
                        "type": "boolean",
                        "description": "Whether to use RAG or direct LLM",
                        "default": True
                    }
                },
                "required": ["question"]
            }
        }, self._rag_query_handler)
        
        return {
            "num_text_chunks": num_text_chunks,
            "num_images": num_images
        }
    
    def _rag_query_handler(self, exchange, arguments):
        """Handle RAG query tool calls"""
        question = arguments.get("question")
        use_rag = arguments.get("use_rag", True)
        
        if not self.rag_system:
            return {"result": "RAG system not initialized", "success": False}
        
        try:
            answer = self.rag_system.answer_question(question, use_rag=use_rag)
            return {"result": answer, "success": True}
        except Exception as e:
            self.logger.error(f"Error in RAG query: {str(e)}")
            return {"result": f"Error: {str(e)}", "success": False}
    
    def add_tool(self, tool_spec, handler):
        """Add a tool to the server"""
        tool_name = tool_spec["name"]
        self.tools[tool_name] = {
            "spec": tool_spec,
            "handler": handler
        }
        self.logger.info(f"Tool '{tool_name}' registered")
        return True
    
    def add_resource(self, resource_spec):
        """Add a resource to the server"""
        resource_name = resource_spec["name"]
        self.resources[resource_name] = resource_spec
        self.logger.info(f"Resource '{resource_name}' registered")
        return True
    
    def add_prompt(self, prompt_spec):
        """Add a prompt to the server"""
        prompt_name = prompt_spec["name"]
        self.prompts[prompt_name] = prompt_spec
        self.logger.info(f"Prompt '{prompt_name}' registered")
        return True
    
    def handle_request(self, request):
        """Handle an MCP request"""
        request_type = request.get("type")
        
        if request_type == "get_capabilities":
            return {
                "type": "capabilities",
                "server_info": self.server_info,
                "capabilities": self.capabilities
            }
        
        elif request_type == "get_tools":
            return {
                "type": "tools",
                "tools": [tool["spec"] for tool in self.tools.values()]
            }
        
        elif request_type == "call_tool":
            tool_name = request.get("tool_name")
            arguments = request.get("arguments", {})
            
            if tool_name not in self.tools:
                return {
                    "type": "tool_result",
                    "success": False,
                    "error": f"Tool '{tool_name}' not found"
                }
            
            try:
                result = self.tools[tool_name]["handler"](request, arguments)
                return {
                    "type": "tool_result",
                    "success": True,
                    "result": result
                }
            except Exception as e:
                self.logger.error(f"Error calling tool '{tool_name}': {str(e)}")
                return {
                    "type": "tool_result",
                    "success": False,
                    "error": str(e)
                }
        
        elif request_type == "get_resources":
            return {
                "type": "resources",
                "resources": list(self.resources.values())
            }
        
        elif request_type == "get_prompts":
            return {
                "type": "prompts",
                "prompts": list(self.prompts.values())
            }
        
        else:
            return {
                "type": "error",
                "error": f"Unknown request type: {request_type}"
            }
    
    def close(self):
        """Close the server and release resources"""
        self.logger.info("Closing MCP server")
        # Clean up any resources
        return True

class McpAsyncServer:
    """
    Asynchronous MCP Server implementation.
    Provides MCP-compatible endpoints for the Multimodal RAG system.
    """
    
    def __init__(self, server_info=None, capabilities=None):
        """Initialize the Async MCP Server"""
        self.sync_server = McpSyncServer(server_info, capabilities)
        self.logger = self.sync_server.logger
    
    async def initialize_rag(self, document_paths=None, image_paths=None):
        """Initialize the RAG system asynchronously"""
        # This is a simple wrapper that calls the sync method
        # In a real implementation, this would be properly async
        return self.sync_server.initialize_rag(document_paths, image_paths)
    
    async def add_tool(self, tool_spec, handler):
        """Add a tool to the server asynchronously"""
        return self.sync_server.add_tool(tool_spec, handler)
    
    async def add_resource(self, resource_spec):
        """Add a resource to the server asynchronously"""
        return self.sync_server.add_resource(resource_spec)
    
    async def add_prompt(self, prompt_spec):
        """Add a prompt to the server asynchronously"""
        return self.sync_server.add_prompt(prompt_spec)
    
    async def handle_request(self, request):
        """Handle an MCP request asynchronously"""
        return self.sync_server.handle_request(request)
    
    async def close(self):
        """Close the server and release resources asynchronously"""
        return self.sync_server.close()
