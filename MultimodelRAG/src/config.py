"""
Configuration settings for the Multimodal RAG system.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API settings
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", 5000))
DEBUG_MODE = os.environ.get("DEBUG_MODE", "True").lower() in ("true", "1", "t")

# LLM settings
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4-vision-preview")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")

# Vector DB settings
VECTOR_DB_PATH = os.environ.get("VECTOR_DB_PATH", "data/vector_db")

# Document processing settings
DOCUMENT_DIRECTORY = os.environ.get("DOCUMENT_DIRECTORY", "data/documents")
IMAGE_DIRECTORY = os.environ.get("IMAGE_DIRECTORY", "data/images")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 50))

# Image processing settings
IMAGE_EMBEDDING_MODEL = os.environ.get("IMAGE_EMBEDDING_MODEL", "clip-ViT-B-32")
OCR_ENABLED = os.environ.get("OCR_ENABLED", "True").lower() in ("true", "1", "t")

# MCP settings
USE_MCP = os.environ.get("USE_MCP", "True").lower() in ("true", "1", "t")
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8080")
MCP_SERVER_INFO = {
    "name": os.environ.get("MCP_SERVER_NAME", "multimodal-rag-server"),
    "version": os.environ.get("MCP_SERVER_VERSION", "1.0.0")
}
