# Configuration settings
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 5000))
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

# LLM settings
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")

# Vector DB settings
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "data/vector_db")

# Document processing settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
PDF_DIRECTORY = os.getenv("PDF_DIRECTORY", "data/pdfs")
