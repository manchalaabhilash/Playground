import os
import shutil

# Define the directories to create
directories = [
    "src",
    "api",
    "api/routes",
    "ui",
    "ui/components",
    "scripts",
    "notebooks",
    "tests",
    "data",
    "data/pdfs",
    "data/vector_db"
]

# Create directories
for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"Created directory: {directory}")

# Define file movements (source -> destination)
file_movements = [
    # Core source files
    ("data_processing.py", "src/data_processing.py"),
    ("embedding.py", "src/embedding.py"),
    ("vector_db.py", "src/vector_database.py"),
    ("retreiver.py", "src/retriever.py"),
    ("llm_interaction.py", "src/llm_interaction.py"),
    ("rag.py", "src/rag.py"),
    
    # API files
    ("rag_api.py", "api/rag_api.py"),
    
    # UI files
    ("rag_web_ui.py", "ui/rag_web_ui.py"),
    
    # Scripts
    ("rag_client.py", "scripts/rag_client.py"),
    ("test_rag.py", "scripts/test_rag.py"),
    
    # Notebooks
    ("rag.ipynb", "notebooks/rag_demo.ipynb"),
]

# Move files
for source, destination in file_movements:
    if os.path.exists(source):
        shutil.copy2(source, destination)
        print(f"Moved: {source} -> {destination}")
    else:
        print(f"Warning: Source file not found: {source}")

# Create empty __init__.py files for Python packages
init_files = [
    "src/__init__.py",
    "api/__init__.py",
    "api/routes/__init__.py",
    "ui/__init__.py",
    "ui/components/__init__.py",
    "tests/__init__.py"
]

for init_file in init_files:
    with open(init_file, "w") as f:
        f.write("# This file makes the directory a Python package\n")
    print(f"Created: {init_file}")

# Create a simple config.py
with open("src/config.py", "w") as f:
    f.write("""# Configuration settings
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
""")
print("Created: src/config.py")

# Update the main README.md
with open("README.md", "w") as f:
    f.write("""# ML Textbook RAG System

A Retrieval-Augmented Generation (RAG) system for answering questions about machine learning concepts using textbook PDFs.

## Features

- Upload and process machine learning textbook PDFs
- Split documents into chunks and create embeddings
- Store embeddings in a vector database
- Retrieve relevant context for questions
- Generate answers using Ollama LLM with retrieved context
- Web UI for easy interaction
- API for programmatic access

## Project Structure

```
RAG/
├── src/                 # Core source code
├── api/                 # API-related code
├── ui/                  # UI-related code
├── scripts/             # Utility scripts
├── notebooks/           # Jupyter notebooks
├── tests/               # Unit tests
└── data/                # Data directory
```

## Getting Started

### Using Docker

1. Clone the repository
2. Run `docker-compose up`
3. Access the web UI at http://localhost:8501

### Manual Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Start Ollama: `ollama run llama2`
4. Start the API: `python api/rag_api.py`
5. Start the UI: `streamlit run ui/rag_web_ui.py`

## Usage

1. Upload your machine learning textbook PDFs
2. Initialize the RAG system
3. Ask questions about machine learning concepts
4. Get answers based on your textbooks!
""")
print("Updated: README.md")

# Create a .env file
with open(".env", "w") as f:
    f.write("""# Environment variables
API_HOST=0.0.0.0
API_PORT=5000
DEBUG_MODE=True

OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama2

VECTOR_DB_PATH=data/vector_db
PDF_DIRECTORY=data/pdfs

CHUNK_SIZE=500
CHUNK_OVERLAP=50
""")
print("Created: .env")

print("\nReorganization complete!")
