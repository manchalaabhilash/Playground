# ML Textbook RAG System

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
