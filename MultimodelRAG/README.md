# Multimodal RAG System

A Retrieval-Augmented Generation (RAG) system for answering questions using both text and image data.

## Features

- Process and index both text documents (PDFs, TXT) and images
- Extract text from images using OCR
- Extract image features using vision models
- Store multimodal embeddings in a vector database
- Retrieve relevant context (text and images) for questions
- Generate answers using multimodal LLM capabilities
- Web UI for easy interaction
- API for programmatic access

## Project Structure

```
MultimodelRAG/
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
3. Start the API: `python api/multimodal_rag_api.py`
4. Start the UI: `streamlit run ui/multimodal_rag_web_ui.py`

## Usage

1. Upload your documents and images
2. Initialize the multimodal RAG system
3. Ask questions about your content
4. Get answers with relevant text and image context