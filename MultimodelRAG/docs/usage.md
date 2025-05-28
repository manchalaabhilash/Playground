# MultimodalRAG Usage Guide

This guide provides instructions for using the MultimodalRAG system, including setup, configuration, and usage examples.

## Installation

### Using Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MultimodelRAG.git
   cd MultimodelRAG
   ```

2. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

3. Start the services:
   ```bash
   docker-compose up -d
   ```

4. Access the UI at http://localhost:8501

### Manual Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MultimodelRAG.git
   cd MultimodelRAG
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

4. Start the API:
   ```bash
   python api/multimodal_rag_api.py
   ```

5. Start the UI (in a separate terminal):
   ```bash
   streamlit run ui/multimodal_rag_web_ui.py
   ```

## Configuration

### Environment Variables

- `LLM_API_KEY`: API key for the LLM provider (OpenAI, Anthropic)
- `LLM_MODEL`: Model to use (e.g., "gpt-4", "claude-3-opus")
- `EMBEDDING_MODEL`: Text embedding model to use
- `IMAGE_EMBEDDING_MODEL`: Image embedding model to use
- `VECTOR_DB_PATH`: Path to store vector database
- `CHUNK_SIZE`: Size of text chunks for processing
- `CHUNK_OVERLAP`: Overlap between text chunks
- `FUSION_STRATEGY`: Strategy for combining text and image results ("linear", "max", "weighted")
- `TEXT_WEIGHT`: Weight for text results in fusion (0.0-1.0)
- `IMAGE_WEIGHT`: Weight for image results in fusion (0.0-1.0)

### Configuration File

For more advanced configuration, edit `config.py` to customize:
- Chunking strategies
- Embedding models
- Retrieval parameters
- Reranking settings
- LLM prompt templates

## Usage Examples

### Adding Documents

#### Using the UI

1. Navigate to the "Upload" tab
2. Select files to upload (PDF, TXT, DOCX, JPG, PNG)
3. Click "Upload Files"
4. Wait for processing to complete

#### Using the API

```python
import requests
import base64

# Text document
with open('document.pdf', 'rb') as f:
    file_content = base64.b64encode(f.read()).decode('utf-8')

response = requests.post(
    'http://localhost:5000/upload',
    json={
        'file_name': 'document.pdf',
        'file_content': file_content,
        'file_type': 'pdf'
    }
)
print(response.json())

# Image
with open('image.jpg', 'rb') as f:
    file_content = base64.b64encode(f.read()).decode('utf-8')

response = requests.post(
    'http://localhost:5000/upload',
    json={
        'file_name': 'image.jpg',
        'file_content': file_content,
        'file_type': 'image'
    }
)
print(response.json())
```

### Querying

#### Using the UI

1. Navigate to the "Query" tab
2. Enter your question in the text box
3. Click "Submit"
4. View the response and retrieved context

#### Using the API

```python
import requests

response = requests.post(
    'http://localhost:5000/query',
    json={
        'query': 'What information can you find about climate change?',
        'use_rag': True,
        'modality': 'auto',
        'top_k': 5
    }
)
print(response.json())
```

## Advanced Features

### Customizing Retrieval

You can customize the retrieval process by specifying parameters:

```python
response = requests.post(
    'http://localhost:5000/query',
    json={
        'query': 'Describe this image',
        'use_rag': True,
        'modality': 'image',  # Force image retrieval
        'top_k': 3,
        'rerank': True,
        'fusion_strategy': 'weighted',
        'text_weight': 0.3,
        'image_weight': 0.7
    }
)
```

### Using MCP Routing

To enable Model Control Protocol routing:

```python
response = requests.post(
    'http://localhost:5000/query',
    json={
        'query': 'What's in this document?',
        'use_rag': True,
        'use_mcp': True,
        'modality': 'auto',
        'top_k': 5
    }
)
```

## Troubleshooting

### Common Issues

1. **API Connection Error**
   - Ensure the API service is running
   - Check the API logs for errors: `docker-compose logs api`

2. **Upload Failures**
   - Check file size limits (default: 10MB)
   - Ensure file format is supported
   - Check disk space availability

3. **Retrieval Issues**
   - Verify documents were properly indexed
   - Check embedding model configuration
   - Increase `top_k` parameter for broader results

4. **LLM Integration Issues**
   - Verify API key is correct
   - Check LLM provider status
   - Review API rate limits

### Logs

- API logs: `docker-compose logs api`
- UI logs: `docker-compose logs ui`
- Application logs: Check `logs/` directory