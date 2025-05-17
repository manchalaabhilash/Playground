FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y curl build-essential && \
    curl -fsSL https://ollama.com/install.sh | sh

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p data/pdfs data/vector_db && \
    chmod -R 777 data

# Expose ports
EXPOSE 8501 11434

# Start Ollama and the Streamlit app
CMD ["sh", "-c", "ollama serve & ollama pull llama2 & streamlit run ui/rag_web_ui.py --server.port=8501 --server.address=0.0.0.0"]
