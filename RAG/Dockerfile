FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p data/pdfs data/vector_db && \
    chmod -R 777 data

# Expose port
EXPOSE 5000

# Start the API
CMD ["python", "api/rag_api.py"]
