FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y tesseract-ocr build-essential

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p data/documents data/images data/vector_db && \
    chmod -R 777 data

# Expose port
EXPOSE 5000

# Start the API
CMD ["python", "api/multimodal_rag_api.py"]