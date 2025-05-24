"""
API for the Multimodal RAG system.
Provides endpoints for initializing the system, uploading files, and asking questions.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import json
import logging
from werkzeug.utils import secure_filename

sys.path.append('.')
from src.multimodal_rag import MultimodalRAG
from src.mcp_server import McpSyncServer
from src.config import API_HOST, API_PORT, DEBUG_MODE, DOCUMENT_DIRECTORY, IMAGE_DIRECTORY

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("multimodal-rag-api")

# Global variables
rag_system = None
mcp_server = None

# Ensure directories exist
os.makedirs(DOCUMENT_DIRECTORY, exist_ok=True)
os.makedirs(IMAGE_DIRECTORY, exist_ok=True)

@app.route('/')
def home():
    return "Multimodal RAG API - Upload documents and images, then ask questions"

@app.route('/initialize', methods=['POST'])
def initialize():
    global rag_system
    
    data = request.json
    document_paths = data.get('document_paths', [])
    image_paths = data.get('image_paths', [])
    
    if not document_paths and not image_paths:
        return jsonify({"error": "No document or image paths provided"}), 400
    
    try:
        # Initialize RAG system
        rag_system = MultimodalRAG(
            document_paths=document_paths,
            image_paths=image_paths
        )
        
        # Process documents
        num_text_chunks = rag_system.process_documents()
        logger.info(f"Processed {num_text_chunks} text chunks from {len(document_paths)} documents")
        
        # Process images
        num_images = rag_system.process_images()
        logger.info(f"Processed {num_images} images")
        
        # Initialize vector database
        rag_system.initialize_vector_db()
        
        return jsonify({
            "status": "success",
            "num_documents": len(document_paths),
            "num_text_chunks": num_text_chunks,
            "num_images": num_images,
            "num_image_chunks": num_images
        })
    except Exception as e:
        logger.error(f"Error initializing RAG system: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_files():
    # Check if the post request has files
    if 'documents' not in request.files and 'images' not in request.files:
        return jsonify({"error": "No files provided"}), 400
    
    uploaded_files = []
    errors = []
    
    # Process document files
    if 'documents' in request.files:
        documents = request.files.getlist('documents')
        for file in documents:
            if file.filename:
                try:
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(DOCUMENT_DIRECTORY, filename)
                    file.save(file_path)
                    uploaded_files.append({
                        "name": filename,
                        "path": file_path,
                        "type": "document"
                    })
                except Exception as e:
                    errors.append({
                        "name": file.filename,
                        "error": str(e)
                    })
    
    # Process image files
    if 'images' in request.files:
        images = request.files.getlist('images')
        for file in images:
            if file.filename:
                try:
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(IMAGE_DIRECTORY, filename)
                    file.save(file_path)
                    uploaded_files.append({
                        "name": filename,
                        "path": file_path,
                        "type": "image"
                    })
                except Exception as e:
                    errors.append({
                        "name": file.filename,
                        "error": str(e)
                    })
    
    return jsonify({
        "status": "success",
        "uploaded_files": uploaded_files,
        "errors": errors
    })

@app.route('/ask', methods=['POST'])
def ask_question():
    global rag_system
    
    if not rag_system:
        return jsonify({"error": "RAG system not initialized"}), 400
    
    data = request.json
    question = data.get('question')
    use_rag = data.get('use_rag', True)
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    try:
        # Get answer from RAG system
        answer = rag_system.answer_question(question, use_rag=use_rag)
        
        return jsonify({
            "status": "success",
            "question": question,
            "answer": answer,
            "used_rag": use_rag
        })
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/mcp', methods=['POST'])
def mcp_endpoint():
    global mcp_server
    
    # Initialize MCP server if not already initialized
    if not mcp_server:
        mcp_server = McpSyncServer()
    
    # Get the MCP request
    request_data = request.json
    
    try:
        # Handle the request
        response = mcp_server.handle_request(request_data)
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error handling MCP request: {str(e)}")
        return jsonify({
            "type": "error",
            "error": str(e)
        }), 500

@app.route('/files/<path:filename>')
def get_file(filename):
    """Serve uploaded files"""
    # Check if the file is a document or image
    if os.path.exists(os.path.join(DOCUMENT_DIRECTORY, filename)):
        return send_from_directory(DOCUMENT_DIRECTORY, filename)
    elif os.path.exists(os.path.join(IMAGE_DIRECTORY, filename)):
        return send_from_directory(IMAGE_DIRECTORY, filename)
    else:
        return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    app.run(host=API_HOST, port=API_PORT, debug=DEBUG_MODE)
