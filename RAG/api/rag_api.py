from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import time
sys.path.append('.')

from src.data_processing import DataLoader, DataSplitter
from src.embedding import EmbeddingModel
from src.vector_database import VectorDB
from src.llm_interaction import LocalLLM
from src.config import API_HOST, API_PORT, DEBUG_MODE

app = Flask(__name__)
CORS(app)

# Global variables
vector_db = None
retriever = None
llm = None
pdf_directory = os.environ.get("PDF_DIRECTORY", "data/pdfs")

@app.route('/')
def home():
    return "ML Textbook RAG API - Upload PDFs and ask questions about machine learning concepts"

@app.route('/initialize', methods=['POST'])
def initialize():
    global vector_db, retriever, llm
    
    try:
        # Get PDF paths from the request
        data = request.json
        pdf_paths = data.get('pdf_paths', [])
        
        if not pdf_paths:
            return jsonify({"error": "No PDF paths provided"}), 400
        
        # Validate paths
        valid_paths = []
        for path in pdf_paths:
            if os.path.exists(path) and path.lower().endswith('.pdf'):
                valid_paths.append(path)
        
        if not valid_paths:
            return jsonify({"error": "No valid PDF paths found"}), 400
        
        # Initialize LLM first to ensure Ollama is ready
        llm = LocalLLM()
        
        # Load documents
        documents = []
        for pdf_path in valid_paths:
            loader = DataLoader(pdf_path)
            doc_chunks = loader.load_data()
            if doc_chunks:
                documents.extend(doc_chunks)
        
        if not documents:
            return jsonify({"error": "Could not extract text from PDFs"}), 400
        
        # Split documents
        splitter = DataSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_data(documents)
        
        if not chunks:
            return jsonify({"error": "Could not split documents into chunks"}), 400
        
        # Create embeddings and store in vector DB
        embedding_model = EmbeddingModel()
        vector_db = VectorDB(embedding_function=embedding_model.embeddings)
        vector_db.create_and_store(chunks)
        
        # Create retriever
        retriever = vector_db.get_retriever(search_kwargs={"k": 5})
        
        return jsonify({
            "status": "success",
            "message": f"Initialized RAG system with {len(chunks)} chunks from {len(valid_paths)} PDFs",
            "num_chunks": len(chunks),
            "num_pdfs": len(valid_paths)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask():
    global retriever, llm
    
    if not llm:
        # Initialize LLM if not already done
        llm = LocalLLM()
    
    try:
        data = request.json
        question = data.get('question')
        use_rag = data.get('use_rag', True)
        
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        if use_rag and not retriever:
            return jsonify({"error": "RAG system not initialized. Call /initialize first"}), 400
        
        if use_rag:
            # Get relevant documents
            relevant_docs = retriever.get_relevant_documents(question)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Create prompt with context
            prompt = f"""You are an AI assistant specialized in machine learning concepts.
Based on the following textbook excerpts, answer the question: '{question}'

Context:
{context}

Answer:"""
            
            # Generate response
            answer = llm.generate_response(prompt)
        else:
            # Direct LLM response without context
            answer = llm.generate_response(question)
        
        return jsonify({
            "status": "success",
            "question": question,
            "answer": answer,
            "used_rag": use_rag
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for the API"""
    return jsonify({
        "status": "healthy",
        "llm_initialized": llm is not None,
        "rag_initialized": retriever is not None
    })

if __name__ == '__main__':
    # Create PDF directory if it doesn't exist
    os.makedirs(pdf_directory, exist_ok=True)
    
    # Wait for Ollama to be ready in container environment
    if os.environ.get("DEPLOYMENT_ENV") == "production":
        time.sleep(10)
    
    # Start the Flask app
    app.run(debug=DEBUG_MODE, host=API_HOST, port=API_PORT)
