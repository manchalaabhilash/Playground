from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
sys.path.append('.')

from src.data_processing import DataLoader, DataSplitter
from src.embedding import EmbeddingModel
from src.vector_database import VectorDB
from src.llm_interaction import LocalLLM

app = Flask(__name__)
CORS(app)

# Global variables
vector_db = None
retriever = None
llm = None
pdf_directory = "pdfs"  # Directory to store uploaded PDFs

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
        
        # Load documents
        documents = []
        for pdf_path in valid_paths:
            loader = DataLoader(pdf_path)
            documents.extend(loader.load_data())
        
        # Split documents
        splitter = DataSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_data(documents)
        
        # Create embeddings and store in vector DB
        embedding_model = EmbeddingModel()
        vector_db = VectorDB(embedding_function=embedding_model.embeddings)
        vector_db.create_and_store(chunks)
        
        # Create retriever
        retriever = vector_db.get_retriever(search_kwargs={"k": 5})
        
        # Initialize LLM
        llm = LocalLLM()
        
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
    
    if not retriever or not llm:
        return jsonify({"error": "RAG system not initialized. Call /initialize first"}), 400
    
    try:
        data = request.json
        question = data.get('question')
        use_rag = data.get('use_rag', True)
        
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
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

if __name__ == '__main__':
    # Create PDF directory if it doesn't exist
    os.makedirs(pdf_directory, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)