import streamlit as st
import requests
import os
import tempfile
import time

# Set page configuration
st.set_page_config(
    page_title="ML Textbook RAG",
    page_icon="📚",
    layout="wide"
)

# API URL - use local URL for development, container URL for production
if os.environ.get("DEPLOYMENT_ENV") == "production":
    API_URL = "http://localhost:5000"  # Within the container
else:
    API_URL = st.sidebar.text_input(
        "API URL",
        value="http://localhost:5000",
        help="Enter the URL of your RAG API"
    )

# Title and description
st.title("📚 ML Textbook RAG")
st.markdown("""
This app allows you to ask questions about machine learning concepts using your textbook PDFs.
Upload your PDFs, initialize the RAG system, and start asking questions!
""")

# File uploader
uploaded_files = st.file_uploader("Upload ML textbook PDFs", type="pdf", accept_multiple_files=True)

# Initialize button
if uploaded_files and st.button("Initialize RAG System"):
    # Save uploaded files to temporary directory
    temp_dir = os.path.join("data", "pdfs")
    os.makedirs(temp_dir, exist_ok=True)
    pdf_paths = []
    
    # Progress bar for file upload
    upload_progress = st.progress(0)
    for i, uploaded_file in enumerate(uploaded_files):
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        pdf_paths.append(file_path)
        upload_progress.progress((i + 1) / len(uploaded_files))
    
    # Initialize RAG system
    with st.spinner("Initializing RAG system... This may take a few minutes for large PDFs."):
        # Wait for Ollama to be ready (important for container startup)
        time.sleep(5)
        
        response = requests.post(
            f"{API_URL}/initialize",
            json={"pdf_paths": pdf_paths}
        )
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"RAG system initialized with {result['num_chunks']} chunks from {result['num_pdfs']} PDFs")
            # Store the PDF paths in session state for future reference
            st.session_state.pdf_paths = pdf_paths
        else:
            st.error(f"Error initializing RAG system: {response.text}")

# Question input
st.subheader("Ask a Question")
question = st.text_input("Enter your question about machine learning concepts")
use_rag = st.checkbox("Use RAG (Retrieval-Augmented Generation)", value=True)

# Ask button
if question and st.button("Ask"):
    # Check if RAG system is initialized
    if not hasattr(st.session_state, 'pdf_paths') and use_rag:
        st.warning("Please upload PDFs and initialize the RAG system first.")
    else:
        with st.spinner("Generating answer..."):
            response = requests.post(
                f"{API_URL}/ask",
                json={"question": question, "use_rag": use_rag}
            )
            
            if response.status_code == 200:
                result = response.json()
                st.subheader("Answer")
                st.write(result["answer"])
                
                # Show whether RAG was used
                st.info(f"RAG was {'used' if result['used_rag'] else 'not used'} to generate this answer")
            else:
                st.error(f"Error: {response.text}")

# Add information about the system
with st.sidebar:
    st.subheader("About")
    st.markdown("""
    This app uses Retrieval-Augmented Generation (RAG) to answer questions about machine learning concepts.
    
    The system:
    1. Loads and processes your ML textbook PDFs
    2. Splits them into chunks
    3. Creates embeddings and stores them in a vector database
    4. Retrieves relevant chunks for your question
    5. Generates an answer using Ollama with the retrieved context
    
    You can toggle RAG on/off to compare answers with and without context retrieval.
    """)
    
    # Show currently loaded PDFs
    if hasattr(st.session_state, 'pdf_paths') and st.session_state.pdf_paths:
        st.subheader("Loaded PDFs")
        for pdf_path in st.session_state.pdf_paths:
            st.write(f"- {os.path.basename(pdf_path)}")
