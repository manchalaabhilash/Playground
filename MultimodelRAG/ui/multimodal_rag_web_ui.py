import streamlit as st
import requests
import os
import tempfile
import time
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Multimodal RAG",
    page_icon="🖼️",
    layout="wide"
)

# API URL - use local URL for development, container URL for production
if os.environ.get("DEPLOYMENT_ENV") == "production":
    API_URL = "http://api:5000"  # Within the container
else:
    API_URL = st.sidebar.text_input(
        "API URL",
        value="http://localhost:5000",
        help="Enter the URL of your Multimodal RAG API"
    )

# Title and description
st.title("🖼️ Multimodal RAG System")
st.markdown("""
This application allows you to ask questions about your documents and images using a Retrieval-Augmented Generation (RAG) system.
Upload your content, initialize the system, and start asking questions!
""")

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["Upload & Initialize", "Ask Questions"])

# Tab 1: Upload and Initialize
with tab1:
    st.header("Upload Content")
    
    # Document upload
    st.subheader("Upload Documents")
    uploaded_documents = st.file_uploader(
        "Upload PDF or text documents",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )
    
    # Image upload
    st.subheader("Upload Images")
    uploaded_images = st.file_uploader(
        "Upload images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    # Save uploaded files
    document_paths = []
    image_paths = []
    
    if uploaded_documents or uploaded_images:
        with st.spinner("Saving uploaded files..."):
            # Save documents
            for doc in uploaded_documents:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{doc.name.split('.')[-1]}") as tmp:
                    tmp.write(doc.getvalue())
                    document_paths.append(tmp.name)
            
            # Save images
            for img in uploaded_images:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{img.name.split('.')[-1]}") as tmp:
                    tmp.write(img.getvalue())
                    image_paths.append(tmp.name)
            
            # Store paths in session state
            st.session_state.document_paths = document_paths
            st.session_state.image_paths = image_paths
            
            st.success(f"Saved {len(document_paths)} documents and {len(image_paths)} images")
    
    # Initialize button
    if st.button("Initialize RAG System") and (document_paths or image_paths or 
                                              hasattr(st.session_state, 'document_paths') or 
                                              hasattr(st.session_state, 'image_paths')):
        # Use stored paths if available
        if not document_paths and hasattr(st.session_state, 'document_paths'):
            document_paths = st.session_state.document_paths
        
        if not image_paths and hasattr(st.session_state, 'image_paths'):
            image_paths = st.session_state.image_paths
        
        # Initialize RAG system
        with st.spinner("Initializing RAG system... This may take a few minutes for large files."):
            response = requests.post(
                f"{API_URL}/initialize",
                json={"document_paths": document_paths, "image_paths": image_paths}
            )
            
            if response.status_code == 200:
                result = response.json()
                st.success(f"RAG system initialized with {result['num_text_chunks']} text chunks from {result['num_documents']} documents and {result['num_image_chunks']} images")
                # Store initialization status
                st.session_state.initialized = True
            else:
                st.error(f"Error initializing RAG system: {response.text}")

# Tab 2: Ask Questions
with tab2:
    st.header("Ask Questions")
    
    # Check if system is initialized
    if not hasattr(st.session_state, 'initialized'):
        st.warning("Please upload content and initialize the RAG system first.")
    
    # Question input
    question = st.text_input("Enter your question")
    
    # RAG toggle
    use_rag = st.checkbox("Use RAG", value=True, help="If unchecked, the question will be sent directly to the LLM without retrieving context")
    
    # Ask button
    if question and st.button("Ask"):
        # Check if RAG system is initialized
        if not hasattr(st.session_state, 'initialized') and use_rag:
            st.warning("Please upload content and initialize the RAG system first.")
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

# Sidebar with additional information
with st.sidebar:
    st.subheader("About")
    st.markdown("""
    This Multimodal RAG system can process both text documents and images to answer your questions.
    
    **Features:**
    - Upload PDFs, text files, and images
    - Process and index content
    - Ask questions about your content
    - Get answers with relevant context
    
    **How it works:**
    1. Documents and images are processed and embedded
    2. When you ask a question, the system retrieves relevant content
    3. The LLM generates an answer based on the retrieved context
    """)
    
    # Add a test connection button
    if st.button("Test API Connection"):
        try:
            response = requests.get(f"{API_URL}/")
            st.success(f"Connection successful! Response: {response.text}")
        except Exception as e:
            st.error(f"Connection failed: {str(e)}")