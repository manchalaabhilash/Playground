import sys
sys.path.append('.')

from src.data_processing import DocumentProcessor, ImageProcessor
from src.embedding import TextEmbeddingModel, ImageEmbeddingModel
from src.vector_database import MultimodalVectorDB
from src.llm_interaction import MultimodalLLM
from src.mcp_integration import McpRagOrchestrator
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

class MultimodalRAG:
    def __init__(self, document_paths=None, image_paths=None, chunk_size=None, chunk_overlap=None, use_mcp=True):
        """Initialize the Multimodal RAG system"""
        self.document_paths = document_paths or []
        self.image_paths = image_paths or []
        self.chunk_size = chunk_size or CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or CHUNK_OVERLAP
        self.use_mcp = use_mcp
        
        # Initialize processors
        self.document_processor = DocumentProcessor(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.image_processor = ImageProcessor()
        
        # Initialize embedding models
        self.text_embedding_model = TextEmbeddingModel()
        self.image_embedding_model = ImageEmbeddingModel()
        
        # Initialize vector database
        self.vector_db = None
        
        # Initialize retrievers
        self.text_retriever = None
        self.image_retriever = None
        
        # Initialize LLM
        self.llm = MultimodalLLM()
        
        # Initialize MCP orchestrator if enabled
        self.mcp_orchestrator = McpRagOrchestrator() if use_mcp else None
        
        # Storage for processed chunks
        self.text_chunks = []
        self.image_chunks = []
    
    def process_documents(self):
        """Process documents and create text chunks"""
        if not self.document_paths:
            return 0
        
        # Process documents
        self.text_chunks = self.document_processor.process_documents(self.document_paths)
        return len(self.text_chunks)
    
    def process_images(self):
        """Process images and create image chunks"""
        if not self.image_paths:
            return 0
        
        # Process images
        self.image_chunks = self.image_processor.process_images(self.image_paths)
        return len(self.image_chunks)
    
    def initialize_vector_db(self):
        """Initialize the vector database with processed chunks"""
        # Create vector database
        self.vector_db = MultimodalVectorDB(
            text_embedding_function=self.text_embedding_model.embeddings,
            image_embedding_function=self.image_embedding_model.embeddings
        )
        
        # Store text chunks
        if self.text_chunks:
            self.text_retriever = self.vector_db.create_text_retriever(self.text_chunks)
        
        # Store image chunks
        if self.image_chunks:
            self.image_retriever = self.vector_db.create_image_retriever(self.image_chunks)
    
    def answer_question(self, question, use_rag=True):
        """Answer a question using multimodal RAG"""
        if use_rag and (not self.text_retriever and not self.image_retriever):
            raise ValueError("Documents and images not processed. Call process_documents and/or process_images first.")
        
        if use_rag:
            # Get relevant text and images
            relevant_texts = []
            relevant_images = []
            
            if self.text_retriever:
                relevant_texts = self.text_retriever.get_relevant_documents(question)
            
            if self.image_retriever:
                relevant_images = self.image_retriever.get_relevant_documents(question)
            
            # Use MCP routing if enabled
            if self.use_mcp and self.mcp_orchestrator:
                return self.mcp_orchestrator.process_query(question, relevant_texts, relevant_images)
            else:
                # Generate response with multimodal context
                return self.llm.generate_response(question, relevant_texts, relevant_images)
        else:
            # Direct LLM response without context
            return self.llm.generate_response(question)
