import sys
sys.path.append('.')

from src.data_processing import DocumentProcessor, ImageProcessor
from src.embedding import TextEmbeddingModel, ImageEmbeddingModel
from src.vector_database import MultimodalVectorDB
from src.llm_interaction import MultimodalLLM
from src.mcp_integration import McpRagOrchestrator
from src.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.llm_training import LLMTrainer
from src.optimized_llm import OptimizedLLM

class MultimodalRAG:
    def __init__(self, document_paths=None, image_paths=None, chunk_size=None, chunk_overlap=None, use_mcp=True, 
                 use_optimized_llm=False, quantization="none", use_onnx=False):
        """Initialize the Multimodal RAG system"""
        self.document_paths = document_paths or []
        self.image_paths = image_paths or []
        self.chunk_size = chunk_size or CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or CHUNK_OVERLAP
        self.use_mcp = use_mcp
        self.use_optimized_llm = use_optimized_llm
        
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
        
        # Add optimized LLM support
        if use_optimized_llm:
            self.optimized_llm = OptimizedLLM()
            self.optimized_llm.load_model(quantization=quantization, use_onnx=use_onnx)
        else:
            # Initialize regular LLM
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
        
        try:
            # Process documents
            self.text_chunks = self.document_processor.process_documents(self.document_paths)
            return len(self.text_chunks)
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise DocumentProcessingError(f"Failed to process documents: {str(e)}")
    
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
        """Answer a question using RAG and optimized LLM if enabled"""
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
            elif self.use_optimized_llm:
                # Format prompt for optimized LLM
                prompt = self._format_prompt_for_optimized_llm(question, relevant_texts, relevant_images)
                return self.optimized_llm.generate(prompt)
            else:
                # Generate response with multimodal context using regular LLM
                return self.llm.generate_response(question, relevant_texts, relevant_images)
        else:
            # Direct LLM response without context
            if self.use_optimized_llm:
                return self.optimized_llm.generate(question)
            else:
                return self.llm.generate_response(question)
    
    def _format_prompt_for_optimized_llm(self, question, relevant_texts, relevant_images):
        """Format prompt for optimized LLM that may not support multimodal inputs"""
        prompt = "You are a helpful assistant that answers questions based on provided information.\n\n"
        
        # Add text context
        if relevant_texts and len(relevant_texts) > 0:
            prompt += "Here is some relevant information from documents:\n\n"
            for i, doc in enumerate(relevant_texts):
                content = getattr(doc, "page_content", str(doc))
                prompt += f"Document {i+1}:\n{content}\n\n"
        
        # Add image descriptions (since optimized LLM might not support images directly)
        if relevant_images and len(relevant_images) > 0:
            prompt += f"Here are descriptions of {len(relevant_images)} relevant images:\n\n"
            for i, img in enumerate(relevant_images):
                caption = img.get("caption", f"Image {i+1}")
                prompt += f"Image {i+1}: {caption}\n"
        
        # Add the question
        prompt += f"\nQuestion: {question}\n\n"
        prompt += "Please provide a comprehensive answer based on the provided information."
        
        return prompt
    
    def train_custom_llm(self, data_path, model_name="meta-llama/Llama-2-7b-hf", 
                         output_dir="./trained_models", epochs=3, use_8bit=True):
        """Train a custom LLM on domain-specific data"""
        try:
            # Initialize trainer
            trainer = LLMTrainer(model_name=model_name, output_dir=output_dir)
            
            # Load model with quantization for training
            trainer.load_model(use_8bit=use_8bit)
            
            # Prepare dataset
            dataset = trainer.prepare_dataset(data_path)
            if dataset is None:
                return False, "Failed to prepare dataset"
            
            # Train model
            success = trainer.train(dataset, epochs=epochs)
            
            if success:
                # Load the trained model as our optimized LLM
                self.optimized_llm = OptimizedLLM(model_path=f"{output_dir}/final")
                self.optimized_llm.load_model()
                self.use_optimized_llm = True
                return True, f"Model trained and saved to {output_dir}/final"
            else:
                return False, "Training failed"
        except Exception as e:
            return False, f"Error during training: {str(e)}"
