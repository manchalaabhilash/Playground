import sys
sys.path.append('.')

from src.data_processing import DataLoader, DataSplitter
from src.embedding import EmbeddingModel
from src.vector_database import VectorDB
from src.llm_interaction import LocalLLM
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

class RAG:
    def __init__(self, pdf_paths, chunk_size=None, chunk_overlap=None):
        """Initialize the RAG system"""
        self.pdf_paths = pdf_paths if isinstance(pdf_paths, list) else [pdf_paths]
        self.chunk_size = chunk_size or CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or CHUNK_OVERLAP
        self.documents = []
        self.chunks = []
        self.vector_db = None
        self.retriever = None
        self.llm = LocalLLM()
        
    def load_and_process_documents(self):
        """Load and process documents, returning the number of chunks created"""
        # Load all PDFs
        for pdf_path in self.pdf_paths:
            loader = DataLoader(pdf_path)
            doc_chunks = loader.load_data()
            if doc_chunks:
                self.documents.extend(doc_chunks)
        
        if not self.documents:
            print("No documents were loaded. Check file paths and formats.")
            return 0
        
        # Split documents
        splitter = DataSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.chunks = splitter.split_data(self.documents)
        
        if not self.chunks:
            print("No chunks were created. Check document content.")
            return 0
        
        # Create embeddings and store in vector DB
        try:
            embedding_model = EmbeddingModel()
            self.vector_db = VectorDB(embedding_function=embedding_model.embeddings)
            success = self.vector_db.create_and_store(self.chunks)
            
            if success:
                # Create retriever
                self.retriever = self.vector_db.get_retriever(search_kwargs={"k": 5})
                return len(self.chunks)
            else:
                return 0
        except Exception as e:
            print(f"Error in document processing: {str(e)}")
            return 0
    
    def answer_question(self, question, use_rag=True):
        """Answer a question using RAG or direct LLM"""
        if use_rag and not self.retriever:
            raise ValueError("Documents not loaded. Call load_and_process_documents first.")
        
        if use_rag:
            # Get relevant documents
            relevant_docs = self.retriever.get_relevant_documents(question)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Create prompt with context
            prompt = f"""You are an AI assistant specialized in machine learning concepts.
Based on the following textbook excerpts, answer the question: '{question}'

Context:
{context}

Answer:"""
            
            # Generate response
            return self.llm.generate_response(prompt)
        else:
            # Direct LLM response without context
            return self.llm.generate_response(question)
