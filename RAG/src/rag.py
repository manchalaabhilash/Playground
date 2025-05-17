import sys
sys.path.append('.')

from src.data_processing import DataLoader, DataSplitter
from src.embedding import EmbeddingModel
from src.vector_database import VectorDB
from src.llm_interaction import LocalLLM

class RAG:
    def __init__(self, pdf_paths, chunk_size=500, chunk_overlap=50):
        self.pdf_paths = pdf_paths if isinstance(pdf_paths, list) else [pdf_paths]
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents = []
        self.vector_db = None
        self.retriever = None
        self.llm = LocalLLM()
        
    def load_and_process_documents(self):
        # Load all PDFs
        for pdf_path in self.pdf_paths:
            loader = DataLoader(pdf_path)
            self.documents.extend(loader.load_data())
        
        # Split documents
        splitter = DataSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.chunks = splitter.split_data(self.documents)
        
        # Create embeddings and store in vector DB
        embedding_model = EmbeddingModel()
        self.vector_db = VectorDB(embedding_function=embedding_model.embeddings)
        self.vector_db.create_and_store(self.chunks)
        
        # Create retriever
        self.retriever = self.vector_db.get_retriever(search_kwargs={"k": 5})
        
        return len(self.chunks)
    
    def answer_question(self, question, use_rag=True):
        if not self.retriever and use_rag:
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