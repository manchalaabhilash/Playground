from langchain.vectorstores import Chroma
import os
from src.config import VECTOR_DB_PATH

class VectorDB:
    def __init__(self, embedding_function, persist_directory=None):
        """Initialize the vector database"""
        self.embedding_function = embedding_function
        self.persist_directory = persist_directory or VECTOR_DB_PATH
        self.db = None
        
        # Create directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
    
    def create_and_store(self, documents):
        """Create a vector database from documents and store it"""
        if not documents:
            print("No documents provided to store in vector database")
            return False
            
        try:
            self.db = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_function,
                persist_directory=self.persist_directory
            )
            self.db.persist()
            print(f"Vector database created and stored at {self.persist_directory}")
            return True
        except Exception as e:
            print(f"Error creating vector database: {str(e)}")
            return False
    
    def get_retriever(self, search_type="similarity", search_kwargs=None):
        """Get a retriever from the vector database"""
        if not self.db:
            print("Vector database not initialized")
            return None
            
        if search_kwargs is None:
            search_kwargs = {"k": 4}
            
        try:
            return self.db.as_retriever(
                search_type=search_type,
                search_kwargs=search_kwargs
            )
        except Exception as e:
            print(f"Error creating retriever: {str(e)}")
            return None
