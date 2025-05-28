from langchain.vectorstores import Chroma
import os
import numpy as np
from src.config import VECTOR_DB_PATH

class MultimodalVectorDB:
    def __init__(self, text_embedding_function, image_embedding_function, persist_directory=None, cache_enabled=True):
        self.text_embedding_function = text_embedding_function
        self.image_embedding_function = image_embedding_function
        self.persist_directory = persist_directory or VECTOR_DB_PATH
        self.cache_enabled = cache_enabled
        self.query_cache = {}
        self.cache_ttl = 3600  # 1 hour
        
        # Create separate collections for text and images
        self.text_db_path = os.path.join(self.persist_directory, "text")
        self.image_db_path = os.path.join(self.persist_directory, "images")
        
        # Initialize empty vector stores
        self.text_vectorstore = None
        self.image_vectorstore = None
    
    def add_text(self, text_chunks):
        """Add text documents to the vector database"""
        os.makedirs(self.text_db_path, exist_ok=True)
        
        self.text_vectorstore = Chroma.from_documents(
            documents=text_chunks,
            embedding=self.text_embedding_function,
            persist_directory=self.text_db_path
        )
        
        self.text_vectorstore.persist()
        return len(text_chunks)
    
    def add_images(self, image_chunks):
        """Add images to the vector database"""
        os.makedirs(self.image_db_path, exist_ok=True)
        
        # Convert image chunks to document format
        from langchain.schema import Document
        
        image_documents = []
        for chunk in image_chunks:
            doc = Document(
                page_content=chunk.get("ocr_text", ""),
                metadata={
                    "image_path": chunk["image_path"],
                    "image_filename": chunk["image_filename"],
                    "width": chunk["width"],
                    "height": chunk["height"],
                    "format": chunk["format"]
                }
            )
            image_documents.append(doc)
        
        self.image_vectorstore = Chroma.from_documents(
            documents=image_documents,
            embedding=self.image_embedding_function,
            persist_directory=self.image_db_path
        )
        
        self.image_vectorstore.persist()
        return len(image_documents)
    
    def get_text_retriever(self, search_type="similarity", search_kwargs=None):
        """Get retriever for text documents"""
        if not self.text_vectorstore:
            # Try to load from disk if exists
            if os.path.exists(self.text_db_path):
                self.text_vectorstore = Chroma(
                    persist_directory=self.text_db_path,
                    embedding_function=self.text_embedding_function
                )
            else:
                return None
        
        search_kwargs = search_kwargs or {"k": 3}
        return self.text_vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
    
    def get_image_retriever(self, search_type="similarity", search_kwargs=None):
        """Get retriever for images"""
        if not self.image_vectorstore:
            # Try to load from disk if exists
            if os.path.exists(self.image_db_path):
                self.image_vectorstore = Chroma(
                    persist_directory=self.image_db_path,
                    embedding_function=self.image_embedding_function
                )
            else:
                return None
        
        search_kwargs = search_kwargs or {"k": 2}
        return self.image_vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
    
    def get_similar_documents(self, query_vector, collection_name, k=5):
        # Check cache first
        cache_key = f"{hash(str(query_vector))}-{collection_name}-{k}"
        if self.cache_enabled and cache_key in self.query_cache:
            return self.query_cache[cache_key]
            
        # Perform actual retrieval
        results = self._perform_vector_search(query_vector, collection_name, k)
        
        # Cache results
        if self.cache_enabled:
            self.query_cache[cache_key] = results
            
        return results
