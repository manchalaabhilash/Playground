from langchain.embeddings import HuggingFaceEmbeddings
import os

class EmbeddingModel:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the embedding model"""
        self.model_name = model_name
        
        # Initialize the embeddings model
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        except Exception as e:
            print(f"Error initializing embedding model: {str(e)}")
            raise

    def get_embeddings(self, texts):
        return self.embeddings.embed_documents(texts)

    def embed_query(self, query):
        return self.embeddings.embed_query(query)
