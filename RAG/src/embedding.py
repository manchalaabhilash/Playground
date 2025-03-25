from langchain.embeddings import HuggingFaceEmbeddings

class EmbeddingModel:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)

    def get_embeddings(self, texts):
        return self.embeddings.embed_documents(texts)

    def embed_query(self, query):
        return self.embeddings.embed_query(query)