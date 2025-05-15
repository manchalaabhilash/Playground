from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

class DocumentRetriever:
    def __init__(self, persist_directory="chroma_db", embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", search_type="similarity", search_kwargs={"k": 3}):
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model_name
        self.search_type = search_type
        self.search_kwargs = search_kwargs
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        self.vectordb = self._load_vector_database()
        self.retriever = self.vectordb.as_retriever(search_type=self.search_type, search_kwargs=self.search_kwargs)

    def _load_vector_database(self):
        try:
            vectordb = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            print(f"Loaded existing vector database from: {self.persist_directory}")
            return vectordb
        except Exception as e:
            print(f"Error loading vector database: {e}. Ensure the database exists at {self.persist_directory} or run data preparation first.")
            return None

    def get_relevant_documents(self, query):
        if self.retriever:
            return self.retriever.get_relevant_documents(query)
        else:
            print("Retriever not initialized.")
            return []