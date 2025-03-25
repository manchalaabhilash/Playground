from langchain.vectorstores import Chroma

class VectorDB:
    def __init__(self, persist_directory="chroma_db", embedding_function=None):
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        self.vectordb = None

    def create_and_store(self, documents):
        if self.embedding_function is None:
            raise ValueError("Embedding function must be provided.")
        self.vectordb = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_function,
            persist_directory=self.persist_directory
        )
        self.vectordb.persist()
        print(f"Vector database created and persisted in: {self.persist_directory}")

    def load_existing(self):
        if self.embedding_function is None:
            raise ValueError("Embedding function must be provided.")
        self.vectordb = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_function
        )
        print(f"Loaded existing vector database from: {self.persist_directory}")

    def get_retriever(self, search_type="similarity", search_kwargs={"k": 3}):
        if self.vectordb is None:
            raise ValueError("Vector database not initialized. Call create_and_store or load_existing first.")
        return self.vectordb.as_retriever(search_type=search_type, search_kwargs=search_kwargs)