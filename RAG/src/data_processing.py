from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        if self.file_path.lower().endswith(".pdf"):
            loader = PyPDFLoader(self.file_path)
        elif self.file_path.lower().endswith(".txt"):
            loader = TextLoader(self.file_path)
        else:
            raise ValueError("Unsupported file type. Please provide a .pdf or .txt file.")
        return loader.load()

class DataSplitter:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

    def split_data(self, documents):
        return self.text_splitter.split_documents(documents)