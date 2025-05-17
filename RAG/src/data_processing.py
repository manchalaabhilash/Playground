from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        """Load data from a file and return a list of documents"""
        try:
            if not os.path.exists(self.file_path):
                print(f"File not found: {self.file_path}")
                return []
                
            if self.file_path.lower().endswith(".pdf"):
                loader = PyPDFLoader(self.file_path)
            elif self.file_path.lower().endswith(".txt"):
                loader = TextLoader(self.file_path)
            else:
                print(f"Unsupported file type: {self.file_path}")
                return []
                
            return loader.load()
        except Exception as e:
            print(f"Error loading file {self.file_path}: {str(e)}")
            return []

class DataSplitter:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )

    def split_data(self, documents):
        """Split documents into chunks"""
        if not documents:
            return []
            
        try:
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            print(f"Error splitting documents: {str(e)}")
            return []
