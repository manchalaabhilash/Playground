import os
import sys
import unittest
import tempfile

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import DataLoader, DataSplitter
from langchain.schema import Document

class TestDataProcessing(unittest.TestCase):
    def test_data_loader_invalid_path(self):
        """Test DataLoader with invalid file path"""
        loader = DataLoader("nonexistent_file.pdf")
        documents = loader.load_data()
        self.assertEqual(len(documents), 0)
    
    def test_data_loader_invalid_format(self):
        """Test DataLoader with invalid file format"""
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(suffix=".txt") as temp_file:
            loader = DataLoader(temp_file.name)
            documents = loader.load_data()
            self.assertEqual(len(documents), 0)
    
    def test_data_splitter_empty_documents(self):
        """Test DataSplitter with empty documents list"""
        splitter = DataSplitter()
        chunks = splitter.split_data([])
        self.assertEqual(len(chunks), 0)
    
    def test_data_splitter(self):
        """Test DataSplitter with sample documents"""
        # Create sample documents
        documents = [
            Document(
                page_content="This is a test document with enough text to be split into multiple chunks. " * 10,
                metadata={"source": "test.pdf", "page": 1}
            )
        ]
        
        # Split documents
        splitter = DataSplitter(chunk_size=100, chunk_overlap=20)
        chunks = splitter.split_data(documents)
        
        # Check that documents were split
        self.assertGreater(len(chunks), 1)
        
        # Check that metadata was preserved
        for chunk in chunks:
            self.assertEqual(chunk.metadata["source"], "test.pdf")
            self.assertEqual(chunk.metadata["page"], 1)

if __name__ == "__main__":
    unittest.main()