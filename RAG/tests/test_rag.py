import os
import sys
import unittest
import tempfile

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag import RAG
from langchain.schema import Document

class TestRAG(unittest.TestCase):
    def test_rag_initialization(self):
        """Test RAG initialization"""
        pdf_paths = ["test.pdf"]
        rag = RAG(pdf_paths)
        
        # Check that attributes are set correctly
        self.assertEqual(rag.pdf_paths, pdf_paths)
        self.assertEqual(rag.chunk_size, 500)
        self.assertEqual(rag.chunk_overlap, 50)
        self.assertEqual(rag.documents, [])
        self.assertIsNone(rag.vector_db)
        self.assertIsNone(rag.retriever)
        self.assertIsNotNone(rag.llm)
    
    def test_answer_question_without_initialization(self):
        """Test answering a question without initializing the RAG system"""
        rag = RAG(["test.pdf"])
        
        # Direct LLM should work without initialization
        try:
            answer = rag.answer_question("What is machine learning?", use_rag=False)
            self.assertIsInstance(answer, str)
            self.assertGreater(len(answer), 0)
        except Exception as e:
            self.fail(f"Direct LLM answer raised exception: {e}")
        
        # RAG should raise an error without initialization
        with self.assertRaises(ValueError):
            rag.answer_question("What is machine learning?", use_rag=True)
    
    def test_load_and_process_documents_invalid_paths(self):
        """Test loading and processing documents with invalid paths"""
        rag = RAG(["nonexistent_file.pdf"])
        num_chunks = rag.load_and_process_documents()
        
        # Should return 0 chunks for invalid paths
        self.assertEqual(num_chunks, 0)
        self.assertEqual(rag.documents, [])
        self.assertIsNone(rag.vector_db)
        self.assertIsNone(rag.retriever)

if __name__ == "__main__":
    unittest.main()