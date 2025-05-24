import unittest
import os
import sys
import tempfile
from PIL import Image
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.multimodal_rag import MultimodalRAG

class TestMultimodalRAG(unittest.TestCase):
    def setUp(self):
        # Create temporary files for testing
        self.temp_txt = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
        self.temp_txt.write(b"This is a test document for the multimodal RAG system.")
        self.temp_txt.close()
        
        self.temp_img = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        self.temp_img.close()
        
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='blue')
        img.save(self.temp_img.name)
        
        # Initialize RAG with mock LLM
        self.rag = MultimodalRAG(
            document_paths=[self.temp_txt.name],
            image_paths=[self.temp_img.name]
        )
    
    def tearDown(self):
        # Clean up temporary files
        if os.path.exists(self.temp_txt.name):
            os.unlink(self.temp_txt.name)
        if os.path.exists(self.temp_img.name):
            os.unlink(self.temp_img.name)
    
    def test_process_documents(self):
        # Test document processing
        num_chunks = self.rag.process_documents()
        self.assertGreater(num_chunks, 0)
        self.assertGreater(len(self.rag.text_chunks), 0)
    
    def test_process_images(self):
        # Test image processing
        num_images = self.rag.process_images()
        self.assertEqual(num_images, 1)
        self.assertEqual(len(self.rag.image_chunks), 1)
    
    @patch('src.llm_interaction.MultimodalLLM.generate_response')
    def test_answer_question_with_rag(self, mock_generate):
        # Mock the LLM response
        mock_generate.return_value = "This is a test answer with RAG."
        
        # Process documents and images
        self.rag.process_documents()
        self.rag.process_images()
        
        # Mock vector database initialization
        self.rag.initialize_vector_db = MagicMock(return_value=True)
        self.rag.text_retriever = MagicMock()
        self.rag.text_retriever.get_relevant_documents = MagicMock(return_value=[])
        self.rag.image_retriever = MagicMock()
        self.rag.image_retriever.get_relevant_documents = MagicMock(return_value=[])
        
        # Test answering a question with RAG
        answer = self.rag.answer_question("Test question?", use_rag=True)
        self.assertEqual(answer, "This is a test answer with RAG.")
        
        # Verify the LLM was called with the right parameters
        mock_generate.assert_called_once()
    
    @patch('src.llm_interaction.MultimodalLLM.generate_response')
    def test_answer_question_without_rag(self, mock_generate):
        # Mock the LLM response
        mock_generate.return_value = "This is a test answer without RAG."
        
        # Test answering a question without RAG
        answer = self.rag.answer_question("Test question?", use_rag=False)
        self.assertEqual(answer, "This is a test answer without RAG.")
        
        # Verify the LLM was called with the right parameters
        mock_generate.assert_called_once_with("Test question?")

if __name__ == '__main__':
    unittest.main()