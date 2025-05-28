import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from PIL import Image
from io import BytesIO

from src.multimodal_retrieval import MultimodalRetriever, ImageRetriever, MultimodalFormatter

class TestMultimodalRetriever:
    def setup_method(self):
        self.text_retriever = MagicMock()
        self.image_retriever = MagicMock()
        self.retriever = MultimodalRetriever(
            text_retriever=self.text_retriever,
            image_retriever=self.image_retriever
        )
    
    def test_get_relevant_documents_text_only(self):
        # Setup
        self.text_retriever.get_relevant_documents.return_value = ["doc1", "doc2"]
        self.image_retriever.get_relevant_documents.return_value = []
        
        # Execute
        result = self.retriever.get_relevant_documents("text query", k=2)
        
        # Verify
        assert result == ["doc1", "doc2"]
        self.text_retriever.get_relevant_documents.assert_called_once()
    
    def test_get_relevant_documents_image_only(self):
        # Setup
        self.text_retriever.get_relevant_documents.return_value = []
        self.image_retriever.get_relevant_documents.return_value = ["img1", "img2"]
        
        # Execute
        result = self.retriever.get_relevant_documents("image query", k=2)
        
        # Verify
        assert result == ["img1", "img2"]
        self.image_retriever.get_relevant_documents.assert_called_once()
    
    def test_fusion_strategy_linear(self):
        # Setup
        self.text_retriever.get_relevant_documents.return_value = ["doc1", "doc2"]
        self.image_retriever.get_relevant_documents.return_value = ["img1", "img2"]
        self.retriever.fusion_strategy = "linear"
        
        # Mock the _linear_fusion method
        self.retriever._linear_fusion = MagicMock(return_value=["doc1", "img1"])
        
        # Execute
        result = self.retriever.get_relevant_documents("query", k=2)
        
        # Verify
        assert result == ["doc1", "img1"]
        self.retriever._linear_fusion.assert_called_once()

class TestImageRetriever:
    def setup_method(self):
        self.vector_store = MagicMock()
        self.embedding_model = MagicMock()
        self.retriever = ImageRetriever(
            vector_store=self.vector_store,
            embedding_model=self.embedding_model
        )
    
    def test_get_relevant_documents_vector_store(self):
        # Setup
        self.vector_store.similarity_search.return_value = ["img1", "img2"]
        
        # Execute
        result = self.retriever.get_relevant_documents("query", k=2)
        
        # Verify
        assert result == ["img1", "img2"]
        self.vector_store.similarity_search.assert_called_once_with("query", k=2, filter={"type": "image"})
    
    def test_get_relevant_documents_embedding_similarity(self):
        # Setup
        self.vector_store = None
        self.retriever.vector_store = None
        self.retriever.images = [{"id": 1}, {"id": 2}]
        self.retriever.image_embeddings = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
        
        # Mock embedding model and cosine similarity
        self.embedding_model.encode.return_value = np.array([0.2, 0.3])
        self.retriever._cosine_similarity = MagicMock(side_effect=[0.8, 0.6])
        
        # Execute
        result = self.retriever.get_relevant_documents("query", k=1)
        
        # Verify
        assert len(result) == 1
        assert result[0]["id"] == 1  # First image has higher similarity

class TestMultimodalFormatter:
    def setup_method(self):
        self.formatter = MultimodalFormatter()
    
    def test_format_text_document(self):
        # Setup
        doc = MagicMock()
        doc.page_content = "Test content"
        doc.metadata = {"source": "test.txt"}
        
        # Execute
        result = self.formatter._format_text_document(doc)
        
        # Verify
        assert "Test content" in result
        assert "source: test.txt" in result
    
    def test_format_image_document_base64(self):
        # Setup
        image = Image.new('RGB', (10, 10), color='red')
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        doc = {
            "image_data": img_byte_arr,
            "caption": "Test image",
            "ocr_text": "Sample text"
        }
        
        # Execute
        result = self.formatter._format_image_document(doc)
        
        # Verify
        assert "![Image](data:image/" in result
        assert "Caption: Test image" in result
        assert "Text in image: Sample text" in result