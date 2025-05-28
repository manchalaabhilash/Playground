import pytest
from unittest.mock import MagicMock, patch
import os

from src.llm_interaction import MultimodalLLM

class TestMultimodalLLM:
    def setup_method(self):
        self.llm = MultimodalLLM(model_name="test-model", api_key="test-key")
    
    @patch('requests.post')
    def test_call_openai_api(self, mock_post):
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_post.return_value = mock_response
        
        # Execute
        result = self.llm._call_openai_api("Test prompt", [])
        
        # Verify
        assert result == "Test response"
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_call_anthropic_api(self, mock_post):
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": [{"text": "Test response"}]
        }
        mock_post.return_value = mock_response
        
        # Execute
        result = self.llm._call_anthropic_api("Test prompt", [])
        
        # Verify
        assert result == "Test response"
        mock_post.assert_called_once()
    
    def test_prepare_prompt_no_context(self):
        # Execute
        result = self.llm._prepare_prompt("Test question")
        
        # Verify
        assert "Test question" in result
        assert "No relevant information found" in result
    
    def test_prepare_prompt_with_context(self):
        # Setup
        text_docs = ["Document 1", "Document 2"]
        image_docs = [{"caption": "Image 1"}]
        
        # Execute
        result = self.llm._prepare_prompt("Test question", text_docs, image_docs)
        
        # Verify
        assert "Test question" in result
        assert "Document 1" in result
        assert "Document 2" in result
        assert "Image 1" in result