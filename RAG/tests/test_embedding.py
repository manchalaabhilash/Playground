import os
import sys
import unittest

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedding import EmbeddingModel

class TestEmbedding(unittest.TestCase):
    def test_embedding_model_initialization(self):
        """Test EmbeddingModel initialization"""
        try:
            model = EmbeddingModel()
            self.assertIsNotNone(model.embeddings)
        except Exception as e:
            self.fail(f"EmbeddingModel initialization raised exception: {e}")
    
    def test_embedding_generation(self):
        """Test embedding generation"""
        model = EmbeddingModel()
        
        # Test with a single text
        text = "This is a test sentence for embedding."
        embedding = model.embeddings.embed_query(text)
        
        # Check that embedding is a list of floats
        self.assertIsInstance(embedding, list)
        self.assertGreater(len(embedding), 0)
        self.assertIsInstance(embedding[0], float)
        
        # Test with multiple texts
        texts = ["First test sentence.", "Second test sentence."]
        embeddings = model.embeddings.embed_documents(texts)
        
        # Check that embeddings is a list of lists of floats
        self.assertIsInstance(embeddings, list)
        self.assertEqual(len(embeddings), 2)
        self.assertIsInstance(embeddings[0], list)
        self.assertGreater(len(embeddings[0]), 0)
        self.assertIsInstance(embeddings[0][0], float)

if __name__ == "__main__":
    unittest.main()