"""
Semantic chunking implementation for advanced RAG.
Creates semantically meaningful chunks rather than arbitrary splits.
"""

import re
import nltk
from typing import List, Dict, Any, Optional
from nltk.tokenize import sent_tokenize

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class SemanticChunker:
    """
    Creates semantically meaningful chunks from text.
    Uses sentence boundaries and semantic similarity to create better chunks.
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, embedding_model=None):
        """Initialize the semantic chunker"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
    
    def create_chunks(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Create semantically meaningful chunks from text"""
        # First split by paragraphs
        paragraphs = self._split_paragraphs(text)
        
        # Then split paragraphs into sentences
        sentences = []
        for para in paragraphs:
            sentences.extend(self._split_sentences(para))
        
        # Create chunks based on sentences
        chunks = self._create_semantic_chunks(sentences)
        
        # Add metadata to chunks
        result = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({
                "chunk_id": i,
                "chunk_type": "semantic"
            })
            
            result.append({
                "content": chunk,
                "metadata": chunk_metadata
            })
        
        return result
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        # Split by double newlines (common paragraph separator)
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Filter out empty paragraphs
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_sentences(self, paragraph: str) -> List[str]:
        """Split paragraph into sentences"""
        return sent_tokenize(paragraph)
    
    def _create_semantic_chunks(self, sentences: List[str]) -> List[str]:
        """Create chunks based on sentences and semantic similarity"""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If adding this sentence would exceed the chunk size and we already have content,
            # finish the current chunk and start a new one
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                
                # Start a new chunk with overlap
                overlap_size = 0
                overlap_chunk = []
                
                # Add sentences from the end of the previous chunk for overlap
                for s in reversed(current_chunk):
                    if overlap_size + len(s) <= self.chunk_overlap:
                        overlap_chunk.insert(0, s)
                        overlap_size += len(s)
                    else:
                        break
                
                current_chunk = overlap_chunk
                current_size = overlap_size
            
            # Add the current sentence to the chunk
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _calculate_semantic_similarity(self, sentence1: str, sentence2: str) -> float:
        """Calculate semantic similarity between two sentences"""
        if not self.embedding_model:
            # If no embedding model is provided, return 0 (no similarity)
            return 0.0
        
        # Get embeddings for the sentences
        embedding1 = self.embedding_model.embed_query(sentence1)
        embedding2 = self.embedding_model.embed_query(sentence2)
        
        # Calculate cosine similarity
        return self._cosine_similarity(embedding1, embedding2)
    
    def _cosine_similarity(self, vec1, vec2) -> float:
        """Calculate cosine similarity between two vectors"""
        import numpy as np
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)