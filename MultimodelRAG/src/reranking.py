"""
Advanced reranking implementation for RAG systems.
Provides multiple reranking strategies to improve retrieval quality.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable
import torch
from sentence_transformers import CrossEncoder

class BaseReranker:
    """Base class for rerankers"""
    
    def rerank(self, query: str, documents: List[Any], top_k: Optional[int] = None) -> List[Any]:
        """Rerank documents based on relevance to the query"""
        raise NotImplementedError("Subclasses must implement rerank method")

class CrossEncoderReranker(BaseReranker):
    """Reranker using cross-encoder models for more accurate relevance scoring"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize the cross-encoder reranker"""
        self.model_name = model_name
        try:
            self.model = CrossEncoder(model_name)
        except Exception as e:
            print(f"Error loading cross-encoder model: {str(e)}")
            self.model = None
    
    def rerank(self, query: str, documents: List[Any], top_k: Optional[int] = None) -> List[Any]:
        """Rerank documents using cross-encoder scores"""
        if not documents:
            return []
        
        if not self.model:
            # Fallback if model failed to load
            if top_k and top_k < len(documents):
                return documents[:top_k]
            return documents
        
        # Prepare document-query pairs for scoring
        pairs = [(query, self._get_document_text(doc)) for doc in documents]
        
        # Score pairs with cross-encoder
        scores = self.model.predict(pairs)
        
        # Sort documents by score
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k documents if specified
        if top_k and top_k < len(scored_docs):
            return [doc for doc, _ in scored_docs[:top_k]]
        
        return [doc for doc, _ in scored_docs]
    
    def _get_document_text(self, doc: Any) -> str:
        """Extract text content from document"""
        if hasattr(doc, "page_content"):
            return doc.page_content
        elif isinstance(doc, dict) and "content" in doc:
            return doc["content"]
        else:
            return str(doc)

class SemanticReranker(BaseReranker):
    """Reranker using semantic similarity for improved relevance"""
    
    def __init__(self, embedding_function: Optional[Callable] = None):
        """Initialize the semantic reranker"""
        self.embedding_function = embedding_function
    
    def rerank(self, query: str, documents: List[Any], top_k: Optional[int] = None) -> List[Any]:
        """Rerank documents using semantic similarity"""
        if not documents or not self.embedding_function:
            if top_k and top_k < len(documents):
                return documents[:top_k]
            return documents
        
        # Get query embedding
        query_embedding = self.embedding_function(query)
        
        # Get document embeddings and calculate similarity
        scored_docs = []
        for doc in documents:
            doc_text = self._get_document_text(doc)
            doc_embedding = self.embedding_function(doc_text)
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            scored_docs.append((doc, similarity))
        
        # Sort by similarity score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k documents if specified
        if top_k and top_k < len(scored_docs):
            return [doc for doc, _ in scored_docs[:top_k]]
        
        return [doc for doc, _ in scored_docs]
    
    def _get_document_text(self, doc: Any) -> str:
        """Extract text content from document"""
        if hasattr(doc, "page_content"):
            return doc.page_content
        elif isinstance(doc, dict) and "content" in doc:
            return doc["content"]
        else:
            return str(doc)
    
    def _cosine_similarity(self, vec1, vec2) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

class HybridReranker(BaseReranker):
    """Combines multiple reranking strategies for better results"""
    
    def __init__(self, rerankers: List[BaseReranker], weights: Optional[List[float]] = None):
        """Initialize the hybrid reranker"""
        self.rerankers = rerankers
        
        # If weights not provided, use equal weights
        if weights is None:
            self.weights = [1.0 / len(rerankers)] * len(rerankers)
        else:
            # Normalize weights to sum to 1
            total = sum(weights)
            self.weights = [w / total for w in weights]
    
    def rerank(self, query: str, documents: List[Any], top_k: Optional[int] = None) -> List[Any]:
        """Rerank documents using multiple strategies"""
        if not documents or not self.rerankers:
            if top_k and top_k < len(documents):
                return documents[:top_k]
            return documents
        
        # Create a dictionary to track document scores
        doc_scores = {self._get_doc_id(doc): 0.0 for doc in documents}
        doc_map = {self._get_doc_id(doc): doc for doc in documents}
        
        # Apply each reranker and combine scores
        for reranker, weight in zip(self.rerankers, self.weights):
            # Get reranked documents
            reranked_docs = reranker.rerank(query, documents)
            
            # Assign scores based on position (higher position = higher score)
            for i, doc in enumerate(reranked_docs):
                doc_id = self._get_doc_id(doc)
                # Normalize score based on position
                score = 1.0 - (i / len(reranked_docs))
                doc_scores[doc_id] += weight * score
        
        # Sort documents by combined score
        sorted_doc_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        
        # Return top_k documents if specified
        if top_k and top_k < len(sorted_doc_ids):
            return [doc_map[doc_id] for doc_id in sorted_doc_ids[:top_k]]
        
        return [doc_map[doc_id] for doc_id in sorted_doc_ids]
    
    def _get_doc_id(self, doc: Any) -> str:
        """Get a unique identifier for a document"""
        if hasattr(doc, "id"):
            return doc.id
        elif isinstance(doc, dict) and "id" in doc:
            return doc["id"]
        else:
            # Use object id as fallback
            return str(id(doc))