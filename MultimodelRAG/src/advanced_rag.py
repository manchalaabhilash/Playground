"""
Advanced RAG techniques implementation.
Includes query routing, reranking, and hybrid search strategies.
"""

import os
import sys
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

sys.path.append('.')
from src.mcp_integration import McpRouter

class QueryRouter:
    """
    Routes queries to different retrieval strategies based on query type.
    """
    
    def __init__(self):
        """Initialize the query router"""
        self.mcp_router = McpRouter()
        self.query_types = {
            "factual": self._handle_factual_query,
            "conceptual": self._handle_conceptual_query,
            "visual": self._handle_visual_query,
            "mixed": self._handle_mixed_query
        }
    
    def classify_query(self, query: str) -> str:
        """Classify the query type"""
        # Simple keyword-based classification
        # In a real system, this would use a more sophisticated approach
        query = query.lower()
        
        if any(word in query for word in ["image", "picture", "photo", "show", "look", "see"]):
            return "visual"
        elif any(word in query for word in ["what is", "who is", "when", "where", "how many"]):
            return "factual"
        elif any(word in query for word in ["explain", "why", "how does", "concept", "understand"]):
            return "conceptual"
        else:
            return "mixed"
    
    def route_query(self, query: str, text_retriever=None, image_retriever=None):
        """Route the query to the appropriate handler"""
        query_type = self.classify_query(query)
        handler = self.query_types.get(query_type, self._handle_mixed_query)
        return handler(query, text_retriever, image_retriever)
    
    def _handle_factual_query(self, query, text_retriever, image_retriever):
        """Handle factual queries - prioritize text retrieval"""
        relevant_texts = []
        relevant_images = []
        
        if text_retriever:
            relevant_texts = text_retriever.get_relevant_documents(query)
        
        # Only use images if text retrieval yields limited results
        if image_retriever and len(relevant_texts) < 2:
            relevant_images = image_retriever.get_relevant_documents(query)
        
        return relevant_texts, relevant_images
    
    def _handle_conceptual_query(self, query, text_retriever, image_retriever):
        """Handle conceptual queries - use both text and images with text priority"""
        relevant_texts = []
        relevant_images = []
        
        if text_retriever:
            relevant_texts = text_retriever.get_relevant_documents(query)
        
        if image_retriever:
            # Get fewer images for conceptual queries
            relevant_images = image_retriever.get_relevant_documents(query, k=2)
        
        return relevant_texts, relevant_images
    
    def _handle_visual_query(self, query, text_retriever, image_retriever):
        """Handle visual queries - prioritize image retrieval"""
        relevant_texts = []
        relevant_images = []
        
        if image_retriever:
            relevant_images = image_retriever.get_relevant_documents(query)
        
        if text_retriever:
            # Get text that might describe the images
            relevant_texts = text_retriever.get_relevant_documents(query)
        
        return relevant_texts, relevant_images
    
    def _handle_mixed_query(self, query, text_retriever, image_retriever):
        """Handle mixed queries - balanced approach"""
        relevant_texts = []
        relevant_images = []
        
        if text_retriever:
            relevant_texts = text_retriever.get_relevant_documents(query)
        
        if image_retriever:
            relevant_images = image_retriever.get_relevant_documents(query)
        
        return relevant_texts, relevant_images

class HybridRetriever:
    """
    Implements hybrid retrieval strategies combining dense and sparse retrieval.
    """
    
    def __init__(self, dense_retriever=None, sparse_retriever=None, alpha=0.5):
        """Initialize the hybrid retriever"""
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.alpha = alpha  # Weight for dense retrieval scores (1-alpha for sparse)
    
    def get_relevant_documents(self, query: str, k: int = 5) -> List[Any]:
        """Get relevant documents using hybrid retrieval"""
        if not self.dense_retriever and not self.sparse_retriever:
            return []
        
        # If only one retriever is available, use it
        if not self.dense_retriever:
            return self.sparse_retriever.get_relevant_documents(query, k=k)
        if not self.sparse_retriever:
            return self.dense_retriever.get_relevant_documents(query, k=k)
        
        # Get results from both retrievers
        dense_results = self.dense_retriever.get_relevant_documents(query, k=k*2)
        sparse_results = self.sparse_retriever.get_relevant_documents(query, k=k*2)
        
        # Combine and rerank results
        combined_results = self._combine_results(dense_results, sparse_results, k)
        return combined_results
    
    def _combine_results(self, dense_results: List[Any], sparse_results: List[Any], k: int) -> List[Any]:
        """Combine and rerank results from dense and sparse retrievers"""
        # Create a dictionary to store combined scores
        combined_scores = {}
        
        # Process dense results
        for i, doc in enumerate(dense_results):
            doc_id = getattr(doc, "id", str(i))
            # Normalize score based on position (higher position = higher score)
            score = 1.0 - (i / len(dense_results))
            combined_scores[doc_id] = {
                "doc": doc,
                "score": self.alpha * score
            }
        
        # Process sparse results
        for i, doc in enumerate(sparse_results):
            doc_id = getattr(doc, "id", str(i))
            # Normalize score based on position
            score = 1.0 - (i / len(sparse_results))
            
            if doc_id in combined_scores:
                # Add sparse score to existing document
                combined_scores[doc_id]["score"] += (1 - self.alpha) * score
            else:
                # Add new document with sparse score
                combined_scores[doc_id] = {
                    "doc": doc,
                    "score": (1 - self.alpha) * score
                }
        
        # Sort by combined score and take top k
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        
        return [item["doc"] for item in sorted_results[:k]]

class Reranker:
    """
    Reranks retrieved documents based on relevance to the query.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize the reranker"""
        self.model_name = model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        # In a real implementation, this would load a cross-encoder model
        # For simplicity, we'll use a dummy implementation
    
    def rerank(self, query: str, documents: List[Any], top_k: int = None) -> List[Any]:
        """Rerank documents based on relevance to the query"""
        if not documents:
            return []
        
        # In a real implementation, this would use the cross-encoder to score documents
        # For now, we'll just return the original documents
        if top_k and top_k < len(documents):
            return documents[:top_k]
        return documents