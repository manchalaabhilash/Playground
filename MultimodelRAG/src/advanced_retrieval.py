"""
Advanced retrieval implementation for RAG systems.
Provides enhanced retrieval strategies beyond basic vector search.
"""

from typing import List, Dict, Any, Optional, Union, Callable
import numpy as np
import re

class HybridRetriever:
    """
    Combines multiple retrieval strategies for better results.
    Implements hybrid search using both vector and keyword-based retrieval.
    """
    
    def __init__(self, vector_retriever, keyword_retriever=None, weights: Optional[List[float]] = None):
        """Initialize the hybrid retriever"""
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        
        # If weights not provided, use equal weights
        if weights is None:
            self.weights = [0.7, 0.3] if keyword_retriever else [1.0]
        else:
            # Normalize weights to sum to 1
            total = sum(weights)
            self.weights = [w / total for w in weights]
    
    def get_relevant_documents(self, query: str, k: int = 5) -> List[Any]:
        """Get relevant documents using hybrid retrieval"""
        if not self.vector_retriever:
            return []
        
        # Get documents from vector retriever
        vector_docs = self.vector_retriever.get_relevant_documents(query, k=k*2)
        
        # If no keyword retriever, return vector results
        if not self.keyword_retriever:
            return vector_docs[:k]
        
        # Get documents from keyword retriever
        keyword_docs = self.keyword_retriever.get_relevant_documents(query, k=k*2)
        
        # Combine results with weights
        combined_docs = self._combine_results(vector_docs, keyword_docs, k)
        
        return combined_docs
    
    def _combine_results(self, vector_docs: List[Any], keyword_docs: List[Any], k: int) -> List[Any]:
        """Combine results from multiple retrievers"""
        # Create a dictionary to track document scores
        doc_scores = {}
        doc_map = {}
        
        # Process vector documents
        for i, doc in enumerate(vector_docs):
            doc_id = self._get_doc_id(doc)
            # Normalize score based on position (higher position = higher score)
            score = self.weights[0] * (1.0 - (i / len(vector_docs) if vector_docs else 1.0))
            doc_scores[doc_id] = score
            doc_map[doc_id] = doc
        
        # Process keyword documents
        for i, doc in enumerate(keyword_docs):
            doc_id = self._get_doc_id(doc)
            # Normalize score based on position
            score = self.weights[1] * (1.0 - (i / len(keyword_docs) if keyword_docs else 1.0))
            # Add to existing score if document already seen
            if doc_id in doc_scores:
                doc_scores[doc_id] += score
            else:
                doc_scores[doc_id] = score
                doc_map[doc_id] = doc
        
        # Sort documents by combined score
        sorted_doc_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        
        # Return top k documents
        return [doc_map[doc_id] for doc_id in sorted_doc_ids[:k]]
    
    def _get_doc_id(self, doc: Any) -> str:
        """Get a unique identifier for a document"""
        if hasattr(doc, "id"):
            return doc.id
        elif isinstance(doc, dict) and "id" in doc:
            return doc["id"]
        else:
            # Use object id as fallback
            return str(id(doc))

class KeywordRetriever:
    """
    Implements keyword-based retrieval using BM25 or TF-IDF.
    Complements vector retrieval for better results.
    """
    
    def __init__(self, documents: List[Any] = None, algorithm: str = "bm25"):
        """Initialize the keyword retriever"""
        self.documents = documents or []
        self.algorithm = algorithm.lower()
        self.index = None
        
        # Build index if documents provided
        if self.documents:
            self._build_index()
    
    def add_documents(self, documents: List[Any]):
        """Add documents to the retriever"""
        self.documents.extend(documents)
        self._build_index()
    
    def get_relevant_documents(self, query: str, k: int = 5) -> List[Any]:
        """Get relevant documents using keyword retrieval"""
        if not self.documents or not self.index:
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Calculate scores for each document
        scores = []
        for i, doc in enumerate(self.documents):
            if self.algorithm == "bm25":
                score = self._bm25_score(query_tokens, i)
            else:  # Default to TF-IDF
                score = self._tfidf_score(query_tokens, i)
            
            scores.append((i, score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k documents
        return [self.documents[i] for i, _ in scores[:k]]
    
    def _build_index(self):
        """Build the search index"""
        # Extract text from documents
        texts = [self._get_document_text(doc) for doc in self.documents]
        
        # Tokenize texts
        tokenized_texts = [self._tokenize(text) for text in texts]
        
        # Build document frequency index
        self.index = {
            "tokenized_texts": tokenized_texts,
            "doc_freqs": self._calculate_doc_freqs(tokenized_texts),
            "avg_doc_length": sum(len(tokens) for tokens in tokenized_texts) / len(tokenized_texts) if tokenized_texts else 0
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Simple tokenization by splitting on non-alphanumeric characters
        return re.findall(r'\w+', text.lower())
    
    def _calculate_doc_freqs(self, tokenized_texts: List[List[str]]) -> Dict[str, int]:
        """Calculate document frequencies for each term"""
        doc_freqs = {}
        for tokens in tokenized_texts:
            # Count each term only once per document
            for term in set(tokens):
                doc_freqs[term] = doc_freqs.get(term, 0) + 1
        
        return doc_freqs
    
    def _bm25_score(self, query_tokens: List[str], doc_idx: int) -> float:
        """Calculate BM25 score for a document"""
        # BM25 parameters
        k1 = 1.5
        b = 0.75
        
        score = 0.0
        doc_tokens = self.index["tokenized_texts"][doc_idx]
        doc_length = len(doc_tokens)
        avg_doc_length = self.index["avg_doc_length"]
        
        # Count term frequencies in document
        term_freqs = {}
        for term in doc_tokens:
            term_freqs[term] = term_freqs.get(term, 0) + 1
        
        for term in query_tokens:
            if term not in self.index["doc_freqs"]:
                continue
            
            # Calculate IDF
            df = self.index["doc_freqs"][term]
            idf = np.log((len(self.documents) - df + 0.5) / (df + 0.5) + 1.0)
            
            # Calculate term frequency
            tf = term_freqs.get(term, 0)
            
            # BM25 formula
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_length / avg_doc_length)
            
            score += idf * numerator / denominator
        
        return score
    
    def _tfidf_score(self, query_tokens: List[str], doc_idx: int) -> float:
        """Calculate TF-IDF score for a document"""
        score = 0.0
        doc_tokens = self.index["tokenized_texts"][doc_idx]
        
        # Count term frequencies in document
        term_freqs = {}
        for term in doc_tokens:
            term_freqs[term] = term_freqs.get(term, 0) + 1
        
        for term in query_tokens:
            if term not in self.index["doc_freqs"]:
                continue
            
            # Calculate IDF
            df = self.index["doc_freqs"][term]
            idf = np.log(len(self.documents) / df) if df > 0 else 0
            
            # Calculate TF
            tf = term_freqs.get(term, 0) / len(doc_tokens) if doc_tokens else 0
            
            # TF-IDF score
            score += tf * idf
        
        return score
    
    def _get_document_text(self, doc: Any) -> str:
        """Extract text content from document"""
        if hasattr(doc, "page_content"):
            return doc.page_content
        elif isinstance(doc, dict) and "content" in doc:
            return doc["content"]
        else:
            return str(doc)


class ReRanker:
    """
    Re-ranks retrieved documents for better relevance.
    Implements cross-encoder or semantic re-ranking.
    """
    
    def __init__(self, model=None, use_cross_encoder: bool = True):
        """Initialize the re-ranker"""
        self.model = model
        self.use_cross_encoder = use_cross_encoder
    
    def rerank(self, query: str, documents: List[Any], top_k: int = None) -> List[Any]:
        """Re-rank documents based on relevance to query"""
        if not documents:
            return []
        
        # If no model provided, use simple keyword matching
        if not self.model:
            return self._keyword_rerank(query, documents, top_k)
        
        # Extract text from documents
        doc_texts = [self._get_document_text(doc) for doc in documents]
        
        # If using cross-encoder
        if self.use_cross_encoder:
            # Prepare pairs for cross-encoder
            pairs = [[query, doc_text] for doc_text in doc_texts]
            
            try:
                # Get relevance scores
                scores = self.model.predict(pairs)
                
                # Create document-score pairs
                doc_scores = list(zip(documents, scores))
                
                # Sort by score in descending order
                doc_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Return top_k documents if specified
                if top_k is not None:
                    return [doc for doc, _ in doc_scores[:top_k]]
                else:
                    return [doc for doc, _ in doc_scores]
            except Exception as e:
                print(f"Error in cross-encoder re-ranking: {str(e)}")
                return documents
        else:
            # Use bi-encoder (embedding similarity)
            try:
                # Get query embedding
                query_embedding = self.model.encode(query, convert_to_tensor=True)
                
                # Get document embeddings
                doc_embeddings = self.model.encode(doc_texts, convert_to_tensor=True)
                
                # Calculate similarities
                similarities = []
                for i, doc_embedding in enumerate(doc_embeddings):
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(query_embedding, doc_embedding)
                    similarities.append((i, similarity))
                
                # Sort by similarity in descending order
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                # Return top_k documents if specified
                if top_k is not None:
                    return [documents[i] for i, _ in similarities[:top_k]]
                else:
                    return [documents[i] for i, _ in similarities]
            except Exception as e:
                print(f"Error in bi-encoder re-ranking: {str(e)}")
                return documents
    
    def _keyword_rerank(self, query: str, documents: List[Any], top_k: int = None) -> List[Any]:
        """Re-rank documents based on keyword matching"""
        # Extract query keywords
        query_keywords = set(self._tokenize(query))
        
        # Score documents based on keyword matches
        doc_scores = []
        for doc in documents:
            doc_text = self._get_document_text(doc)
            doc_tokens = set(self._tokenize(doc_text))
            
            # Calculate score based on keyword overlap
            overlap = len(query_keywords.intersection(doc_tokens))
            score = overlap / len(query_keywords) if query_keywords else 0
            
            doc_scores.append((doc, score))
        
        # Sort by score in descending order
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k documents if specified
        if top_k is not None:
            return [doc for doc, _ in doc_scores[:top_k]]
        else:
            return [doc for doc, _ in doc_scores]
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        return re.findall(r'\w+', text.lower())
    
    def _cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _get_document_text(self, doc: Any) -> str:
        """Extract text content from document"""
        if hasattr(doc, "page_content"):
            return doc.page_content
        elif isinstance(doc, dict) and "content" in doc:
            return doc["content"]
        else:
            return str(doc)


class SemanticRouter:
    """
    Routes queries to appropriate retrievers based on semantic understanding.
    Implements query routing for multi-index or specialized retrieval.
    """
    
    def __init__(self, retrievers: Dict[str, Any] = None, embedding_model=None):
        """Initialize the semantic router"""
        self.retrievers = retrievers or {}
        self.embedding_model = embedding_model
        self.fallback_retriever = None
    
    def add_retriever(self, name: str, retriever, description: str = ""):
        """Add a retriever with a description"""
        self.retrievers[name] = {
            "retriever": retriever,
            "description": description
        }
    
    def set_fallback_retriever(self, retriever):
        """Set a fallback retriever to use when no suitable retriever is found"""
        self.fallback_retriever = retriever
    
    def route_query(self, query: str) -> Dict[str, Any]:
        """Route a query to the most appropriate retriever(s)"""
        if not self.retrievers:
            return {
                "retriever": self.fallback_retriever,
                "confidence": 0.0,
                "name": "fallback"
            }
        
        # If only one retriever, use it
        if len(self.retrievers) == 1:
            name = next(iter(self.retrievers))
            return {
                "retriever": self.retrievers[name]["retriever"],
                "confidence": 1.0,
                "name": name
            }
        
        # If embedding model available, use semantic routing
        if self.embedding_model:
            return self._semantic_route(query)
        
        # Otherwise, use keyword-based routing
        return self._keyword_route(query)
    
    def _semantic_route(self, query: str) -> Dict[str, Any]:
        """Route query based on semantic similarity to retriever descriptions"""
        try:
            # Get query embedding
            query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
            
            # Calculate similarity to each retriever description
            similarities = []
            for name, info in self.retrievers.items():
                description = info["description"]
                
                # Skip if no description
                if not description:
                    continue
                
                # Get description embedding
                desc_embedding = self.embedding_model.encode(description, convert_to_tensor=True)
                
                # Calculate similarity
                similarity = self._cosine_similarity(query_embedding, desc_embedding)
                similarities.append((name, similarity))
            
            # If no similarities calculated, use fallback
            if not similarities:
                return {
                    "retriever": self.fallback_retriever,
                    "confidence": 0.0,
                    "name": "fallback"
                }
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Get best match
            best_name, confidence = similarities[0]
            
            # If confidence too low, use fallback
            if confidence < 0.5 and self.fallback_retriever:
                return {
                    "retriever": self.fallback_retriever,
                    "confidence": confidence,
                    "name": "fallback"
                }
            
            return {
                "retriever": self.retrievers[best_name]["retriever"],
                "confidence": confidence,
                "name": best_name
            }
        except Exception as e:
            print(f"Error in semantic routing: {str(e)}")
            
            # Use fallback if available
            if self.fallback_retriever:
                return {
                    "retriever": self.fallback_retriever,
                    "confidence": 0.0,
                    "name": "fallback"
                }
            
            # Otherwise use first retriever
            name = next(iter(self.retrievers))
            return {
                "retriever": self.retrievers[name]["retriever"],
                "confidence": 0.0,
                "name": name
            }
    
    def _keyword_route(self, query: str) -> Dict[str, Any]:
        """Route query based on keyword matching to retriever descriptions"""
        # Extract query keywords
        query_keywords = set(self._tokenize(query))
        
        # Score retrievers based on keyword matches
        scores = []
        for name, info in self.retrievers.items():
            description = info["description"]
            
            # Skip if no description
            if not description:
                continue
            
            # Tokenize description
            desc_tokens = set(self._tokenize(description))
            
            # Calculate score based on keyword overlap
            overlap = len(query_keywords.intersection(desc_tokens))
            score = overlap / len(query_keywords) if query_keywords else 0
            
            scores.append((name, score))
        
        # If no scores calculated, use fallback
        if not scores:
            return {
                "retriever": self.fallback_retriever,
                "confidence": 0.0,
                "name": "fallback"
            }
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get best match
        best_name, confidence = scores[0]
        
        # If confidence too low, use fallback
        if confidence < 0.2 and self.fallback_retriever:
            return {
                "retriever": self.fallback_retriever,
                "confidence": confidence,
                "name": "fallback"
            }
        
        return {
            "retriever": self.retrievers[best_name]["retriever"],
            "confidence": confidence,
            "name": best_name
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        return re.findall(r'\w+', text.lower())
    
    def _cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
