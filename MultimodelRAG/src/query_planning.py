"""
Query planning and routing implementation for advanced RAG systems.
Provides strategies for decomposing complex queries and routing to appropriate retrievers.
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Callable
import json

class QueryPlanner:
    """
    Plans and decomposes complex queries into simpler sub-queries.
    """
    
    def __init__(self, llm_client=None):
        """Initialize the query planner"""
        self.llm_client = llm_client
    
    def plan_query(self, query: str) -> Dict[str, Any]:
        """Plan a complex query by decomposing it into sub-queries"""
        if not self.llm_client:
            # If no LLM client, return simple plan with original query
            return {
                "original_query": query,
                "is_complex": False,
                "sub_queries": [query],
                "reasoning": "No LLM client available for complex planning."
            }
        
        # Prompt for the LLM to decompose the query
        prompt = f"""You are an AI assistant that helps decompose complex queries into simpler sub-queries.
Given the following query, determine if it's complex (requires multiple pieces of information or steps to answer).
If it is complex, break it down into 2-5 simpler sub-queries that together would help answer the original query.

Query: {query}

Output your response in the following JSON format:
{{
  "is_complex": true/false,
  "reasoning": "Brief explanation of why the query is complex or not",
  "sub_queries": [
    "First sub-query",
    "Second sub-query",
    ...
  ]
}}
"""
        
        # Get response from LLM
        try:
            response = self.llm_client.generate_text(prompt)
            plan = json.loads(response)
            
            # Add original query to the plan
            plan["original_query"] = query
            
            return plan
        except Exception as e:
            print(f"Error in query planning: {str(e)}")
            # Return simple plan with original query
            return {
                "original_query": query,
                "is_complex": False,
                "sub_queries": [query],
                "reasoning": f"Error in query planning: {str(e)}"
            }

class QueryRouter:
    """
    Routes queries to appropriate retrievers based on query type and content.
    """
    
    def __init__(self, retrievers: Dict[str, Any] = None, classifier=None):
        """Initialize the query router"""
        self.retrievers = retrievers or {}
        self.classifier = classifier
        
        # Default routing rules based on keywords
        self.keyword_rules = {
            "image": ["image", "picture", "photo", "visual", "look", "see", "shown"],
            "table": ["table", "row", "column", "cell", "spreadsheet", "excel", "csv"],
            "code": ["code", "function", "class", "method", "programming", "algorithm"],
            "math": ["equation", "formula", "calculation", "math", "compute", "solve"],
            "general": []  # Default category
        }
    
    def route_query(self, query: str) -> Tuple[str, Any]:
        """Route a query to the appropriate retriever"""
        if not self.retrievers:
            return "general", None
        
        # If classifier is available, use it
        if self.classifier:
            try:
                category = self.classifier.classify(query)
                retriever = self.retrievers.get(category)
                if retriever:
                    return category, retriever
            except Exception as e:
                print(f"Error in query classification: {str(e)}")
        
        # Fallback to keyword-based routing
        query_lower = query.lower()
        
        for category, keywords in self.keyword_rules.items():
            if any(keyword in query_lower for keyword in keywords):
                retriever = self.retrievers.get(category)
                if retriever:
                    return category, retriever
        
        # Default to general retriever
        return "general", self.retrievers.get("general")
    
    def add_retriever(self, category: str, retriever: Any):
        """Add a retriever for a specific category"""
        self.retrievers[category] = retriever
    
    def add_keyword_rule(self, category: str, keywords: List[str]):
        """Add keyword routing rules for a category"""
        if category in self.keyword_rules:
            self.keyword_rules[category].extend(keywords)
        else:
            self.keyword_rules[category] = keywords

class MultiRetrievalOrchestrator:
    """
    Orchestrates the retrieval process for complex queries using query planning and routing.
    """
    
    def __init__(self, query_planner: QueryPlanner, query_router: QueryRouter, reranker=None):
        """Initialize the multi-retrieval orchestrator"""
        self.query_planner = query_planner
        self.query_router = query_router
        self.reranker = reranker
    
    def retrieve(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Retrieve documents for a complex query"""
        # Plan the query
        plan = self.query_planner.plan_query(query)
        
        # If query is not complex, use simple retrieval
        if not plan.get("is_complex", False):
            category, retriever = self.query_router.route_query(query)
            
            if not retriever:
                return {
                    "original_query": query,
                    "is_complex": False,
                    "documents": [],
                    "error": "No suitable retriever found"
                }
            
            documents = retriever.get_relevant_documents(query, k=top_k)
            
            # Rerank if reranker is available
            if self.reranker:
                documents = self.reranker.rerank(query, documents, top_k=top_k)
            
            return {
                "original_query": query,
                "is_complex": False,
                "documents": documents,
                "category": category
            }
        
        # For complex queries, retrieve documents for each sub-query
        results = {}
        all_documents = []
        
        for sub_query in plan.get("sub_queries", []):
            category, retriever = self.query_router.route_query(sub_query)
            
            if retriever:
                documents = retriever.get_relevant_documents(sub_query, k=top_k)
                
                # Add to results
                results[sub_query] = {
                    "category": category,
                    "documents": documents
                }
                
                all_documents.extend(documents)
        
        # Rerank all documents if reranker is available
        if self.reranker and all_documents:
            all_documents = self.reranker.rerank(query, all_documents, top_k=top_k)
        
        return {
            "original_query": query,
            "is_complex": True,
            "sub_queries": results,
            "documents": all_documents,
            "reasoning": plan.get("reasoning", "")
        }