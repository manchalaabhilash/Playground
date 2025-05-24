"""
Recursive retrieval implementation for advanced RAG systems.
Provides strategies for iterative and recursive document retrieval.
"""

from typing import List, Dict, Any, Optional, Callable
import json

class RecursiveRetriever:
    """
    Implements recursive retrieval strategies for improved document retrieval.
    Uses initial retrieval results to formulate better follow-up queries.
    """
    
    def __init__(self, base_retriever, llm_client=None, max_iterations: int = 3):
        """Initialize the recursive retriever"""
        self.base_retriever = base_retriever
        self.llm_client = llm_client
        self.max_iterations = max_iterations
    
    def get_relevant_documents(self, query: str, k: int = 5) -> List[Any]:
        """Get relevant documents using recursive retrieval"""
        if not self.base_retriever:
            return []
        
        # If no LLM client, fall back to base retriever
        if not self.llm_client:
            return self.base_retriever.get_relevant_documents(query, k=k)
        
        # Initial retrieval
        documents = self.base_retriever.get_relevant_documents(query, k=k)
        
        # Track all retrieved documents
        all_documents = {self._get_doc_id(doc): doc for doc in documents}
        
        # Recursive retrieval
        current_query = query
        for i in range(self.max_iterations - 1):
            # If no documents retrieved, break
            if not documents:
                break
            
            # Generate follow-up query based on retrieved documents
            follow_up_query = self._generate_follow_up_query(current_query, documents)
            
            # If follow-up query is the same as current query, break
            if follow_up_query == current_query:
                break
            
            # Retrieve documents with follow-up query
            current_query = follow_up_query
            documents = self.base_retriever.get_relevant_documents(current_query, k=k)
            
            # Add new documents to the collection
            for doc in documents:
                doc_id = self._get_doc_id(doc)
                if doc_id not in all_documents:
                    all_documents[doc_id] = doc
        
        # Return top k documents
        return list(all_documents.values())[:k]
    
    def _generate_follow_up_query(self, query: str, documents: List[Any]) -> str:
        """Generate a follow-up query based on retrieved documents"""
        # Extract text from documents
        doc_texts = [self._get_document_text(doc) for doc in documents]
        context = "\n\n".join(doc_texts)
        
        # Prompt for the LLM to generate a follow-up query
        prompt = f"""You are an AI assistant that helps improve search queries based on initial search results.
Given the following original query and search results, generate a more specific follow-up query that would help find more relevant information.

Original Query: {query}

Search Results:
{context}

Based on these results, generate a more specific follow-up query that would help find additional relevant information.
If the original query already seems optimal, return it unchanged.

Follow-up Query:"""
        
        try:
            follow_up_query = self.llm_client.generate_text(prompt).strip()
            return follow_up_query
        except Exception as e:
            print(f"Error generating follow-up query: {str(e)}")
            return query
    
    def _get_document_text(self, doc: Any) -> str:
        """Extract text content from document"""
        if hasattr(doc, "page_content"):
            return doc.page_content
        elif isinstance(doc, dict) and "content" in doc:
            return doc["content"]
        else:
            return str(doc)
    
    def _get_doc_id(self, doc: Any) -> str:
        """Get a unique identifier for a document"""
        if hasattr(doc, "id"):
            return doc.id
        elif isinstance(doc, dict) and "id" in doc:
            return doc["id"]
        else:
            # Use object id as fallback
            return str(id(doc))

class SmallToLargeRetriever:
    """
    Implements small-to-large retrieval strategy.
    First retrieves smaller chunks, then expands to larger context.
    """
    
    def __init__(self, small_chunk_retriever, large_chunk_retriever):
        """Initialize the small-to-large retriever"""
        self.small_chunk_retriever = small_chunk_retriever
        self.large_chunk_retriever = large_chunk_retriever
    
    def get_relevant_documents(self, query: str, k: int = 5) -> List[Any]:
        """Get relevant documents using small-to-large retrieval"""
        if not self.small_chunk_retriever or not self.large_chunk_retriever:
            # Fall back to base retriever if available
            if hasattr(self, "base_retriever") and self.base_retriever:
                return self.base_retriever.get_relevant_documents(query, k=k)
            return []
        
        # First retrieve from small chunks
        small_chunks = self.small_chunk_retriever.get_relevant_documents(query, k=k*2)
        
        if not small_chunks:
            return []
        
        # Get parent IDs or document IDs from small chunks
        parent_ids = set()
        for chunk in small_chunks:
            if hasattr(chunk, "metadata") and chunk.metadata.get("parent_id"):
                parent_ids.add(chunk.metadata.get("parent_id"))
            elif hasattr(chunk, "metadata") and chunk.metadata.get("document_id"):
                parent_ids.add(chunk.metadata.get("document_id"))
        
        # If no parent IDs found, return small chunks
        if not parent_ids:
            return small_chunks[:k]
        
        # Retrieve large chunks based on parent IDs
        large_chunks = []
        for parent_id in parent_ids:
            # Create filter for parent ID
            filter_dict = {"metadata": {"parent_id": parent_id}}
            
            # Retrieve large chunks with filter
            parent_chunks = self.large_chunk_retriever.get_relevant_documents(
                query, 
                k=1,
                filter=filter_dict
            )
            
            large_chunks.extend(parent_chunks)
        
        # If no large chunks found, return small chunks
        if not large_chunks:
            return small_chunks[:k]
        
        # Return top k large chunks
        return large_chunks[:k]


class TableRetriever:
    """
    Specialized retriever for table data.
    Handles embedded tables and structured data.
    """
    
    def __init__(self, base_retriever, table_parser=None):
        """Initialize the table retriever"""
        self.base_retriever = base_retriever
        self.table_parser = table_parser
    
    def get_relevant_documents(self, query: str, k: int = 5) -> List[Any]:
        """Get relevant table documents"""
        if not self.base_retriever:
            return []
        
        # Retrieve documents that might contain tables
        documents = self.base_retriever.get_relevant_documents(query, k=k*2)
        
        # If no table parser, return as is
        if not self.table_parser:
            return documents[:k]
        
        # Extract and parse tables from documents
        table_documents = []
        for doc in documents:
            tables = self.table_parser.extract_tables(self._get_document_text(doc))
            
            if tables:
                for i, table in enumerate(tables):
                    # Create metadata for the table
                    table_metadata = {}
                    if hasattr(doc, "metadata"):
                        table_metadata = doc.metadata.copy()
                    
                    table_metadata.update({
                        "table_index": i,
                        "source_doc_id": self._get_doc_id(doc),
                        "content_type": "table"
                    })
                    
                    # Create a new document for the table
                    table_doc = {
                        "content": table,
                        "metadata": table_metadata
                    }
                    
                    table_documents.append(table_doc)
        
        # If no tables found, return original documents
        if not table_documents:
            return documents[:k]
        
        # Rerank table documents based on relevance to query
        table_documents = self._rerank_tables(query, table_documents)
        
        # Return top k table documents
        return table_documents[:k]
    
    def _rerank_tables(self, query: str, table_documents: List[Any]) -> List[Any]:
        """Rerank table documents based on relevance to query"""
        # Simple keyword-based scoring
        query_keywords = set(query.lower().split())
        
        scored_tables = []
        for doc in table_documents:
            table_content = doc["content"] if isinstance(doc, dict) else str(doc)
            table_content_lower = table_content.lower()
            
            # Count keyword matches
            score = sum(1 for keyword in query_keywords if keyword in table_content_lower)
            
            scored_tables.append((doc, score))
        
        # Sort by score
        scored_tables.sort(key=lambda x: x[1], reverse=True)
        
        # Return documents only
        return [doc for doc, _ in scored_tables]
    
    def _get_document_text(self, doc: Any) -> str:
        """Extract text content from document"""
        if hasattr(doc, "page_content"):
            return doc.page_content
        elif isinstance(doc, dict) and "content" in doc:
            return doc["content"]
        else:
            return str(doc)
    
    def _get_doc_id(self, doc: Any) -> str:
        """Get a unique identifier for a document"""
        if hasattr(doc, "id"):
            return doc.id
        elif isinstance(doc, dict) and "id" in doc:
            return doc["id"]
        else:
            # Use object id as fallback
            return str(id(doc))


class MultiDocumentAgent:
    """
    Agent that works with multiple documents to answer complex queries.
    Uses recursive retrieval and reasoning to synthesize information.
    """
    
    def __init__(self, retriever, llm_client=None, max_documents: int = 10):
        """Initialize the multi-document agent"""
        self.retriever = retriever
        self.llm_client = llm_client
        self.max_documents = max_documents
    
    def answer_query(self, query: str) -> Dict[str, Any]:
        """Answer a query using multiple documents"""
        if not self.retriever or not self.llm_client:
            return {
                "answer": "I don't have enough information to answer this query.",
                "documents": [],
                "reasoning": "Missing retriever or LLM client."
            }
        
        # Retrieve relevant documents
        documents = self.retriever.get_relevant_documents(query, k=self.max_documents)
        
        if not documents:
            return {
                "answer": "I couldn't find any relevant information to answer your query.",
                "documents": [],
                "reasoning": "No relevant documents found."
            }
        
        # Extract text from documents
        doc_texts = [self._get_document_text(doc) for doc in documents]
        context = "\n\n".join(doc_texts)
        
        # Generate reasoning and answer
        prompt = f"""You are an AI assistant that answers questions based on multiple documents.
Given the following query and context from multiple documents, provide a comprehensive answer.
First, analyze the information from each document. Then, synthesize the information to provide a complete answer.

Query: {query}

Context from multiple documents:
{context}

Please provide your answer in the following format:
REASONING: Step-by-step analysis of the information from the documents
ANSWER: Your final comprehensive answer to the query
"""
        
        try:
            response = self.llm_client.generate_text(prompt)
            
            # Extract reasoning and answer
            reasoning = ""
            answer = ""
            
            if "REASONING:" in response and "ANSWER:" in response:
                reasoning_part = response.split("ANSWER:")[0]
                reasoning = reasoning_part.replace("REASONING:", "").strip()
                
                answer_part = response.split("ANSWER:")[1]
                answer = answer_part.strip()
            else:
                # If format not followed, use the whole response as answer
                answer = response
            
            return {
                "answer": answer,
                "documents": documents,
                "reasoning": reasoning
            }
        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "documents": documents,
                "reasoning": "Error in LLM processing."
            }
    
    def _get_document_text(self, doc: Any) -> str:
        """Extract text content from document"""
        if hasattr(doc, "page_content"):
            return doc.page_content
        elif isinstance(doc, dict) and "content" in doc:
            return doc["content"]
        else:
            return str(doc)
