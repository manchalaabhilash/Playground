"""
Parent-child chunking implementation for advanced RAG.
Creates hierarchical chunks with parent-child relationships.
"""

import re
from typing import List, Dict, Any, Optional, Tuple

class ParentChildChunker:
    """
    Creates hierarchical chunks with parent-child relationships.
    Useful for maintaining context across different levels of detail.
    """
    
    def __init__(self, parent_chunk_size: int = 1000, child_chunk_size: int = 300, chunk_overlap: int = 50):
        """Initialize the parent-child chunker"""
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.chunk_overlap = chunk_overlap
    
    def create_hierarchical_chunks(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create hierarchical chunks with parent-child relationships"""
        # First create parent chunks
        parent_chunks = self._create_chunks(text, self.parent_chunk_size, self.chunk_overlap)
        
        # Then create child chunks for each parent
        result = {
            "parents": [],
            "children": []
        }
        
        for i, parent in enumerate(parent_chunks):
            parent_id = f"parent_{i}"
            
            # Add parent metadata
            parent_metadata = metadata.copy() if metadata else {}
            parent_metadata.update({
                "chunk_id": parent_id,
                "chunk_type": "parent",
                "level": 0
            })
            
            # Add parent to result
            result["parents"].append({
                "content": parent,
                "metadata": parent_metadata
            })
            
            # Create child chunks
            child_chunks = self._create_chunks(parent, self.child_chunk_size, self.chunk_overlap)
            
            for j, child in enumerate(child_chunks):
                child_id = f"child_{i}_{j}"
                
                # Add child metadata
                child_metadata = metadata.copy() if metadata else {}
                child_metadata.update({
                    "chunk_id": child_id,
                    "parent_id": parent_id,
                    "chunk_type": "child",
                    "level": 1
                })
                
                # Add child to result
                result["children"].append({
                    "content": child,
                    "metadata": child_metadata
                })
        
        return result
    
    def _create_chunks(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Create chunks of specified size with overlap"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Find the end of the chunk
            end = start + chunk_size
            
            # Adjust end to not cut words
            if end < text_length:
                # Find the last space before the end
                while end > start and text[end] != ' ':
                    end -= 1
                
                # If no space found, just use the chunk_size
                if end == start:
                    end = start + chunk_size
            else:
                end = text_length
            
            # Add the chunk
            chunks.append(text[start:end])
            
            # Move to the next chunk with overlap
            start = end - chunk_overlap
        
        return chunks
    
    def retrieve_with_hierarchy(self, query: str, vector_db, k: int = 3) -> Dict[str, List[Any]]:
        """Retrieve documents with hierarchical context"""
        # First retrieve the most relevant child chunks
        child_results = vector_db.similarity_search(
            query, 
            filter={"chunk_type": "child"},
            k=k
        )
        
        # Get the parent IDs of the retrieved children
        parent_ids = [doc.metadata.get("parent_id") for doc in child_results if "parent_id" in doc.metadata]
        
        # Retrieve the parent chunks
        parent_results = []
        if parent_ids:
            parent_results = vector_db.similarity_search(
                query,
                filter={"chunk_id": {"$in": parent_ids}},
                k=len(parent_ids)
            )
        
        return {
            "children": child_results,
            "parents": parent_results
        }