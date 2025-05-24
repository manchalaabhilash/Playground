"""
Multimodal retrieval implementation for RAG systems.
Provides retrieval capabilities for both text and image content.
"""

from typing import List, Dict, Any, Optional, Union
import numpy as np
import re
import base64
from io import BytesIO
from PIL import Image

class MultimodalRetriever:
    """
    Retrieves both text and image content based on queries.
    Implements unified retrieval across modalities.
    """
    
    def __init__(self, text_retriever=None, image_retriever=None, fusion_strategy: str = "linear"):
        """Initialize the multimodal retriever"""
        self.text_retriever = text_retriever
        self.image_retriever = image_retriever
        self.fusion_strategy = fusion_strategy  # linear, max, or weighted
        self.text_weight = 0.7
        self.image_weight = 0.3
    
    def get_relevant_documents(self, query: str, k: int = 5, modality: str = "auto") -> List[Any]:
        """Get relevant documents across modalities"""
        # Determine which retrievers to use based on modality
        use_text = modality in ["auto", "text"]
        use_image = modality in ["auto", "image"]
        
        # Check if query is image-focused
        is_image_query = self._is_image_query(query)
        
        # Adjust weights based on query
        if is_image_query and modality == "auto":
            self.text_weight = 0.3
            self.image_weight = 0.7
        else:
            self.text_weight = 0.7
            self.image_weight = 0.3
        
        # Get text documents if applicable
        text_docs = []
        if use_text and self.text_retriever:
            text_docs = self.text_retriever.get_relevant_documents(query, k=k*2)
        
        # Get image documents if applicable
        image_docs = []
        if use_image and self.image_retriever:
            image_docs = self.image_retriever.get_relevant_documents(query, k=k*2)
        
        # If only one modality has results, return those
        if not text_docs and image_docs:
            return image_docs[:k]
        elif text_docs and not image_docs:
            return text_docs[:k]
        elif not text_docs and not image_docs:
            return []
        
        # Combine results based on fusion strategy
        if self.fusion_strategy == "linear":
            combined_docs = self._linear_fusion(text_docs, image_docs, k)
        elif self.fusion_strategy == "max":
            combined_docs = self._max_fusion(text_docs, image_docs, k)
        else:  # weighted
            combined_docs = self._weighted_fusion(text_docs, image_docs, k)
        
        return combined_docs
    
    def _is_image_query(self, query: str) -> bool:
        """Determine if a query is focused on images"""
        image_keywords = ["image", "picture", "photo", "visual", "see", "look", "show", "display", 
                         "diagram", "graph", "chart", "figure", "illustration"]
        
        query_lower = query.lower()
        
        # Check for image-related keywords
        for keyword in image_keywords:
            if keyword in query_lower:
                return True
        
        return False
    
    def _linear_fusion(self, text_docs: List[Any], image_docs: List[Any], k: int) -> List[Any]:
        """Combine results using linear fusion"""
        # Create a dictionary to track document scores
        doc_scores = {}
        doc_map = {}
        
        # Process text documents
        for i, doc in enumerate(text_docs):
            doc_id = self._get_doc_id(doc)
            # Normalize score based on position (higher position = higher score)
            score = self.text_weight * (1.0 - (i / len(text_docs) if text_docs else 1.0))
            doc_scores[doc_id] = score
            doc_map[doc_id] = doc
        
        # Process image documents
        for i, doc in enumerate(image_docs):
            doc_id = self._get_doc_id(doc)
            # Normalize score based on position
            score = self.image_weight * (1.0 - (i / len(image_docs) if image_docs else 1.0))
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
    
    def _max_fusion(self, text_docs: List[Any], image_docs: List[Any], k: int) -> List[Any]:
        """Combine results using max fusion"""
        # Create a dictionary to track document scores
        doc_scores = {}
        doc_map = {}
        
        # Process text documents
        for i, doc in enumerate(text_docs):
            doc_id = self._get_doc_id(doc)
            # Normalize score based on position (higher position = higher score)
            score = self.text_weight * (1.0 - (i / len(text_docs) if text_docs else 1.0))
            doc_scores[doc_id] = score
            doc_map[doc_id] = doc
        
        # Process image documents
        for i, doc in enumerate(image_docs):
            doc_id = self._get_doc_id(doc)
            # Normalize score based on position
            score = self.image_weight * (1.0 - (i / len(image_docs) if image_docs else 1.0))
            # Take max score if document already seen
            if doc_id in doc_scores:
                doc_scores[doc_id] = max(doc_scores[doc_id], score)
            else:
                doc_scores[doc_id] = score
                doc_map[doc_id] = doc
        
        # Sort documents by score
        sorted_doc_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        
        # Return top k documents
        return [doc_map[doc_id] for doc_id in sorted_doc_ids[:k]]
    
    def _weighted_fusion(self, text_docs: List[Any], image_docs: List[Any], k: int) -> List[Any]:
        """Combine results using weighted fusion"""
        # Determine number of documents to take from each modality
        text_k = int(k * self.text_weight)
        image_k = k - text_k
        
        # Ensure at least one document from each modality if available
        if text_k == 0 and text_docs:
            text_k = 1
            image_k = k - 1
        elif image_k == 0 and image_docs:
            image_k = 1
            text_k = k - 1
        
        # Get top documents from each modality
        top_text_docs = text_docs[:text_k] if text_docs else []
        top_image_docs = image_docs[:image_k] if image_docs else []
        
        # Combine results
        combined_docs = top_text_docs + top_image_docs
        
        # If not enough documents, add more from either modality
        if len(combined_docs) < k:
            remaining_text = text_docs[text_k:] if text_docs and text_k < len(text_docs) else []
            remaining_images = image_docs[image_k:] if image_docs and image_k < len(image_docs) else []
            
            # Interleave remaining documents
            remaining = []
            for i in range(max(len(remaining_text), len(remaining_images))):
                if i < len(remaining_text):
                    remaining.append(remaining_text[i])
                if i < len(remaining_images):
                    remaining.append(remaining_images[i])
            
            # Add remaining documents up to k
            combined_docs.extend(remaining[:k - len(combined_docs)])
        
        return combined_docs
    
    def _get_doc_id(self, doc: Any) -> str:
        """Get a unique identifier for a document"""
        if hasattr(doc, "id"):
            return doc.id
        elif isinstance(doc, dict) and "id" in doc:
            return doc["id"]
        else:
            # Use object id as fallback
            return str(id(doc))


class ImageRetriever:
    """
    Specialized retriever for image content.
    Implements image retrieval using embeddings and metadata.
    """
    
    def __init__(self, vector_store=None, embedding_model=None):
        """Initialize the image retriever"""
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.images = []
        self.image_embeddings = []
    
    def add_images(self, images: List[Dict[str, Any]]):
        """Add images to the retriever"""
        self.images.extend(images)
        
        # Generate embeddings if model available
        if self.embedding_model and not self.vector_store:
            # Extract image data or paths
            image_data = []
            for img in images:
                if "image_path" in img:
                    try:
                        with open(img["image_path"], "rb") as f:
                            image_data.append(Image.open(BytesIO(f.read())))
                    except Exception as e:
                        print(f"Error loading image {img['image_path']}: {str(e)}")
                        image_data.append(None)
                elif "image_data" in img:
                    try:
                        image_data.append(Image.open(BytesIO(img["image_data"])))
                    except Exception as e:
                        print(f"Error loading image data: {str(e)}")
                        image_data.append(None)
                else:
                    image_data.append(None)
            
            # Generate embeddings for valid images
            for i, img in enumerate(image_data):
                if img is not None:
                    try:
                        embedding = self.embedding_model.encode(img)
                        self.image_embeddings.append(embedding)
                    except Exception as e:
                        print(f"Error generating embedding for image: {str(e)}")
                        # Use zero embedding as fallback
                        self.image_embeddings.append(np.zeros(512))
                else:
                    # Use zero embedding for invalid images
                    self.image_embeddings.append(np.zeros(512))
    
    def get_relevant_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Get relevant images based on the query"""
        # If using vector store
        if self.vector_store:
            try:
                results = self.vector_store.similarity_search(query, k=k, filter={"type": "image"})
                return results
            except Exception as e:
                print(f"Error in vector store search: {str(e)}")
                return []
        
        # If no images or embeddings, return empty list
        if not self.images or not self.image_embeddings:
            return []
        
        # If embedding model available, use it for query
        if self.embedding_model:
            try:
                # Generate query embedding
                query_embedding = self.embedding_model.encode(query)
                
                # Calculate similarities
                similarities = []
                for i, img_embedding in enumerate(self.image_embeddings):
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(query_embedding, img_embedding)
                    similarities.append((i, similarity))
                
                # Sort by similarity
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                # Return top k images
                return [self.images[i] for i, _ in similarities[:k]]
            except Exception as e:
                print(f"Error in embedding similarity search: {str(e)}")
        
        # Fallback to keyword matching on image metadata
        return self._keyword_search(query, k)
    
    def _keyword_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Search images based on keyword matching in metadata"""
        # Extract query keywords
        query_keywords = set(self._tokenize(query))
        
        # Score images based on metadata matches
        scores = []
        for i, img in enumerate(self.images):
            score = 0
            
            # Check caption
            if "caption" in img:
                caption_tokens = set(self._tokenize(img["caption"]))
                caption_overlap = len(query_keywords.intersection(caption_tokens))
                score += caption_overlap * 2  # Weight caption matches higher
            
            # Check OCR text
            if "ocr_text" in img:
                ocr_tokens = set(self._tokenize(img["ocr_text"]))
                ocr_overlap = len(query_keywords.intersection(ocr_tokens))
                score += ocr_overlap
            
            # Check other metadata
            for key, value in img.items():
                if key not in ["caption", "ocr_text", "image_path", "image_data"] and isinstance(value, str):
                    metadata_tokens = set(self._tokenize(value))
                    metadata_overlap = len(query_keywords.intersection(metadata_tokens))
                    score += metadata_overlap
            
            scores.append((i, score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k images
        return [self.images[i] for i, _ in scores[:k] if scores[i][1] > 0]
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        if not isinstance(text, str):
            return []
        return re.findall(r'\w+', text.lower())
    
    def _cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class MultimodalFormatter:
    """
    Formats multimodal content for LLM consumption.
    Implements formatting strategies for text and images.
    """
    
    def __init__(self, image_format: str = "base64"):
        """Initialize the formatter"""
        self.image_format = image_format  # base64, url, or description
    
    def format_documents(self, documents: List[Any]) -> str:
        """Format documents for LLM consumption"""
        formatted_content = []
        
        for doc in documents:
            # Determine document type
            doc_type = self._get_document_type(doc)
            
            if doc_type == "text":
                # Format text document
                formatted_content.append(self._format_text_document(doc))
            elif doc_type == "image":
                # Format image document
                formatted_content.append(self._format_image_document(doc))
            else:
                # Unknown document type
                formatted_content.append(f"[Unknown document type: {doc_type}]")
        
        # Join formatted content
        return "\n\n".join(formatted_content)
    
    def _get_document_type(self, doc: Any) -> str:
        """Determine the type of a document"""
        if isinstance(doc, dict):
            if "image_path" in doc or "image_data" in doc:
                return "image"
            elif "content" in doc:
                return "text"
        
        # Check for common text document attributes
        if hasattr(doc, "page_content"):
            return "text"
        
        # Default to text
        return "text"
    
    def _format_text_document(self, doc: Any) -> str:
        """Format a text document"""
        # Extract text content
        if hasattr(doc, "page_content"):
            content = doc.page_content
        elif isinstance(doc, dict) and "content" in doc:
            content = doc["content"]
        else:
            content = str(doc)
        
        # Extract metadata if available
        metadata = {}
        if hasattr(doc, "metadata"):
            metadata = doc.metadata
        elif isinstance(doc, dict) and "metadata" in doc:
            metadata = doc["metadata"]
        
        # Format metadata
        metadata_str = ""
        if metadata:
            metadata_items = []
            for key, value in metadata.items():
                if key not in ["content", "page_content"]:
                    metadata_items.append(f"{key}: {value}")
            
            if metadata_items:
                metadata_str = f"[{', '.join(metadata_items)}]"
        
        # Combine content and metadata
        if metadata_str:
            return f"{metadata_str}\n{content}"
        else:
            return content
    
    def _format_image_document(self, doc: Any) -> str:
        """Format an image document"""
        # Extract image data or path
        image_data = None
        image_path = None
        
        if isinstance(doc, dict):
            if "image_data" in doc:
                image_data = doc["image_data"]
            elif "image_path" in doc:
                image_path = doc["image_path"]
        
        # Extract metadata
        metadata = {}
        if isinstance(doc, dict):
            metadata = {k: v for k, v in doc.items() if k not in ["image_data", "image_path"]}
        
        # Format based on image format
        if self.image_format == "base64" and (image_data or image_path):
            # Get image data
            if image_data:
                img_data = image_data
            else:
                try:
                    with open(image_path, "rb") as f:
                        img_data = f.read()
                except Exception as e:
                    return f"[Error loading image: {str(e)}]"
            
            # Convert to base64
            try:
                base64_str = base64.b64encode(img_data).decode("utf-8")
                img_format = "jpeg"  # Default format
                
                # Try to determine format
                if image_path:
                    ext = image_path.split(".")[-1].lower()
                    if ext in ["png", "jpg", "jpeg", "gif"]:
                        img_format = "png" if ext == "png" else "jpeg"
                
                # Format as markdown image with base64
                img_markdown = f"![Image](data:image/{img_format};base64,{base64_str})"
                
                # Add metadata
                metadata_str = self._format_image_metadata(metadata)
                
                return f"{img_markdown}\n{metadata_str}"
            except Exception as e:
                return f"[Error encoding image: {str(e)}]"
        
        elif self.image_format == "url" and image_path:
            # Format as markdown image with URL
            img_markdown = f"![Image]({image_path})"
            
            # Add metadata
            metadata_str = self._format_image_metadata(metadata)
            
            return f"{img_markdown}\n{metadata_str}"
        
        else:
            # Format as description
            return self._format_image_metadata(metadata)
    
    def _format_image_metadata(self, metadata: Dict[str, Any]) -> str:
        """Format image metadata"""
        metadata_items = []
        
        # Add caption if available
        if "caption" in metadata:
            metadata_items.append(f"Caption: {metadata['caption']}")
        
        # Add OCR text if available
        if "ocr_text" in metadata:
            metadata_items.append(f"Text in image: {metadata['ocr_text']}")
        
        # Add other metadata
        for key, value in metadata.items():
            if key not in ["caption", "ocr_text"]:
                metadata_items.append(f"{key}: {value}")
        
        return "\n".join(metadata_items)
