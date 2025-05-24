#!/usr/bin/env python3
"""
Test script for the Multimodal RAG system.
This script tests the basic functionality of the system.
"""

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.multimodal_rag import MultimodalRAG

def main():
    parser = argparse.ArgumentParser(description="Test the Multimodal RAG system")
    parser.add_argument("--document", "-d", help="Path to a document file", required=False)
    parser.add_argument("--image", "-i", help="Path to an image file", required=False)
    parser.add_argument("--question", "-q", help="Question to ask", default="What is shown in the image and how does it relate to the document?")
    args = parser.parse_args()
    
    # Check if at least one document or image is provided
    if not args.document and not args.image:
        print("Error: At least one document or image must be provided")
        parser.print_help()
        return 1
    
    # Prepare paths
    document_paths = [args.document] if args.document else []
    image_paths = [args.image] if args.image else []
    
    # Initialize RAG system
    print(f"Initializing Multimodal RAG with {len(document_paths)} documents and {len(image_paths)} images...")
    rag = MultimodalRAG(document_paths=document_paths, image_paths=image_paths)
    
    # Process documents
    if document_paths:
        print("Processing documents...")
        num_text_chunks = rag.process_documents()
        print(f"Processed {num_text_chunks} text chunks from {len(document_paths)} documents")
    
    # Process images
    if image_paths:
        print("Processing images...")
        num_images = rag.process_images()
        print(f"Processed {num_images} images")
    
    # Initialize vector database
    print("Initializing vector database...")
    rag.initialize_vector_db()
    
    # Ask question with RAG
    print(f"\nQuestion: {args.question}")
    print("Generating answer with RAG...")
    answer = rag.answer_question(args.question, use_rag=True)
    print(f"Answer: {answer}")
    
    # Ask question without RAG
    print("\nGenerating answer without RAG...")
    answer_no_rag = rag.answer_question(args.question, use_rag=False)
    print(f"Answer (No RAG): {answer_no_rag}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())