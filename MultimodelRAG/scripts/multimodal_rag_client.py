#!/usr/bin/env python3
"""
Client script for the Multimodal RAG API.
This script allows you to interact with the API from the command line.
"""

import argparse
import requests
import json
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Client for the Multimodal RAG API")
    parser.add_argument("--api-url", default="http://localhost:5000", help="URL of the Multimodal RAG API")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Initialize command
    init_parser = subparsers.add_parser("initialize", help="Initialize the RAG system")
    init_parser.add_argument("--documents", "-d", nargs="+", help="Paths to document files")
    init_parser.add_argument("--images", "-i", nargs="+", help="Paths to image files")
    
    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question", help="Question to ask")
    ask_parser.add_argument("--no-rag", action="store_true", help="Don't use RAG (direct LLM)")
    
    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload files")
    upload_parser.add_argument("--documents", "-d", nargs="+", help="Paths to document files")
    upload_parser.add_argument("--images", "-i", nargs="+", help="Paths to image files")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    if args.command == "initialize":
        return initialize(args.api_url, args.documents, args.images)
    elif args.command == "ask":
        return ask(args.api_url, args.question, not args.no_rag)
    elif args.command == "upload":
        return upload(args.api_url, args.documents, args.images)
    
    return 0

def initialize(api_url, document_paths, image_paths):
    """Initialize the RAG system"""
    document_paths = document_paths or []
    image_paths = image_paths or []
    
    if not document_paths and not image_paths:
        print("Error: At least one document or image must be provided")
        return 1
    
    # Validate paths
    valid_document_paths = [path for path in document_paths if os.path.exists(path)]
    valid_image_paths = [path for path in image_paths if os.path.exists(path)]
    
    if not valid_document_paths and not valid_image_paths:
        print("Error: No valid document or image paths provided")
        return 1
    
    print(f"Initializing RAG system with {len(valid_document_paths)} documents and {len(valid_image_paths)} images...")
    
    try:
        response = requests.post(
            f"{api_url}/initialize",
            json={"document_paths": valid_document_paths, "image_paths": valid_image_paths}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success: RAG system initialized with {result['num_text_chunks']} text chunks from {result['num_documents']} documents and {result['num_image_chunks']} images")
            return 0
        else:
            print(f"Error: {response.text}")
            return 1
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

def ask(api_url, question, use_rag):
    """Ask a question"""
    print(f"Question: {question}")
    print(f"Using RAG: {use_rag}")
    
    try:
        response = requests.post(
            f"{api_url}/ask",
            json={"question": question, "use_rag": use_rag}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nAnswer: {result['answer']}")
            return 0
        else:
            print(f"Error: {response.text}")
            return 1
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

def upload(api_url, document_paths, image_paths):
    """Upload files"""
    document_paths = document_paths or []
    image_paths = image_paths or []
    
    if not document_paths and not image_paths:
        print("Error: At least one document or image must be provided")
        return 1
    
    # Validate paths
    valid_document_paths = [path for path in document_paths if os.path.exists(path)]
    valid_image_paths = [path for path in image_paths if os.path.exists(path)]
    
    if not valid_document_paths and not valid_image_paths:
        print("Error: No valid document or image paths provided")
        return 1
    
    print(f"Uploading {len(valid_document_paths)} documents and {len(valid_image_paths)} images...")
    
    try:
        files = []
        
        # Add documents
        for path in valid_document_paths:
            files.append(("documents", (os.path.basename(path), open(path, "rb"))))
        
        # Add images
        for path in valid_image_paths:
            files.append(("images", (os.path.basename(path), open(path, "rb"))))
        
        response = requests.post(f"{api_url}/upload", files=files)
        
        # Close all file handles
        for _, (_, file_obj) in files:
            file_obj.close()
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success: Uploaded {len(result['uploaded_files'])} files")
            if result['errors']:
                print(f"Errors: {json.dumps(result['errors'], indent=2)}")
            return 0
        else:
            print(f"Error: {response.text}")
            return 1
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())