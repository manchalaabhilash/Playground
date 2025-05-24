import os
from typing import List, Dict, Any
import pytesseract
from PIL import Image
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import OCR_ENABLED

class DocumentProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    def process_documents(self, document_paths):
        """Process multiple documents and return chunks"""
        all_chunks = []
        
        for doc_path in document_paths:
            if not os.path.exists(doc_path):
                print(f"Warning: Document {doc_path} not found, skipping")
                continue
                
            try:
                if doc_path.lower().endswith('.pdf'):
                    loader = PyPDFLoader(doc_path)
                elif doc_path.lower().endswith('.txt'):
                    loader = TextLoader(doc_path)
                else:
                    print(f"Warning: Unsupported file format for {doc_path}, skipping")
                    continue
                
                documents = loader.load()
                chunks = self.text_splitter.split_documents(documents)
                all_chunks.extend(chunks)
                
            except Exception as e:
                print(f"Error processing document {doc_path}: {str(e)}")
        
        return all_chunks

class ImageProcessor:
    def __init__(self):
        self.ocr_enabled = OCR_ENABLED
    
    def process_images(self, image_paths):
        """Process multiple images and return image chunks with metadata"""
        image_chunks = []
        
        for img_path in image_paths:
            if not os.path.exists(img_path):
                print(f"Warning: Image {img_path} not found, skipping")
                continue
                
            try:
                # Load image
                img = Image.open(img_path)
                
                # Extract text using OCR if enabled
                ocr_text = ""
                if self.ocr_enabled:
                    try:
                        ocr_text = pytesseract.image_to_string(img)
                    except Exception as e:
                        print(f"OCR error for {img_path}: {str(e)}")
                
                # Create image chunk with metadata
                image_chunk = {
                    "image_path": img_path,
                    "image_filename": os.path.basename(img_path),
                    "ocr_text": ocr_text,
                    "width": img.width,
                    "height": img.height,
                    "format": img.format
                }
                
                image_chunks.append(image_chunk)
                
            except Exception as e:
                print(f"Error processing image {img_path}: {str(e)}")
        
        return image_chunks