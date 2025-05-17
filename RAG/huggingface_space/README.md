---
title: ML Textbook RAG
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
---

# ML Textbook RAG System

A Retrieval-Augmented Generation (RAG) system for answering questions about machine learning concepts using your own textbooks.

## Features

- Upload and process machine learning textbook PDFs
- Split documents into chunks and create embeddings
- Store embeddings in a vector database
- Retrieve relevant context for questions
- Generate answers using Ollama LLM with retrieved context
- Web UI for easy interaction

## How to Use

1. Upload your machine learning textbook PDFs using the file uploader
2. Click "Initialize RAG System" to process the PDFs
3. Ask questions about machine learning concepts in the text input
4. Get answers based on your textbooks!

## Technical Details

This application uses:
- Ollama for local LLM inference
- LangChain for the RAG pipeline
- Streamlit for the web interface
- Chroma for the vector database