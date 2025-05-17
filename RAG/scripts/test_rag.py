import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag import RAG

# List your ML textbook PDFs
pdf_paths = [
    "path/to/your/ml_textbook1.pdf",
    "path/to/your/ml_textbook2.pdf"
]

# Initialize RAG system
rag = RAG(pdf_paths)

# Load and process documents
num_chunks = rag.load_and_process_documents()
print(f"Processed {num_chunks} chunks from {len(pdf_paths)} PDFs")

# Test questions
questions = [
    "What is gradient descent?",
    "Explain the difference between supervised and unsupervised learning",
    "How does a decision tree work?",
]

# Get answers with RAG
print("\n--- Answers with RAG ---")
for question in questions:
    print(f"\nQuestion: {question}")
    answer = rag.answer_question(question, use_rag=True)
    print(f"Answer: {answer}")

# Get answers without RAG (direct LLM)
print("\n--- Answers without RAG ---")
for question in questions:
    print(f"\nQuestion: {question}")
    answer = rag.answer_question(question, use_rag=False)
    print(f"Answer: {answer}")

# Compare the differences
print("\nComparing answers with and without RAG shows how retrieval augmentation")
print("improves the quality and accuracy of responses for domain-specific questions.")
