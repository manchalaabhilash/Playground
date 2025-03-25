import sys
sys.path.append('../src')  # Add the src directory to the Python path

from data_processing import DataLoader, DataSplitter
from embedding import EmbeddingModel
from vector_database import VectorDB
from retriever import DocumentRetriever
from llm_interaction import LocalLLM

# --- Data Preparation ---
file_path = "../data/your_medical_manuals.pdf"  # Assuming you have a 'data' folder

# Load Data
data_loader = DataLoader(file_path)
documents = data_loader.load_data()

# Split Data
data_splitter = DataSplitter()
chunks = data_splitter.split_data(documents)

# Embed and Store in Vector Database
embedding_model = EmbeddingModel()
vector_db = VectorDB(embedding_function=embedding_model.embeddings)
vector_db.create_and_store(chunks)

# --- Define Retriever ---
retriever = vector_db.get_retriever()

# --- Load LLM ---
llm = LocalLLM()

# --- Question Answering using RAG ---
questions = [
    "What is the protocol for managing sepsis in a critical care unit?",
    "What are the common symptoms of appendicitis, and can it be cured via medicine? If not, what surgical procedure should be followed to treat it?",
    # ... other questions
]

print("Answers using RAG:")
for question in questions:
    relevant_docs = retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    prompt = f"Based on the following medical context, answer the question: '{question}'\n\nContext:\n{context}\n\nAnswer:"
    response = llm.generate_response(prompt)
    print(f"\nQuestion: {question}")
    print(f"Answer: {response}")
    print("-" * 50)

# --- Question Answering using LLM (without RAG) ---
print("\nAnswers using LLM directly:")
for question in questions:
    response = llm.generate_response(question, system_message="Answer the question directly in 2-3 sentences.")
    print(f"\nQuestion: {question}")
    print(f"Answer: {response}")
    print("-" * 50)

# --- Prompt Engineering with RAG (Example) ---
print("\nAnswers using RAG with Prompt Engineering:")
for question in questions:
    relevant_docs = retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    engineered_prompt = f"As a medical expert, based on the provided context, what is the recommended approach for '{question}'? Explain concisely in 3-4 sentences.\n\nContext:\n{context}\n\nAnswer:"
    response = llm.generate_response(engineered_prompt)
    print(f"\nQuestion: {question}")
    print(f"Answer: {response}")
    print("-" * 50)