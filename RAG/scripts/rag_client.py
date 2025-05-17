import requests
import json
import argparse

def initialize_rag(api_url, pdf_paths):
    response = requests.post(
        f"{api_url}/initialize",
        json={"pdf_paths": pdf_paths},
        headers={"Content-Type": "application/json"}
    )
    return response.json()

def ask_question(api_url, question, use_rag=True):
    response = requests.post(
        f"{api_url}/ask",
        json={"question": question, "use_rag": use_rag},
        headers={"Content-Type": "application/json"}
    )
    return response.json()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Textbook RAG Client")
    parser.add_argument("--api_url", default="http://localhost:5000", help="URL of the RAG API")
    parser.add_argument("--initialize", action="store_true", help="Initialize the RAG system")
    parser.add_argument("--pdf_paths", nargs="+", help="Paths to PDF files")
    parser.add_argument("--question", help="Question to ask")
    parser.add_argument("--no_rag", action="store_true", help="Don't use RAG (direct LLM)")
    
    args = parser.parse_args()
    
    if args.initialize:
        if not args.pdf_paths:
            print("Error: PDF paths required for initialization")
        else:
            result = initialize_rag(args.api_url, args.pdf_paths)
            print(json.dumps(result, indent=2))
    
    if args.question:
        result = ask_question(args.api_url, args.question, not args.no_rag)
        print("\nQuestion:", args.question)
        print("\nAnswer:", result.get("answer", "Error: " + result.get("error", "Unknown error")))