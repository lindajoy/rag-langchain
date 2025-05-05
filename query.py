#!/usr/bin/env python3
"""
Harry Potter Gen Z - Command Line Query Tool

This script provides a command-line interface to query the Harry Potter RAG system.
It can be used to ask questions about Harry Potter and get answers in Gen Z style.
"""
import argparse
import os
import time
from typing import List, Optional
from dotenv import load_dotenv

# Langchain imports
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate

# Gemini imports
import google.generativeai as genai

# Import the RAG system from main.py
try:
    from main import HarryPotterRAG
except ImportError:
    print("Error: Could not import HarryPotterRAG from main.py")
    print("Make sure main.py is in the same directory as query.py")
    exit(1)

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")
genai.configure(api_key=GEMINI_API_KEY)

# Constants
CHROMA_PATH = "chroma"
DATA_PATH = "book_1"  # Default to book_1, can be changed to "data/books" or other directories

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Query the Harry Potter RAG system from the command line"
    )
    parser.add_argument(
        "query", 
        nargs="?", 
        type=str, 
        help="The query text (if not provided, interactive mode will be used)"
    )
    parser.add_argument(
        "--data-path", 
        type=str, 
        default=DATA_PATH,
        help=f"Path to the data directory (default: {DATA_PATH})"
    )
    parser.add_argument(
        "--k", 
        type=int, 
        default=3,
        help="Number of documents to retrieve (default: 3)"
    )
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="Run in interactive mode"
    )

    return parser.parse_args()

def direct_query_with_gemini(query_text: str, context: str) -> str:
    """Generate a response using Gemini Pro directly (without Langchain)"""
    try:
        # Create prompt
        prompt = f"""
        You are a helpful assistant that speaks in Gen Z slang and internet language.
        Answer the question based only on the following context:

        {context}

        ---

        Question: {query_text}

        Answer in Gen Z style with slang, internet language, and emojis:
        """

        # Generate response
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Error generating response with Gemini: {str(e)}"

def main():
    """Main function"""
    args = parse_arguments()

    # Initialize RAG system
    rag = HarryPotterRAG(data_path=args.data_path)

    # Create index if it doesn't exist
    if not os.path.exists(CHROMA_PATH):
        print("Creating new vector index...")
        rag.create_index()
    else:
        print("Using existing vector index...")
        rag.load_vectorstore()

    # Interactive mode
    if args.interactive or not args.query:
        print("\n✨ Ready to answer your Harry Potter questions in Gen Z style! ✨")
        print("Type 'exit' to quit")

        while True:
            query = input("\nYour question: ")
            if query.lower() in ["exit", "quit", "q"]:
                break

            process_query(rag, query, args.k)

    # Single query mode
    else:
        process_query(rag, args.query, args.k)

def process_query(rag, query_text, k):
    """Process a single query"""
    try:
        start_time = time.time()

        # Get relevant documents
        if not rag.vectorstore:
            rag.load_vectorstore()

        # Use the RAG system to get a response
        result = rag.query(query_text, k=k)
        response_text = result["response"]
        docs = result["source_documents"]

        end_time = time.time()

        # Print response
        print("\n" + "=" * 80)
        print(f"🔮 Answer (in {end_time - start_time:.2f}s):")
        print(response_text)
        print("=" * 80)

        # Show sources
        show_sources = input("\nShow sources? (y/n): ").lower() == "y"
        if show_sources:
            print("\nSources:")
            for i, doc in enumerate(docs):
                print(f"\nSource {i+1}:")
                print(f"From: {doc.metadata.get('source', 'Unknown')}")
                print(f"Content: {doc.page_content[:200]}...")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
