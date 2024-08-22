# main.py
import os
import sys
from document_processor import load_documents
from embedding_generator import generate_embeddings
from search_engine import SearchEngine
from response_generator import ResponseGenerator
from config import Config
from utils import setup_logger

logger = setup_logger('main')

def initialize_system():
    """Initialize the search and response system."""
    logger.info("Initializing the system...")
    
    # Load and process documents
    documents = load_documents(Config.DOCUMENT_FOLDER)
    logger.info(f"Loaded {len(documents)} documents")

    # Generate embeddings
    embeddings = generate_embeddings(documents)
    logger.info(f"Generated embeddings of shape {embeddings.shape}")

    # Initialize search engine
    search_engine = SearchEngine(documents, embeddings)
    logger.info("Search engine initialized")

    # Initialize response generator
    response_generator = ResponseGenerator()
    logger.info("Response generator initialized")

    return search_engine, response_generator

def process_query(query, search_engine, response_generator):
    """Process a single query and return the response."""
    relevant_docs = search_engine.search(query)
    logger.info(f"Found {len(relevant_docs)} relevant documents")

    response = response_generator.generate_response(query, relevant_docs)
    return response

def interactive_mode(search_engine, response_generator):
    """Run the system in interactive mode, processing queries from user input."""
    print("Enter your queries. Type 'quit' to exit.")
    while True:
        query = input("Query: ").strip()
        if query.lower() == 'quit':
            break
        
        response = process_query(query, search_engine, response_generator)
        print(f"Response: {response}\n")

def batch_mode(input_file, output_file, search_engine, response_generator):
    """Process queries from an input file and write responses to an output file."""
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            query = line.strip()
            response = process_query(query, search_engine, response_generator)
            outfile.write(f"Query: {query}\nResponse: {response}\n\n")
    logger.info(f"Batch processing completed. Results written to {output_file}")

def main():
    search_engine, response_generator = initialize_system()

    if len(sys.argv) > 1:
        if sys.argv[1] == '--batch':
            if len(sys.argv) != 4:
                print("Usage for batch mode: python main.py --batch input_file output_file")
                sys.exit(1)
            batch_mode(sys.argv[2], sys.argv[3], search_engine, response_generator)
        else:
            print("Unknown argument. Use --batch for batch mode or no arguments for interactive mode.")
            sys.exit(1)
    else:
        interactive_mode(search_engine, response_generator)

if __name__ == "__main__":
    main()