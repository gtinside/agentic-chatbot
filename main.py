import json
import argparse
from embeddings.vector import Vector
from classification.classifier import Classifier, Intent
from llm.openai_requests import OpenAIRequests

def process_document(v: Vector):
    """
    Process the document available in the assets/docs directory.
    This function initializes chromadb and leverages llamindex to process the document, generate embeddings and store them in the ChromaVectorStore.
    Args:
        v (Vector): The Vector object.
    Returns:
        None
    """
    v.process_document()
    print("Document processed successfully")

def classify_query_type(query: str) -> Intent:
    return Classifier.classify_query_type(query)

def classify_document_type(query: str, metadata: dict):
    return Classifier.classify_document_type(query, metadata)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", help="The query string")
    args = parser.parse_args()
    query = args.query
    metadata = {}
    with open("assets/metadata.json", "r") as f:
        data = json.load(f)
        for item in data['metadata']:
            metadata[item["file_name"]] = item["attributes"]
    v = Vector("assets/docs", metadata=metadata)
    # Step 1: Process the document and store them in chromadb
    process_document(v)
    
    # Step 2: Classify the query type
    query_type = classify_query_type(query)
    print("The query type is: ", query_type)
    
    # Step 3: Classify the document type
    document_type = classify_document_type(query, metadata)
    print(f"{document_type} will be queried.")

    # Step 4: Prepare the context to be sent to OpenAI from chromadb
    rag_context = v.query_document(query, query_type, document_type)

    # Step 5: Send the context to OpenAI with the user question and get the response
    response = OpenAIRequests.send_request_with_context(rag_context, query)
    print(response)

if __name__ == "__main__":
    main()

