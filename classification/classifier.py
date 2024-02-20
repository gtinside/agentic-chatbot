from enum import Enum
from llm.openai_requests import OpenAIRequests

class Intent(Enum):
    SUMMARIZATION = "SUMMARIZATION"
    GENERAL = "GENERAL"

class Classifier:
    @staticmethod
    def classify_query_type(query):
        """
        Classify the query type using the specified query.

        Args:
            query (str): The query string.

        Returns:
            type: The type of the query.
        """
        prompt = "Given the input message, determine its intent category. If the message explicitly requests a summary of the document, " \
                 "respond with 'SUMMARIZATION'; otherwise, respond with 'GENERAL'. Do not provide additional commentary."
        response = OpenAIRequests.send_request(prompt, query)
        for intent in (Intent.SUMMARIZATION, Intent.GENERAL):
            if intent.value == str(response).strip():
                return intent
    
    @staticmethod
    def classify_document_type(query, metadata):
        """
        Classify the document type using the specified query and type.

        Args:
            query (str): The query string.
            attributes (dict): The metadata attributes.

        Returns:
            response: The type of document
        """
        for attribute in metadata.values():
            if attribute in query:
                return attribute