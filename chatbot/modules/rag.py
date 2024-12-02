from .weaviate_knowledge import WeaviateKnowledge
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RAGModule:
    def __init__(self, weaviate_url, model_name='all-MiniLM-L6-v2'):
        self.knowledge = WeaviateKnowledge(weaviate_url)
        self.model = SentenceTransformer(model_name)
        logging.info(f"Initialized RAG module with Weaviate URL: {weaviate_url}")

    def retrieve_relevant_data(self, query):
        try:
            embedding = self.model.encode(query)
            results = self.knowledge.query_database(query)
            logging.info(f"Retrieved {len(results)} relevant items for query: {query}")
            return results
        except Exception as e:
            logging.error(f"Error retrieving relevant data: {str(e)}")
            return []

    def generate_response(self, query, relevant_data, intent):
        # Handle vague queries with a follow-up question
        if len(query.split()) < 2:
            return "Could you please provide more details about what you'd like me to show?"

        if not relevant_data:
            return "I'm sorry, I couldn't find any relevant information."

        # Generate a detailed response using the retrieved data
        response = f"Here are some items I found for '{query}':\n"
        for i, item in enumerate(relevant_data[:3], 1):  # Limit to 3 items for brevity
            review_text = item.get('cleaned_reviewText', 'No review text available')
            response += f"{i}. {review_text[:100]}...\n"
        response += "Let me know if you'd like more details on any of these!"
        logging.info(f"Generated detailed response for query: {query}")
        return response
