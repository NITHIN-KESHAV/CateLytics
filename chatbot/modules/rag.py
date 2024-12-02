from .weaviate_knowledge import WeaviateKnowledge
from .summarizer import Summarizer
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RAGModule:
    def __init__(self, weaviate_url, model_name='all-MiniLM-L6-v2'):
        self.knowledge = WeaviateKnowledge(weaviate_url)
        self.model = SentenceTransformer(model_name)
        self.summarizer = Summarizer()
        logging.info(f"Initialized RAG module with Weaviate URL: {weaviate_url}")

    def retrieve_relevant_data(self, query=None, asin=None):
        try:
            results = self.knowledge.query_database(query=query, asin=asin)
            logging.info(f"Retrieved {len(results)} relevant items for query: '{query}', ASIN: '{asin}'")
            return results
        except Exception as e:
            logging.error(f"Error retrieving relevant data: {str(e)}")
            return []

    def generate_response(self, query, relevant_data, intent):
        if not relevant_data:
            return f"I'm sorry, I couldn't find any information for your query: '{query}'"

        response = f"Here are some items I found for '{query}':\n"
        for i, item in enumerate(relevant_data[:3], 1):  # Show up to 3 results
            review_text = item.get('cleaned_reviewText', 'No review text available').replace("\n", " ").strip()
            overall_rating = item.get('overall', 'N/A')
            asin = item.get('asin', 'N/A')
            summary = self.summarizer.summarize(review_text)
            response += f"{i}. Summary: {summary} | Rating: {overall_rating}/5 | ASIN: {asin}\n"
        response += "Let me know if you'd like more details on any of these!"
        logging.info(f"Generated detailed response for query: '{query}'")
        return response

    def highlight_best_product(self, relevant_data):
        if not relevant_data:
            return "I'm sorry, I couldn't find any products to recommend."

        highest_rated = max(relevant_data, key=lambda x: x.get("overall", 0))
        review_text = highest_rated.get("cleaned_reviewText", "No review text available").replace("\n", " ").strip()
        overall_rating = highest_rated.get("overall", "N/A")
        asin = highest_rated.get("asin", "N/A")
        summary = self.summarizer.summarize(review_text)
        response = (
            f"The best-rated product has a rating of {overall_rating}/5. "
            f"ASIN: {asin}. "
            f"Here’s a summary: {summary}"
        )
        logging.info("Generated best product recommendation.")
        return response
