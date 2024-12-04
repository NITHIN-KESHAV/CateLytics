from modules.weaviate_knowledge import WeaviateKnowledge
from modules.summarizer import Summarizer
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class RAGModule:
    def __init__(self, weaviate_url, model_path="models/fine_tuned_t8"):
        self.knowledge = WeaviateKnowledge(weaviate_url)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")  # For semantic search
        self.summarizer = Summarizer(model_path=model_path)  # Use fine-tuned summarizer
        self.last_response = None
        logging.info(f"Initialized RAG module with Weaviate URL: {weaviate_url}")

    def retrieve_relevant_data(self, query=None, asin=None):
        try:
            results = self.knowledge.query_database(query=query, asin=asin)
            logging.info(f"Retrieved {len(results)} relevant items for query: '{query}', ASIN: '{asin}'")
            return results
        except Exception as e:
            logging.error(f"Error retrieving relevant data: {str(e)}")
            return []

    def generate_response(self, query, relevant_data, intent, asin=None):
        if self.last_response and asin:
            return f"I already shared the review summary for ASIN: {asin}."

        if not relevant_data:
            return f"I'm sorry, I couldn't find any information for your query: '{query}'"

        response = f"Here are some items I found for '{query}':\n"
        for i, item in enumerate(relevant_data[:3], 1):
            review_text = item.get("reviewText_cleaned", "No review text available").replace("\n", " ").strip()
            summary = self.summarizer.summarize(review_text, max_length=100, min_length=30)
            overall_rating = item.get("overall", "N/A")
            asin = item.get("asin", "N/A")
            response += f"{i}. Summary: {summary} | Rating: {overall_rating}/5 | ASIN: {asin}\n"

        response += "\nWould you like more reviews or details about the author?"
        self.last_response = response
        logging.info(f"Generated detailed response for query: '{query}'")
        return response

    def highlight_best_product(self, relevant_data):
        if not relevant_data:
            return "I'm sorry, I couldn't find any products to recommend."

        highest_rated = max(relevant_data, key=lambda x: x.get("overall", 0))
        review_text = highest_rated.get("reviewText_cleaned", "No review text available").replace("\n", " ").strip()
        summary = self.summarizer.summarize(review_text, max_length=100, min_length=30)
        overall_rating = highest_rated.get("overall", "N/A")
        asin = highest_rated.get("asin", "N/A")
        response = (
            f"The best-rated product has a rating of {overall_rating}/5. "
            f"ASIN: {asin}. "
            f"Summary: {summary}"
        )
        logging.info("Generated best product recommendation.")
        return response

    def provide_review_summary(self, relevant_data):
        if not relevant_data:
            return "I'm sorry, I couldn't find any reviews to summarize."

        review_summaries = [
            self.summarizer.summarize(item.get("reviewText_cleaned", "No review text available"), max_length=50, min_length=20)
            for item in relevant_data[:3]
        ]
        response = "Here are the summaries for the top reviews:\n" + "\n".join(review_summaries)
        logging.info("Generated review summaries.")
        return response
