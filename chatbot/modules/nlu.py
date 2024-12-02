from sentence_transformers import SentenceTransformer
import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NLUModule:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.intents = ["product_info", "review_search", "recommendation", "general_query"]
        logging.info(f"Initialized NLU module with model: {model_name}")

    def process_input(self, text):
        try:
            embedding = self.model.encode(text)
            intent = self._classify_intent(embedding)
            entities = self._extract_entities(text)
            logging.info(f"Processed input: intent={intent}, entities={entities}")
            return {"intent": intent, "entities": entities, "embedding": embedding}
        except Exception as e:
            logging.error(f"Error processing input: {str(e)}")
            return {"intent": "general_query", "entities": [], "embedding": None}

    def _classify_intent(self, embedding):
        # Placeholder for intent classification
        return self.intents[0]

    def _extract_entities(self, text):
        # Placeholder for entity extraction
        return []