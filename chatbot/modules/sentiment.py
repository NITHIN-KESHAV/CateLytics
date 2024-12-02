from transformers import pipeline
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SentimentAnalyzer:
    def __init__(self):
        self.sentiment_pipeline = pipeline("sentiment-analysis")
        logging.info("Initialized Sentiment Analyzer")

    def analyze(self, text):
        try:
            result = self.sentiment_pipeline(text)[0]
            logging.info(f"Analyzed sentiment for text: {text[:50]}...")
            return {
                "label": result["label"],
                "score": result["score"]
            }
        except Exception as e:
            logging.error(f"Error analyzing sentiment: {str(e)}")
            return {"label": "NEUTRAL", "score": 0.5}