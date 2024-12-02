from transformers import pipeline
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Summarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.summarizer = pipeline("summarization", model=model_name)
        logging.info(f"Initialized summarizer with model: {model_name}")

    def summarize(self, text, max_length=50, min_length=25):
        try:
            summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            logging.error(f"Error summarizing text: {str(e)}")
            return "No summary available."
