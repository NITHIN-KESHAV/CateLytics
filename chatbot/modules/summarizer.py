from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Summarizer:
    def __init__(self, model_path="models/fine_tuned_t5"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(self.device)
        logging.info(f"Initialized summarizer with model: {model_path}")

    def summarize(self, text, max_length=100, min_length=30, num_beams=3):
        """
        Generate a summary using the fine-tuned model.
        """
        try:
            inputs = self.tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
            inputs = inputs.to(self.device)

            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                early_stopping=True,
            )

            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logging.error(f"Error summarizing text: {str(e)}")
            return "No summary available."
