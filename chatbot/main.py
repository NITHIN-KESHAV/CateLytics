from modules.nlu import NLUModule
from modules.rag import RAGModule
from modules.memory import ConversationalMemory
from modules.sentiment import SentimentAnalyzer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

USE_LLM = True  # Toggle LLM usage

def main():
    weaviate_url = "http://50.18.99.196:8080"  # Replace with your EC2 instance IP
    nlu = NLUModule()
    rag = RAGModule(weaviate_url)
    memory = ConversationalMemory()
    sentiment_analyzer = SentimentAnalyzer()

    logging.info("Chatbot initialized. Starting conversation.")
    print("Chatbot: Hello! How can I help you today?")

    while True:
        user_input = input("User: ")
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye! Have a great day!")
            logging.info("User ended the conversation.")
            break

        asin = None
        if "asin:" in user_input.lower():
            parts = user_input.lower().split("asin:")
            query = parts[0].strip() if len(parts[0].strip()) > 0 else "Show reviews for ASIN"
            asin = parts[1].strip() if len(parts) > 1 else None
        else:
            query = user_input.strip()

        relevant_data = rag.retrieve_relevant_data(query=query, asin=asin)
        nlu_result = nlu.process_input(query)
        sentiment = sentiment_analyzer.analyze(query)

        if nlu_result["intent"] == "recommendation" and not asin:
            response = rag.highlight_best_product(relevant_data)
        else:
            response = rag.generate_response(query, relevant_data, nlu_result["intent"])

        memory.add_interaction(user_input, response)
        
        print(f"Chatbot: {response}")
        logging.info(f"Intent: {nlu_result['intent']}, Sentiment: {sentiment['label']} (score: {sentiment['score']:.2f})")
        logging.info(f"Recent memory: {memory.get_recent_history()}")

if __name__ == "__main__":
    main()
