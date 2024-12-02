from modules.nlu import NLUModule
from modules.rag import RAGModule
from modules.memory import ConversationalMemory
from modules.sentiment import SentimentAnalyzer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

        nlu_result = nlu.process_input(user_input)
        sentiment = sentiment_analyzer.analyze(user_input)
        relevant_data = rag.retrieve_relevant_data(user_input)

        # Handle vague queries with follow-up questions
        if nlu_result["intent"] == "clarification_request":
            response = "Could you clarify what you'd like me to show? For example, 'Show me reviews for headphones.'"
        else:
            response = rag.generate_response(user_input, relevant_data, nlu_result["intent"])

        memory.add_interaction(user_input, response)
        
        print(f"Chatbot: {response}")
        logging.info(f"Intent: {nlu_result['intent']}, Sentiment: {sentiment['label']} (score: {sentiment['score']:.2f})")
        logging.info(f"Recent memory: {memory.get_recent_history()}")

if __name__ == "__main__":
    main()
