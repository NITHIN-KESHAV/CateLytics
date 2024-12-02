import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ConversationalMemory:
    def __init__(self, max_history=10):
        self.history = []
        self.max_history = max_history
        logging.info(f"Initialized Conversational Memory with max_history={max_history}")

    def add_interaction(self, user_input, bot_response):
        self.history.append({"user": user_input, "bot": bot_response})
        if len(self.history) > self.max_history:
            self.history.pop(0)
        logging.info(f"Added interaction to memory. Current history size: {len(self.history)}")

    def get_recent_history(self, n=5):
        return self.history[-n:]

    def clear_history(self):
        self.history.clear()
        logging.info("Cleared conversation history")