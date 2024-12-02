from modules.nlu import NLUModule
from modules.rag import RAGModule

def main():
    nlu = NLUModule("bert-base-uncased")
    rag = RAGModule("http://localhost:8080")
    
    # Implement main chatbot logic

if __name__ == "__main__":
    main()