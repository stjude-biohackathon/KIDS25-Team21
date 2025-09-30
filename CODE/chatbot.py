# chatbot.py
from typing import Dict

class CRISPRChatbot:
    """
    Main chatbot interface.
    """
    
    def __init__(self, qa_chain):
        self.qa_chain = qa_chain
    
    def answer_question(self, question: str) -> Dict:
        result = self.qa_chain.invoke({"query": question})
        return {
            "answer": result["result"],
            "sources": [doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]
        }
    
    def chat(self):
        print("\n" + "="*50)
        print("CRISPR Protocol Chatbot Ready!")
        print("="*50)
        print("Type 'exit' to quit\n")
        
        while True:
            question = input("You: ").strip()
            if question.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            result = self.answer_question(question)
            print(f"\nBot: {result['answer']}")
            print(f"Sources: {', '.join(set(result['sources']))}\n")
