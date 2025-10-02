# chatbot.py
from typing import Dict
import threading
import sys
import time
import itertools
import os

class CRISPRChatbot:
    """
    Main chatbot interface.
    """
    
    def __init__(self, qa_chain):
        self.qa_chain = qa_chain
    
    # def answer_question(self, question: str) -> Dict:
    #     result = self.qa_chain.invoke({"query": question})
    #     return {
    #         "answer": result["result"],
    #         "sources": [doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]
    #     }

    def answer_question(self, question: str) -> Dict:
        result = self.qa_chain.invoke({"query": question})
        return {
            "answer": result["result"],
            "sources": [doc.metadata.get("source","Unknown") for doc in result["source_documents"]],
            "source_documents": result["source_documents"]
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

            print("Generating response...")
            result = self.answer_question(question)

            #process sources with page numbers - pv edit
            unique_sources_info= set()
            for doc in result["source_documents"]:
                source_path=doc.metadata.get("source", "Unknown")
                source_file=os.path.basename(source_path) #get just filename
                page=doc.metadata.get("page","n/a")

                
                page_info = ""

                if isinstance(page, int):
                    page_info = f"(Page {page})"
                elif page != "n/a":
                    page_info = f"(Page {page})" # For non-integer page formats
                else:
                    page_info = "(Page N/A)"
                
                unique_sources_info.add(f"{source_file} {page_info}")
                
                    
            #end pv edits for page number - not sure if works yet  
            
            print(f"\nResponse: {result['answer']}")

            if unique_sources_info:
                print("\nSources:")
                for source in sorted(list(unique_sources_info)):
                    print(f" - {source}")
            else:
                print("Sources: None Found")
            #print(f"Sources: {', '.join(set(result['sources']))}\n")


