# chatbot.py
from typing import Dict
import threading
import os
import sys
import time
import itertools

class CRISPRChatbot:
    """
    Main chatbot interface.
    """
    
    def __init__(self, qa_chain):
        self.qa_chain = qa_chain
    
    def answer_question(self, question: str) -> Dict:
        result = self.qa_chain.invoke({"query": question})
        
        # Return the result and the source documents for processing later in chat code
        return {
            "answer": result["result"],
            "source_documents": result["source_documents"] 
        }
    
    def loading_animation(self, stop_event, message="Generating response"):
        """
        Displays a loading animation in the console.
        """
        for char in itertools.cycle(['|', '/', '-', '\\']):
            if stop_event.is_set():  # Stop animation when signaled
                break
            sys.stdout.write(f'\r{message} {char}')  # \r moves cursor to beginning of line
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write('\r' + ' ' * (len(message) + 10) + '\r')  # Clear the loading message
        sys.stdout.flush()
    
    def chat(self):
        print("\n" + "="*50)
        print("LabAssist is here to help! Chatbot Ready!")
        print("="*50)
        print("Type 'q', 'quit', or 'exit' to end the session:\n")
        
        while True:
            question = input("Ask LabAssist: ").strip()
            if question.lower() in ['q', 'exit', 'quit']:
                print("Session ended. Thank you for using LabAssist.\n")
                break
            
            if not question:
                continue
            
            # Create stop event for animation
            stop_event = threading.Event()
            
            # Start loading animation in separate thread
            animation_thread = threading.Thread(
                target=self.loading_animation, 
                args=(stop_event, "Generating response")
            )
            animation_thread.start()
            
            try:
                # Get the answer
                result = self.answer_question(question)
            finally:
                # Stop the animation
                stop_event.set()
                animation_thread.join()
            
            # Process sources with page numbers
            unique_sources_info = set()
            for doc in result["source_documents"]:
                source_path = doc.metadata.get("source", "Unknown")  # Fixed typo: metadat -> metadata
                source_file = os.path.basename(source_path)  # Get just filename
                page = doc.metadata.get("page", "n/a")
                page_info = ""

                if isinstance(page, int):
                    page_info = f"(Page {page + 1})"
                elif page != "n/a":
                    page_info = f"(Page {page})"  # For non-integer page formats
                else:
                    page_info = "(Page N/A)"
                
                unique_sources_info.add(f"{source_file} {page_info}")
            
            # Display response
            print(f"\nResponse: {result['answer']}\n")
            
            # Display sources with page numbers
            if unique_sources_info:
                print("Sources:")
                for source in sorted(list(unique_sources_info)):  # Fixed: unique_sources -> unique_sources_info
                    print(f" - {source}")
            else:
                print("Sources: None Found")
            
            print()  # Add blank line for readability
