# main.py
import os
from document_loader import DocumentLoader
from text_processor import TextProcessor
from vector_store import VectorStoreManager
from qa_chain import QAChainBuilder
from chatbot import CRISPRChatbot

def initialize_chatbot(documents_folder: str, force_reload: bool = False):
    """Initialize the chatbot with document processing."""
    
    vector_store_manager = VectorStoreManager()
    
    # Check if vector store exists
    if os.path.exists(vector_store_manager.persist_dir) and not force_reload:
        print("Loading existing vector store...")
        vectorstore = vector_store_manager.load_vectorstore()
    else:
        print("Processing documents...")
        # Load documents
        doc_loader = DocumentLoader()
        documents = doc_loader.load_from_folder(documents_folder)
        
        # Process and chunk
        text_processor = TextProcessor()
        chunks = text_processor.split_documents(documents)
        
        # Create vector store
        vectorstore = vector_store_manager.create_vectorstore(chunks)
    
    # Build QA chain
    retriever = vector_store_manager.get_retriever(k=3)
    qa_builder = QAChainBuilder()
    qa_chain = qa_builder.build_chain(retriever)
    
    # Create chatbot
    chatbot = CRISPRChatbot(qa_chain)
    return chatbot

if __name__ == "__main__":
    # Configuration
    DOCUMENTS_FOLDER = "./crispr_protocols"
    FORCE_RELOAD = False  # Set to True to reprocess documents
    
    # Initialize and run
    chatbot = initialize_chatbot(DOCUMENTS_FOLDER, FORCE_RELOAD)
    chatbot.chat()
