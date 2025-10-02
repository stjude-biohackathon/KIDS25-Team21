# main.py
import os
os.environ['OLLAMA_NUM_GPU'] = '1'
import argparse
import subprocess
from document_loader import DocumentLoader
from text_processor import TextProcessor
from vector_store import VectorStoreManager
from qa_chain import QAChainBuilder
from chatbot_mt import CRISPRChatbot

def get_args():
    parser = argparse.ArgumentParser(description="CRISPR Lab Protocol Chatbot Configuration")

    parser.add_argument(
        "-d", "--crispr_dir",
        type=str,
        default="./crispr_protocols",
        help="Path to folder containing CRISPR protocol documents (PDF/Word)."
    )

    parser.add_argument(
        "-f", "--force_reload",
        action="store_true",
        help="Reprocess documents even if embeddings/vector DB already exist."
    )

    return parser.parse_args()

def initialize_chatbot(documents_folder: str, force_reload: bool = False):
    """
    Initialize the chatbot with document processing.
    """
    
    vector_store_manager = VectorStoreManager()
    
    # Delete existing vector store if force reload
    if force_reload and os.path.exists(vector_store_manager.persist_dir):
        import shutil
        shutil.rmtree(vector_store_manager.persist_dir)
        print("Deleted existing vector store")
    
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
    # Check GPU availability
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ GPU detected")
        else:
            print("⚠ No GPU detected")
    except FileNotFoundError:
        print("⚠ nvidia-smi not found - GPU may not be available")

    # Configuration
    args = get_args()
    DOCUMENTS_FOLDER = args.crispr_dir
    FORCE_RELOAD = args.force_reload  # Set to True to reprocess documents
    
    # Initialize and run
    chatbot = initialize_chatbot(DOCUMENTS_FOLDER, FORCE_RELOAD)
    #chatbot.chat()
    chatbot.chat()
