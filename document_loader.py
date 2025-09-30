# document_loader.py
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain.schema import Document

class DocumentLoader:
    """
    Handles loading of PDF and Word documents.
    """
    
    @staticmethod
    def load_from_folder(folder_path: str) -> List[Document]:
        documents = []
        folder = Path(folder_path)
        
        for file_path in folder.glob("**/*"):
            if file_path.suffix.lower() == '.pdf':
                loader = PyMuPDFLoader(str(file_path))
                documents.extend(loader.load())
                print(f"Loaded: {file_path.name}")
            elif file_path.suffix.lower() in ['.docx', '.doc']:
                loader = Docx2txtLoader(str(file_path))
                documents.extend(loader.load())
                print(f"Loaded: {file_path.name}")
        
        return documents

if __name__ == "__main__":
    # Configuration
    DOCUMENTS_FOLDER = "./crispr_protocols"
    FORCE_RELOAD = False  # Set to True to reprocess documents

    # Load documents
    doc_loader = DocumentLoader()
    documents = doc_loader.load_from_folder(DOCUMENTS_FOLDER)
    print(f"Total documents loaded: {len(documents)}")
