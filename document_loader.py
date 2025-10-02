# document_loader.py
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain.schema import Document

class DocumentLoader:
    """Handles loading of PDF and Word documents."""
    
    @staticmethod
    def load_from_folder(folder_path: str) -> List[Document]:
        documents = []
        folder = Path(folder_path)
        
        for file_path in folder.glob("**/*"):
            if file_path.suffix.lower() == '.pdf':
                loader = PyMuPDFLoader(str(file_path))
                docs = loader.load()
                
                # Add filename to metadata for each page
                for doc in docs:
                    doc.metadata['source'] = file_path.name  # Just filename
                    # page number is already in metadata from PyMuPDFLoader
                
                documents.extend(docs)
                print(f"Loaded: {file_path.name} ({len(docs)} pages)")
                
            elif file_path.suffix.lower() in ['.docx', '.doc']:
                loader = Docx2txtLoader(str(file_path))
                docs = loader.load()
                
                # Add filename and page info for Word docs
                for i, doc in enumerate(docs):
                    doc.metadata['source'] = file_path.name
                    doc.metadata['page'] = 0  # Word docs don't have pages in loader
                
                documents.extend(docs)
                print(f"Loaded: {file_path.name}")
        
        return documents

if __name__ == "__main__":
    # Configuration
    DOCUMENTS_FOLDER = "./crispr_protocols"
    FORCE_RELOAD = True  # Set to True to reprocess documents

    # Load documents
    doc_loader = DocumentLoader()
    documents = doc_loader.load_from_folder(DOCUMENTS_FOLDER)
    print(f"Total documents loaded: {len(documents)}")

    for doc in documents:
        if doc.metadata.get('source', '').lower().endswith('.pdf'):
            if 'page' in doc.metadata:
                doc.metadata['page'] = doc.metadata['page'] + 1
        elif doc.metadata.get('source', '').lower().endswith(('.docx', '.doc')):
            doc.metadata['page'] = "n/a"
