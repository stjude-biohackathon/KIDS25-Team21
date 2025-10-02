# vector_store.py
from typing import List, Optional
from langchain.schema import Document
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
import os
import shutil

class VectorStoreManager:
    """
    Manages vector database operations.
    """
    
    def __init__(self, persist_dir: str = "./chroma_db", embedding_model: str = "mxbai-embed-large"):
        # nomic-embed-text, mxbai-embed-large
        self.persist_dir = persist_dir
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.vectorstore: Optional[Chroma] = None
    
    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        print("Vector store created and persisted")
        return self.vectorstore

    def delete_vectorstore(self):
        #Delete the existing vector store to avoid clash
        
        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)
            print(f"Deleted existing vector store at {self.persist_dir}")
            self.vectorstore = None
            return True
        else:
            print(f"No vector store found at {self.persist_dir}")
            return False
    
    def load_vectorstore(self) -> Chroma:
        self.vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )
        print("Vector store loaded from disk")
        return self.vectorstore
    
    def get_retriever(self, k: int = 3):
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
