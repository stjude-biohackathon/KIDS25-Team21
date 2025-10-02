# text_processor.py
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class TextProcessor:
    """
    Handles text chunking and processing.
    """


    #try smaller chunk size
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 76):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        splits = self.text_splitter.split_documents(documents)
        print(f"Created {len(splits)} chunks")
        return splits
