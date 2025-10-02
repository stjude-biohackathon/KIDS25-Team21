import fitz
import os
import json
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter



def extract_text_from_pdf(pdf_path):

    pdf = fitz.open(pdf_path)
    
    full_text = ""

    for page_num in range(len(pdf)):
        page = pdf[page_num]
        page_text = page.get_text()
        full_text += f"\n---Page {page_num + 1}---\n"
        full_text += page_text

    pdf.close()
    
    return full_text


#Step 2: Chunk the text
def chunk_text(text, chunk_size=400, chunk_overlap=50):
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    
    return chunks

#step2b - chunk text and store metadata for source information
def chunk_text_with_metadata(text, pdf_path, chunk_size=400, chunk_overlap=50):
     
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_text(text)

    chunks_with_metadata = []

    PAGE_MARKER_PATTERN = re.compile(r'---Page (\d+)---')
    
    # Loop through generated chunks
    for i, chunk in enumerate(chunks):
        
        # Initialize page_number INSIDE the loop for each chunk
        page_number = "Unknown" 
        
        # Find first page number in the chunk
        page_match = PAGE_MARKER_PATTERN.search(chunk)

        if page_match:
            try:
                page_number = int(page_match.group(1))
            except ValueError:
                page_number = "Error"
        
        # Append the structured data
        chunks_with_metadata.append(
            {
                "text": chunk.strip(),
                "source": pdf_path,
                "page": page_number, 
                "chunk_id": i
            }
        )
            
    return chunks_with_metadata
        
        
