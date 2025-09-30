import fitz
import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

from extract_and_chunk import extract_text_from_pdf, chunk_text
from embedding import get_embedding
from db import save_embeddings_to_db
from query import get_query


pdf = fitz.open("mytaq_dna_polymerase_product_manual.pdf")

pdf_path = "mytaq_dna_polymerase_product_manual.pdf"

#Step 1: Extract text from pdf
extracted_text = extract_text_from_pdf(pdf_path)

print(extracted_text[:100])

#Step 2 Chunk Text
chunks = chunk_text(extracted_text)

print(f"Total chunks: {len(chunks)}")
print(chunks[0])

#Step 3: Generate embeddings and store
embeddings = get_embedding(chunks)

print(f"Embeddings generated for {len(embeddings)} chunks")
with open("embeddings.json", "w") as f:
    json.dump(embeddings, f)
    
print("Embeddings saved to embeddings.json")


#Step 4: Save to ChromaDB
collection = save_embeddings_to_db(embeddings)

print("Embeddings saved to ChromaDB collection:", collection.name)



#Step 5: Query the database
question = "What is the estiamted range for optimal annealing temperature for primers with Tm of 65 degrees?" #should be 58-60

#Gives context to the LLM on how to answer.  Added the hobbit thing for giggles. 
instructions = "Answer the question based on the context above.  If no answer is found just make something up but be sure to mention hobbits."


answer, chunks = get_query(question, instructions)

print(f"Answer: {answer}\n\n")

#TODO Possible to clean this up for display?  Mainly want to use this as a reference to see what was cited for the anser.
print(f"Chunks used for answer: {chunks[:100]}")


