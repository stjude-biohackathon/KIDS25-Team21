import ollama

def get_embedding(chunk, model="nomic-embed-text"):
    
    embedding_data = []
    
    for i, chunk in enumerate(chunk):
        response = ollama.embeddings(model=model, prompt=chunk)
        
        embedding_data.append(
            {
            "text": chunk,
            "embedding": response['embedding'],
            "chunk_id": i,
            }
        )
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(chunk)} chunks")
    
    print("Embedding generation complete!")
    return embedding_data