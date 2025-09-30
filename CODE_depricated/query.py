import ollama
import chromadb

def get_query(question,instructions, collection_name="pdf_embeddings", persist_direcotry="./chroma_db", n_results=3):
    
    
    client = chromadb.PersistentClient(path=persist_direcotry)
    collection = client.get_collection(name=collection_name)
    
    question_embedding = ollama.embeddings(model="nomic-embed-text", prompt=question)
    
    result = collection.query(
        query_embeddings=[question_embedding["embedding"]],
        n_results=n_results
    )

    retrieved_chunks = result['documents'][0]
    
    context = "\n\n".join(retrieved_chunks)
    
    prompt = f"""
     Context from the document {context}
     
     Question: {question}
     
     {instructions}
    
    """
    
    print("Prompt to LLM:", prompt)
    
    response = ollama.chat(model="phi3:mini",
                           messages=[
                               {
                                   "role":"user",
                                   "content":prompt
                                   
                               }
                           ] 
    )
    
    answer = response['message']['content']
    

    return answer, retrieved_chunks
