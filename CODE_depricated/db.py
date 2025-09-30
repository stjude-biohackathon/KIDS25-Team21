import chromadb
from chromadb.config import Settings


def save_embeddings_to_db(embeddings, collection_name="pdf_embeddings", persist_directory="./chroma_db"):


    client = chromadb.PersistentClient(path=persist_directory)
    
    collection = client.get_or_create_collection(name=collection_name, metadata={"source": "pdf_embeddings"})
    
    docs =[]
    embeddings_list = []
    metadata = []
    ids = []
    
    
    for item in embeddings:
        docs.append(item['text'])
        embeddings_list.append(item['embedding'])
        metadata.append({"chunk_id": item['chunk_id']})
        ids.append(str(item['chunk_id']))
        
    collection.add(
        documents=docs,
        embeddings=embeddings_list,
        metadatas=metadata,
        ids=ids
    )
    
    return collection