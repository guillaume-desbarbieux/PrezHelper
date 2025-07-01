import chromadb

COLLECTION_NAME = "prezevent_articles"

def ask_question(question, top_k=3):
    client = chromadb.Client()
    collection = client.get_or_create_collection(COLLECTION_NAME)
    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )
    return results
