import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma")
collection = client.get_or_create_collection("prezevent_articles", embedding_function=embedding_function)

query = input("Entrez une question pour tester la recherche ChromaDB : ")
results = collection.query(query_texts=[query], n_results=5, include=["documents", "metadatas", "distances"])

print("\nRésultat documents :", results.get('documents'))
print("\nRésultat metadatas :", results.get('metadatas'))
print("\nRésultat distances :", results.get('distances'))
if 'error' in results:
    print("\nErreur :", results['error'])
