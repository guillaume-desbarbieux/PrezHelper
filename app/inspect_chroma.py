import chromadb
import pkg_resources

print(f"ChromaDB version utilisée : {pkg_resources.get_distribution('chromadb').version}")

client = chromadb.PersistentClient(path="chroma")

print("Collections disponibles :")
for col in client.list_collections():
    print(f"- {col.name}")

col_name = "prezevent_articles"
try:
    collection = client.get_collection(col_name)
    count = collection.count()
    print(f"\nCollection '{col_name}' : {count} documents")
    # Affiche le contenu réel des 3 premiers documents
    docs = collection.get(limit=3)
    print("\nExemple de documents:")
    for i, doc in enumerate(docs.get('documents', [])):
        print(f"--- Document {i+1} ---")
        print(doc)
        print("Méta:", docs.get('metadatas', [])[i] if i < len(docs.get('metadatas', [])) else None)
        print()
except Exception as e:
    print(f"Erreur lors de l'accès à la collection '{col_name}' : {e}")
