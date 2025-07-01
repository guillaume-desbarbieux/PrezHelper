import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Nom de la collection Chroma utilisée pour stocker les articles
COLLECTION_NAME = "prezevent_articles"

# Embedding local cohérent avec l'indexation
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

def ask_question(question, top_k=3):
    """
    Interroge la base documentaire Chroma pour trouver les passages les plus pertinents pour la question.
    - question : question utilisateur
    - top_k : nombre de résultats à retourner
    Retourne un dictionnaire avec documents, metadatas, scores, distances.
    """
    try:
        client = chromadb.PersistentClient(path="chroma")
        collection = client.get_or_create_collection(COLLECTION_NAME, embedding_function=embedding_function)
        results = collection.query(
            query_texts=[question],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        # Vérification de la structure minimale attendue
        if not results or 'documents' not in results or not results['documents'] or not results['documents'][0]:
            return {'documents': [[]], 'metadatas': [[]], 'scores': [[0.0]], 'distances': [[0.0]]}
        # Calcul du score de pertinence (%) pour chaque document
        distances = results.get('distances', [[0.0]*top_k])[0]
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        if distances:
            d_min = min(distances)
            d_max = max(distances)
            scored = []
            for i, d in enumerate(distances):
                # Score de pertinence normalisé (100% = plus proche)
                if d_max > d_min:
                    score = 100 * (1 - (d - d_min) / (d_max - d_min))
                else:
                    score = 100.0
                scored.append((score, docs[i], metas[i], d))
            # Tri décroissant par score de pertinence
            scored.sort(reverse=True, key=lambda x: x[0])
            # Décompose pour remettre dans results
            results['documents'][0] = [x[1] for x in scored]
            results['metadatas'][0] = [x[2] for x in scored]
            results['scores'] = [[round(x[0], 1) for x in scored]]
            results['distances'][0] = [x[3] for x in scored]
        else:
            results['scores'] = [[100.0] * len(docs)]  # Si pas de distances, score par défaut
        return results
    except Exception as e:
        # Retourne un résultat vide structuré en cas d'erreur
        return {'documents': [[]], 'metadatas': [[]], 'scores': [[0.0]], 'distances': [[0.0]], 'error': str(e)}
