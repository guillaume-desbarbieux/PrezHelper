import chromadb

COLLECTION_NAME = "prezevent_articles"

def ask_question(question, top_k=3):
    client = chromadb.Client()
    collection = client.get_or_create_collection(COLLECTION_NAME)
    results = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["distances"]
    )
    # Calcul du score de pertinence (%) pour chaque document
    distances = results.get('distances', [[0.0]*top_k])[0]
    docs = results['documents'][0]
    metas = results['metadatas'][0]
    if distances:
        d_min = min(distances)
        d_max = max(distances)
        scored = []
        for i, d in enumerate(distances):
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
        results['scores'] = [[100.0] * len(docs)]
    return results
