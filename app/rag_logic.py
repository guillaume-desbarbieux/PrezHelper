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
    if distances:
        d_min = min(distances)
        d_max = max(distances)
        scores = []
        for d in distances:
            if d_max > d_min:
                score = 100 * (1 - (d - d_min) / (d_max - d_min))
            else:
                score = 100.0
            scores.append(round(score, 1))
    else:
        scores = [100.0] * len(results['documents'][0])
    results['scores'] = [scores]
    return results
