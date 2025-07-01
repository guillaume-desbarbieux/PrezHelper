import requests

def generate_answer_ollama(question, context, model="mistral", ollama_host="http://ollama:11434"):
    """
    Envoie la question et le contexte à Ollama (mistral) pour générer une réponse.
    """
    prompt = f"Réponds à la question suivante en t'appuyant uniquement sur le contexte fourni.\n\nContexte :\n{context}\n\nQuestion : {question}\n\nRéponse :"
    response = requests.post(
        f"{ollama_host}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        },
        timeout=60
    )
    response.raise_for_status()
    return response.json()["response"]
