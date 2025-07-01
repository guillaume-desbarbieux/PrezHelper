import requests
import os
ollama_url = os.getenv("OLLAMA_URL", "http://ollama:11434")

def generate_answer_ollama(question, context, model="mistral"):
    """
    Envoie la question et le contexte à Ollama (modèle LLM, ex: mistral) pour générer une réponse.
    """
    prompt = f"Réponds à la question suivante en t'appuyant uniquement sur le contexte fourni.\n\nContexte :\n{context}\n\nQuestion : {question}\n\nRéponse :"
    response = requests.post(
        f"{ollama_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        },
        timeout=180  # timeout augmenté
    )
    response.raise_for_status()
    return response.json()["response"]
