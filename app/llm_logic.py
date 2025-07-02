import requests
import os
ollama_url = os.getenv("OLLAMA_URL", "http://ollama:11434")

def generate_answer_ollama(prompt, model="mistral"):
    """
    Envoie la question et le contexte à Ollama (modèle LLM, ex: mistral) pour générer une réponse.
    """
    response = requests.post(
        f"{ollama_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        },
        timeout=600  # timeout augmenté
    )
    response.raise_for_status()
    return response.json()["response"]
