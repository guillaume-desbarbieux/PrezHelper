import requests
import os
ollama_url = os.getenv("OLLAMA_URL", "http://ollama:11434")
# puis requests.post(ollama_url + "/api/generate", ...)

def generate_answer_ollama(question, context, model="mistral", ollama_host="http://ollama:11434"):
    """
    Envoie la question et le contexte à Ollama (modèle LLM, ex: mistral) pour générer une réponse.
    - question : la question de l'utilisateur
    - context : le contexte documentaire à fournir au modèle
    - model : nom du modèle LLM à utiliser (par défaut 'mistral')
    - ollama_host : URL de l'API Ollama
    Retourne la réponse générée (texte).
    """
    # Prépare le prompt à envoyer au modèle
    prompt = f"Réponds à la question suivante en t'appuyant uniquement sur le contexte fourni.\n\nContexte :\n{context}\n\nQuestion : {question}\n\nRéponse :"
    # Appel HTTP à l'API Ollama
    response = requests.post(
        f"{ollama_host}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        },
        timeout=60
    )
    response.raise_for_status()  # Lève une erreur si la requête a échoué
    return response.json()["response"]  # Retourne la réponse générée
