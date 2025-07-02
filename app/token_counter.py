import tiktoken
import streamlit as st

# Coût estimé par millier de tokens (en USD), INPUT uniquement
COST_PER_1K_INPUT = {
    "gpt-3.5-turbo": 0.0015,
    "gpt-3.5-turbo-1106": 0.0010,
    "gpt-4": 0.03,
    "gpt-4-1106-preview": 0.01,
    "gpt-4o": 0.005,
}

def count_tokens(messages: list[dict], model: str = "gpt-3.5-turbo") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")  # fallback

    num_tokens = 0
    for message in messages:
        num_tokens += 4  # overhead par message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
    num_tokens += 2  # fin de message assistant
    return num_tokens

def estimate_cost(tokens: int, model: str = "gpt-3.5-turbo") -> float:
    cost_per_1k = COST_PER_1K_INPUT.get(model, 0.0015)
    return round((tokens / 1000) * cost_per_1k, 6)

# Exemple d'utilisation
if __name__ == "__main__":
    model = "gpt-3.5-turbo"


prompt_intro = "Tu es un assistant expert de la documentation Prezevent. Réponds uniquement en français, de façon claire et concise, en t'appuyant exclusivement sur la documentation ci-dessous. Si la réponse n'est pas présente, indique-le poliment."

def load_corpus():
    with open("data/corpus_llm_gpt_optimized.txt", "r", encoding="utf-8") as f:
        return f.read()
corpus = load_corpus()


# Construction du prompt complet
full_prompt = (
    prompt_intro +
    "\n\n[QUESTION UTILISATEUR]:\n" +
    "Comment importer une liste de contacts d’un fichier excel ?" +
    "\n\n[CONTEXTE DOCUMENTAIRE]:\n" +
    corpus +
    "\n\n[REPONSE ATTENDUE]:\n"
)

token_count = count_tokens(full_prompt, model=model)
cost = estimate_cost(token_count, model=model)

print(f"Modèle utilisé       : {model}")
print(f"Nombre de tokens     : {token_count}")
print(f"Coût estimé (USD)    : ${cost}")
