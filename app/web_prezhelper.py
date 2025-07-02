# Importation des modules nécessaires
import streamlit as st
import openai
import tiktoken

# Configuration de la page Streamlit
st.set_page_config(page_title="PrezHelper IA", layout="centered")
st.title("PrezHelper IA")

# Paramètre : introduction personnalisable du prompt envoyé au LLM
prompt_intro = "Tu es un assistant expert de Prezevent. Tu ne réponds qu'à partir des documents ci-dessous."

# Chargement du corpus documentaire
@st.cache_data(show_spinner=False)
def load_corpus():
    with open("data/corpus_llm_gpt_optimized.txt", "r", encoding="utf-8") as f:
        return f.read()
corpus = load_corpus()

# Champ de saisie pour la question de l'utilisateur
question = st.text_input("Posez votre question :", "")

# Clé API OpenAI
openai_api_key = st.sidebar.text_input("Clé API OpenAI", type="password")

COST_PER_1K_INPUT = {
    "gpt-3.5-turbo": 0.0015,
    "gpt-3.5-turbo-1106": 0.0010,
    "gpt-4": 0.03,
    "gpt-4-1106-preview": 0.01,
    "gpt-4o": 0.005,
}
COST_PER_1K_OUTPUT = {
    "gpt-3.5-turbo": 0.002,
    "gpt-3.5-turbo-1106": 0.002,
    "gpt-4": 0.06,
    "gpt-4-1106-preview": 0.03,
    "gpt-4o": 0.015,
}

def count_tokens(messages, model="gpt-4o"):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = 0
    if isinstance(messages, str):
        num_tokens = len(encoding.encode(messages))
    else:
        for message in messages:
            num_tokens += 4
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
        num_tokens += 2
    return num_tokens

def estimate_cost(input_tokens, output_tokens, model="gpt-4o"):
    in_cost = COST_PER_1K_INPUT.get(model, 0.0015)
    out_cost = COST_PER_1K_OUTPUT.get(model, 0.002)
    return round((input_tokens / 1000) * in_cost + (output_tokens / 1000) * out_cost, 6)

if question:
    if st.button("Générer une réponse LLM à partir de la documentation"):
        with st.spinner("Génération de la réponse par ChatGPT à partir de la documentation complète..."):
            # Construction du prompt complet
            full_prompt = (
                "\n\n[INSTRUCTIONS SYSTEME]" +
                prompt_intro +
                "\n\n[QUESTION UTILISATEUR]:\n" +
                question +
                "\n\n[DOCUMENTATION DISPONIBLE]:\n" +
                corpus +
                "\n\n[ATTENTION] Ne réponds que si la réponse est clairement indiquée. Sinon, dis que tu ne sais pas.\n"
            )
            if not openai_api_key:
                st.error("Veuillez renseigner votre clé API OpenAI dans la sidebar.")
            else:
                try:
                    client = openai.OpenAI(api_key=openai_api_key)
                    model_name = "gpt-4o"
                    messages = [
                        {"role": "system", "content": prompt_intro},
                        {"role": "user", "content": full_prompt}
                    ]
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        max_tokens=800,
                        temperature=0.2
                    )
                    answer = response.choices[0].message.content
                    # Comptage des tokens input/output
                    input_tokens = count_tokens(messages, model=model_name)
                    output_tokens = count_tokens(answer, model=model_name)
                    total_tokens = input_tokens + output_tokens
                    cost = estimate_cost(input_tokens, output_tokens, model=model_name)
                    st.success("Réponse générée par ChatGPT ci-dessous.")
                    st.subheader("Réponse générée par ChatGPT :")
                    st.write(answer)
                    st.info(f"Modèle : {model_name} | Input tokens : {input_tokens} | Output tokens : {output_tokens} | Total : {total_tokens} | Coût estimé : ${cost}")
                except Exception as e:
                    st.error(f"Erreur lors de l'appel à l'API OpenAI : {e}")
