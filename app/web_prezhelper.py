# Importation des modules nécessaires
import streamlit as st
import openai
import tiktoken
import time
from sentence_transformers import SentenceTransformer, util
import re

# Configuration de la page Streamlit
st.set_page_config(page_title="PrezHelper IA", layout="centered")
st.title("PrezHelper IA")

# Paramètre : introduction personnalisable du prompt envoyé au LLM
prompt_intro = (
    "Tu es un assistant expert de Prezevent. "
    "Tu dois répondre uniquement en utilisant les documents fournis dans chaque requête. "
    "Si aucune réponse claire n’est présente, tu réponds simplement que tu ne sais pas. "
    "En fin de réponse, indique toujours les articles utilisés sous la forme : "
    "📄 Source : [Titre de l’article](URL)"
)

# Chargement du corpus documentaire
@st.cache_data(show_spinner=False)
def load_corpus():
    with open("data/corpus_llm.txt", "r", encoding="utf-8") as f:
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

# Sélection du ou des modèles à utiliser
model_options = [
    ("gpt-3.5-turbo-1106", "GPT-3.5 Turbo (1106)"),
    ("gpt-4o", "GPT-4o"),
    ("rag-gpt-4o", "RAG + GPT-4o (démo)")
]
model_labels = [label for _, label in model_options]
model_keys = [key for key, _ in model_options]
selected = st.sidebar.multiselect(
    "Modèles à comparer",
    options=model_keys,
    default=["gpt-4o"]
)

@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer("msmarco-MiniLM-L6-cos-v5")

@st.cache_data(show_spinner=False)
def load_corpus_docs():
    # Découpe le corpus en documents individuels
    with open("data/corpus_llm.txt", "r", encoding="utf-8") as f:
        txt = f.read()
    docs = re.split(r"\[\/DOCUMENT\]", txt)
    docs = [d.strip() + "\n[/DOCUMENT]" for d in docs if d.strip()]
    return docs
corpus_docs = load_corpus_docs()

if question and selected:
    if st.button("Générer une réponse LLM à partir de la documentation"):
        with st.spinner("Recherche des 3 documents les plus pertinents dans la base documentaire..."):
            embedder = get_embedder()
            # Embedding de la question
            q_emb = embedder.encode(question, convert_to_tensor=True)
            # Embedding des documents
            doc_embs = embedder.encode(corpus_docs, convert_to_tensor=True, show_progress_bar=False)
            # Similarité cosinus
            hits = util.cos_sim(q_emb, doc_embs)[0].cpu().numpy()
            top_idx = hits.argsort()[-3:][::-1]
            top_docs = [corpus_docs[i] for i in top_idx]
            top_scores = [hits[i] for i in top_idx]
            rag_corpus = "\n\n".join(top_docs)
        with st.expander("🔎 Debug : Documents les plus pertinents (RAG)"):
            for i, doc in enumerate(top_docs):
                # Extraction du titre
                titre_match = re.search(r"Titre\s*:\s*(.*)", doc)
                titre = titre_match.group(1).strip() if titre_match else "(Titre inconnu)"
                st.markdown(f"**{i+1}. {titre}**  ")
                st.markdown(f"Score de similarité : `{top_scores[i]:.4f}`")
                st.text_area("Contenu", doc, height=120)
        with st.spinner("Génération de la réponse par ChatGPT à partir des documents sélectionnés..."):
            cols = st.columns(len(selected))
            for idx, model_name in enumerate(selected):
                with cols[idx]:
                    try:
                        client = openai.OpenAI(api_key=openai_api_key)
                        rag_prompt = (
                            f"Voici la question d'un utilisateur :\n{question}\n\n"
                            f"Voici les documents pertinents à ta disposition :\n{rag_corpus}\n\n"
                            "ATTENTION : ne fais pas d'invention. Ne réponds que si la réponse est clairement présente."
                        )
                        messages = [
                            {"role": "system", "content": prompt_intro},
                            {"role": "user", "content": rag_prompt}
                        ]
                        start = time.time()
                        response = client.chat.completions.create(
                            model=model_name if model_name != "rag-gpt-4o" else "gpt-4o",
                            messages=messages,
                            max_tokens=800,
                            temperature=0.2
                        )
                        elapsed = time.time() - start
                        answer = response.choices[0].message.content
                        input_tokens = count_tokens(messages, model=model_name if model_name != "rag-gpt-4o" else "gpt-4o")
                        output_tokens = count_tokens(answer, model=model_name if model_name != "rag-gpt-4o" else "gpt-4o")
                        total_tokens = input_tokens + output_tokens
                        cost = estimate_cost(input_tokens, output_tokens, model=model_name if model_name != "rag-gpt-4o" else "gpt-4o")
                        st.success(f"Réponse générée par {model_name}")
                        st.subheader(f"Réponse {model_name} :")
                        st.write(answer)
                        st.info(f"Input tokens : {input_tokens} | Output tokens : {output_tokens} | Total : {total_tokens} | Coût estimé : ${cost} | Temps de réponse : {elapsed:.2f}s")
                    except Exception as e:
                        st.error(f"Erreur {model_name} : {e}")
