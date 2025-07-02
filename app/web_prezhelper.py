# Importation des modules nécessaires
import streamlit as st
import openai
import tiktoken
import time
from sentence_transformers import SentenceTransformer, util
import re
import json

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
    # Charge la liste de documents individuels pré-découpés
    with open("data/corpus_docs.json", "r", encoding="utf-8") as f:
        docs = json.load(f)
    return docs
corpus_docs = load_corpus_docs()

# Variables globales pour stocker les résultats RAG
if 'top_docs' not in st.session_state:
    st.session_state['top_docs'] = []
    st.session_state['top_scores'] = []

def extraire_titre_et_contenu(doc):
    titre_match = re.search(r"Titre\s*:\s*(.*)", doc)
    titre = titre_match.group(1).strip() if titre_match else ""
    contenu_match = re.search(r"Contenu\s*:\s*(.*)", doc, flags=re.DOTALL)
    contenu = contenu_match.group(1).strip() if contenu_match else ""
    return titre, contenu

# Sidebar : paramètres RAG
st.sidebar.markdown("---")
st.sidebar.header("Paramètres RAG")
top_k = st.sidebar.number_input(
    "Nombre de documents pertinents (top_k)", min_value=1, max_value=10, value=3, step=1
)
min_score = st.sidebar.slider(
    "Score minimal de pertinence (cosinus)", min_value=0.0, max_value=1.0, value=0.3, step=0.01
)
relative_margin = st.sidebar.slider(
    "Marge relative (écart au meilleur score)", min_value=0.0, max_value=1.0, value=0.3, step=0.01,
    help="Un document n'est retenu que si son score >= max_score - relative_margin."
)
alpha = st.sidebar.slider(
    "Poids du titre (alpha)", min_value=0.0, max_value=1.0, value=0.7, step=0.05,
    help="Score mixte = alpha * score_titre + (1-alpha) * score_contenu"
)

# Bouton pour déclencher uniquement la recherche de documents pertinents
show_docs = st.button("Afficher les documents les plus pertinents (RAG)")

if show_docs and question:
    with st.spinner("Recherche des documents les plus pertinents dans la base documentaire..."):
        embedder = get_embedder()
        q_emb = embedder.encode(question, convert_to_tensor=True)
        scores = []
        titres = []
        for doc in corpus_docs:
            titre, contenu = extraire_titre_et_contenu(doc)
            titres.append(titre)
            titre_emb = embedder.encode(titre, convert_to_tensor=True)
            contenu_emb = embedder.encode(contenu, convert_to_tensor=True)
            score_titre = util.cos_sim(q_emb, titre_emb).item()
            score_contenu = util.cos_sim(q_emb, contenu_emb).item()
            score_mixte = alpha * score_titre + (1 - alpha) * score_contenu
            scores.append(score_mixte)
        max_score = max(scores) if scores else 0.0
        idx_sorted = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        filtered = [
            (i, scores[i]) for i in idx_sorted
            if scores[i] >= min_score and scores[i] >= max_score - relative_margin
        ]
        top_filtered = filtered[:top_k]
        st.session_state['top_docs'] = [corpus_docs[i] for i, _ in top_filtered]
        st.session_state['top_scores'] = [score for _, score in top_filtered]
    with st.expander("🔎 Debug : Documents les plus pertinents (RAG)"):
        for i, doc in enumerate(st.session_state['top_docs']):
            titre_match = re.search(r"Titre\s*:\s*(.*)", doc)
            titre = titre_match.group(1).strip() if titre_match else "(Titre inconnu)"
            st.markdown(f"**{i+1}. {titre}**  ")
            st.markdown(f"Score mixte : `{st.session_state['top_scores'][i]:.4f}`")
            st.text_area("Contenu", doc, height=120)

if question and selected:
    if st.button("Générer une réponse LLM à partir de la documentation"):
        if not st.session_state['top_docs']:
            st.warning("Veuillez d'abord afficher les documents pertinents (RAG) avec le bouton dédié.")
        else:
            rag_corpus = "\n\n".join(st.session_state['top_docs'])
            with st.expander("🔎 Debug : Documents les plus pertinents (RAG)"):
                for i, doc in enumerate(st.session_state['top_docs']):
                    titre_match = re.search(r"Titre\s*:\s*(.*)", doc)
                    titre = titre_match.group(1).strip() if titre_match else "(Titre inconnu)"
                    st.markdown(f"**{i+1}. {titre}**  ")
                    st.markdown(f"Score de similarité : `{st.session_state['top_scores'][i]:.4f}`")
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
