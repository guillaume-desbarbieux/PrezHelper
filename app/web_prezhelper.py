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

model_llm = "gpt-4o"

# Paramètre : introduction personnalisable du prompt envoyé au LLM
prompt_intro = (
    "Tu es un assistant expert de Prezevent. "
    "Tu dois répondre uniquement en utilisant les documents fournis dans chaque requête. "
    "Ces documents peuvent inclure des descriptions détaillées d’interfaces visuelles (captures d’écran) sous forme de texte. "
    "Ces descriptions remplacent l'image et doivent être prises en compte pour comprendre les boutons, menus ou options affichés. "
    "Tu ne dois jamais mentionner d'images."
    "Si aucune réponse claire n’est présente dans les documents, réponds simplement que tu ne sais pas. "
    "En fin de réponse, indique toujours les articles utilisés sous la forme : "
    "📄 Source : [Titre de l’article](URL)"
)

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

def count_tokens(messages, model=model_llm):
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

def estimate_cost(input_tokens, output_tokens, model=model_llm):
    in_cost = COST_PER_1K_INPUT.get(model, 0.0015)
    out_cost = COST_PER_1K_OUTPUT.get(model, 0.002)
    return round((input_tokens / 1000) * in_cost + (output_tokens / 1000) * out_cost, 6)

@st.cache_resource(show_spinner=False)
def get_embedder():
    try:
        return SentenceTransformer("msmarco-MiniLM-L6-cos-v5")
    except Exception as e:
        st.error("❌ Le modèle d'embedding 'msmarco-MiniLM-L6-cos-v5' n'a pas pu être chargé.")
        return None

@st.cache_data(show_spinner=False)
def load_corpus():
    # Charge la liste de documents individuels pré-découpés
    with open("data/corpus_list.json", "r", encoding="utf-8") as f:
        docs = json.load(f)
    return docs
corpus = load_corpus()

# Variables globales pour stocker les résultats RAG et reformulation
if 'top_docs' not in st.session_state:
    st.session_state['top_docs'] = []
    st.session_state['top_scores'] = []
if 'reformulation_infos' not in st.session_state:
    st.session_state['reformulation_infos'] = None
if 'question_recherche' not in st.session_state:
    st.session_state['question_recherche'] = ''

def extraire_titre_et_contenu(doc):
    titre_match = re.search(r"Titre\s*:\s*(.*)", doc)
    titre = titre_match.group(1).strip() if titre_match else ""
    contenu_match = re.search(r"Contenu\s*:\s*(.*)", doc, flags=re.DOTALL)
    contenu = contenu_match.group(1).strip() if contenu_match else ""
    return titre, contenu

# Champ de saisie pour la question de l'utilisateur
question = st.text_input("Posez votre question :", "")

# Clé API OpenAI
openai_api_key = st.sidebar.text_input("Clé API OpenAI", type="password")

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
    "Poids du titre (alpha)", min_value=0.0, max_value=1.0, value=0.6, step=0.05,
    help="Score mixte = alpha * score_titre + (1-alpha) * score_contenu"
)

# --- Affichage des 4 boutons côte à côte sous la zone de question ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    reform_btn = st.button("Reformuler la question")
with col2:
    rag_btn = st.button("Rechercher les documents pertinents")
with col3:
    llm_btn = st.button("Générer une réponse à partir de la documentation pertinente")
with col4:
    llm_all_btn = st.button("Générer une réponse avec TOUTE la documentation")

# --- Étape 1 : Reformulation ---
if reform_btn:
    if openai_api_key and question:
        system_prompt = (
            "Tu es un assistant qui reformule des questions posées en langage naturel par des utilisateurs, "
            "pour les adapter au style de la documentation de l’application Prezevent. "
            "La documentation utilise un style clair, direct, sous forme de questions techniques. "
            "Les titres des articles commencent souvent par 'Comment' et se concentrent sur une action précise, "
            "comme 'Comment exporter une liste de contacts ?' ou 'Comment créer une campagne de mail ?'. "
            "Ta tâche est de reformuler les questions utilisateur dans ce style."
        )
        try:
            start = time.time()
            client = openai.OpenAI(api_key=openai_api_key)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
            response = client.chat.completions.create(
                model=model_llm,
                messages=messages,
                max_tokens=100,
                temperature=0.2
            )
            elapsed = time.time() - start
            question_recherche = response.choices[0].message.content.strip()
            input_tokens = count_tokens(messages, model=model_llm)
            output_tokens = count_tokens(question_recherche, model=model_llm)
            total_tokens = input_tokens + output_tokens
            cost = estimate_cost(input_tokens, output_tokens, model=model_llm)
            reformulation_infos = {
                "question_reformulee": question_recherche,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "cost": cost,
                "elapsed": elapsed
            }
            st.session_state['reformulation_infos'] = reformulation_infos
            st.session_state['question_recherche'] = question_recherche
        except Exception as e:
            st.session_state['reformulation_infos'] = None
            st.session_state['question_recherche'] = question
            st.warning(f"Erreur lors de la reformulation : {e}")
    else:
        st.session_state['reformulation_infos'] = None
        st.session_state['question_recherche'] = question
        st.info("Aucune reformulation effectuée (clé API manquante ou question vide).")

# Affichage de la question reformulée et du coût si disponible
reformulation_infos = st.session_state.get('reformulation_infos')
if reformulation_infos:
    st.info(f"**Question reformulée :** {reformulation_infos['question_reformulee']}\n\nCoût estimé : ${reformulation_infos['cost']} | Temps de réponse : {reformulation_infos['elapsed']:.2f}s")

# --- Étape 2 : Recherche de documents pertinents ---
if rag_btn and question :
    with st.spinner("Recherche des documents les plus pertinents dans la base documentaire..."):
        question_recherche = st.session_state.get('question_recherche') or question
        embedder = get_embedder()
        if embedder is None:
            st.stop()
        q_emb = embedder.encode(question_recherche, convert_to_tensor=True)
        scores = []
        titres = []
        for doc in corpus:
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
        st.session_state['top_docs'] = [corpus[i] for i, _ in top_filtered]
        st.session_state['top_scores'] = [score for _, score in top_filtered]

        st.success(f"{len(st.session_state['top_docs'])} document(s) pertinent(s) trouvé(s).", icon="✅")
        for i, doc in enumerate(st.session_state['top_docs']):
            titre_match = re.search(r"Titre\\s*:\\s*(.*)", doc)
            titre = titre_match.group(1).strip() if titre_match else "(Titre inconnu)"
            st.markdown(f"**{i+1}. {titre}**  ")
            st.markdown(f"Score mixte : `{st.session_state['top_scores'][i]:.4f}`")
            st.text_area("Contenu", doc, height=120)

# --- Étape 3 : Génération de la réponse LLM ---
if llm_btn and question :
    if not st.session_state['top_docs']:
        st.warning("Veuillez d'abord rechercher les documents pertinents (RAG) avec le bouton dédié.")
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
            try:
                client = openai.OpenAI(api_key=openai_api_key)
                rag_prompt = (
                            f"Voici la question d'un utilisateur :\n{st.session_state.get('question_recherche') or question}\n\n"
                            "Voici les documents pertinents à ta disposition :\n"
                            "Certains passages décrivent l’interface visuelle (ex : boutons, onglets, textes affichés). "
                            "Ces descriptions textuelles remplacent les captures d’écran et sont à interpréter comme si tu voyais l’écran.\n\n"
                            f"{rag_corpus}\n\n"
                            "ATTENTION : ne fais pas d'invention. Ne réponds que si la réponse est clairement présente."
                        )

                messages = [
                            {"role": "system", "content": prompt_intro},
                            {"role": "user", "content": rag_prompt}
                        ]
                start = time.time()
                response = client.chat.completions.create(
                            model=model_llm,
                            messages=messages,
                            max_tokens=800,
                            temperature=0.2
                )
                elapsed = time.time() - start
                answer = response.choices[0].message.content
                input_tokens = count_tokens(messages, model=model_llm)
                output_tokens = count_tokens(answer, model=model_llm)
                total_tokens = input_tokens + output_tokens
                cost = estimate_cost(input_tokens, output_tokens, model=model_llm)
                st.subheader(f"Réponse {model_llm} :")
                st.write(answer)
                st.info(f"Coût estimé : ${cost} | Temps de réponse : {elapsed:.2f}s")        
            except Exception as e:
                        st.error(f"Erreur {model_llm} : {e}")

# --- Étape 4 : Génération de la réponse LLM avec toute la documentation ---
if llm_all_btn and question :
    with st.spinner("Génération de la réponse par ChatGPT à partir de toute la documentation..."):
        rag_corpus = "\n\n".join(corpus)  # toute la documentation brute
        try:
            client = openai.OpenAI(api_key=openai_api_key)
            rag_prompt = (
                            f"Voici la question d'un utilisateur :\n{question}\n\n"
                            "Voici les documents pertinents à ta disposition :\n"
                            "Certains passages décrivent l’interface visuelle (ex : boutons, onglets, textes affichés). "
                            "Ces descriptions textuelles remplacent les captures d’écran et sont à interpréter comme si tu voyais l’écran.\n\n"
                            f"{rag_corpus}\n\n"
                            "ATTENTION : ne fais pas d'invention. Ne réponds que si la réponse est clairement présente dans la documentation."
                        )
            messages = [{"role": "system", "content": prompt_intro},
                        {"role": "user", "content": rag_prompt}]
            start = time.time()
            response = client.chat.completions.create(
                            model=model_llm,
                            messages=messages,
                            max_tokens=800,
                            temperature=0.2
                        )
            elapsed = time.time() - start
            answer = response.choices[0].message.content
            input_tokens = count_tokens(messages, model=model_llm)
            output_tokens = count_tokens(answer, model=model_llm)
            total_tokens = input_tokens + output_tokens
            cost = estimate_cost(input_tokens, output_tokens, model=model_llm)
            st.subheader(f"Réponse gpt-4o (toute la doc) :")
            st.write(answer)
            st.info(f"Coût estimé : ${cost} | Temps de réponse : {elapsed:.2f}s")
        except Exception as e:
            st.error(f"Erreur gpt-4o : {e}")