# Importation des modules nécessaires
import streamlit as st
import openai
import tiktoken
import time
from sentence_transformers import SentenceTransformer, util
import re
import json
import os
from datetime import datetime
import streamlit.components.v1 as components

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
    try:
        return SentenceTransformer("msmarco-MiniLM-L6-cos-v5")
    except Exception as e:
        st.error("❌ Le modèle d'embedding 'msmarco-MiniLM-L6-cos-v5' n'a pas pu être chargé.\n"\
                 "Vérifiez votre connexion Internet ou pré-téléchargez le modèle pour un usage offline.\n"\
                 "Pour pré-télécharger le modèle, exécutez sur une machine connectée :\n"
                 "\n    from sentence_transformers import SentenceTransformer\n"
                 "    SentenceTransformer('msmarco-MiniLM-L6-cos-v5')\n"
                 "\nCopiez ensuite le dossier du cache HuggingFace (généralement ~/.cache/huggingface ou C:/Users/<user>/.cache/huggingface) sur la machine cible.")
        return None

@st.cache_data(show_spinner=False)
def load_corpus_docs():
    # Charge la liste de documents individuels pré-découpés
    with open("data/corpus_docs.json", "r", encoding="utf-8") as f:
        docs = json.load(f)
    return docs
corpus_docs = load_corpus_docs()

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

HISTO_PATH = "data/historique_llm.jsonl"

def save_history(entry):
    os.makedirs(os.path.dirname(HISTO_PATH), exist_ok=True)
    with open(HISTO_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# --- Affichage des 4 boutons côte à côte sous la zone de question ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    reform_btn = st.button("Reformuler la question (optionnel)")
with col2:
    rag_btn = st.button("Rechercher les documents pertinents (RAG)")
with col3:
    llm_btn = st.button("Générer une réponse LLM à partir de la documentation")
with col4:
    llm_all_btn = st.button("Générer une réponse LLM avec TOUTE la documentation")

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
                model="gpt-4o",
                messages=messages,
                max_tokens=100,
                temperature=0.2
            )
            elapsed = time.time() - start
            question_recherche = response.choices[0].message.content.strip()
            input_tokens = count_tokens(messages, model="gpt-4o")
            output_tokens = count_tokens(question_recherche, model="gpt-4o")
            total_tokens = input_tokens + output_tokens
            cost = estimate_cost(input_tokens, output_tokens, model="gpt-4o")
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

# Affichage persistant de la question reformulée et du coût si disponible
reformulation_infos = st.session_state.get('reformulation_infos')
if reformulation_infos:
    st.info(f"**Question reformulée :** {reformulation_infos['question_reformulee']}\n\nCoût estimé : ${reformulation_infos['cost']} | Temps de réponse : {reformulation_infos['elapsed']:.2f}s")

# --- Étape 2 : Recherche de documents pertinents ---
if rag_btn:
    with st.spinner("Recherche des documents les plus pertinents dans la base documentaire..."):
        start = time.time()
        question_recherche = st.session_state.get('question_recherche') or question
        embedder = get_embedder()
        if embedder is None:
            st.stop()
        q_emb = embedder.encode(question_recherche, convert_to_tensor=True)
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
        elapsed = time.time() - start
        # --- Sauvegarde historique ---
        save_history({
            "timestamp": datetime.now().isoformat(),
            "type": "rag_search",
            "question": question,
            "question_recherche": question_recherche,
            "params": {
                "top_k": top_k,
                "min_score": min_score,
                "relative_margin": relative_margin,
                "alpha": alpha
            },
            "docs": st.session_state['top_docs'],
            "scores": st.session_state['top_scores'],
            "delai": elapsed
        })
        st.success(f"{len(st.session_state['top_docs'])} document(s) pertinent(s) trouvé(s).", icon="✅")
        for i, doc in enumerate(st.session_state['top_docs']):
            titre_match = re.search(r"Titre\\s*:\\s*(.*)", doc)
            titre = titre_match.group(1).strip() if titre_match else "(Titre inconnu)"
            st.markdown(f"**{i+1}. {titre}**  ")
            st.markdown(f"Score mixte : `{st.session_state['top_scores'][i]:.4f}`")
            st.text_area("Contenu", doc, height=120)

# --- Étape 3 : Génération de la réponse LLM ---
if llm_btn and question and selected:
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
            cols = st.columns(len(selected))
            for idx, model_name in enumerate(selected):
                with cols[idx]:
                    try:
                        client = openai.OpenAI(api_key=openai_api_key)
                        rag_prompt = (
                            f"Voici la question d'un utilisateur :\n{st.session_state.get('question_recherche') or question}\n\n"
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
                        st.subheader(f"Réponse {model_name} :")
                        st.write(answer)
                        st.info(f"Coût estimé : ${cost} | Temps de réponse : {elapsed:.2f}s")
                        # --- Sauvegarde historique ---
                        save_history({
                            "timestamp": datetime.now().isoformat(),
                            "type": "llm_rag",
                            "question": question,
                            "question_reformulee": st.session_state.get('question_recherche'),
                            "model": model_name,
                            "params": {
                                "top_k": top_k,
                                "min_score": min_score,
                                "relative_margin": relative_margin,
                                "alpha": alpha
                            },
                            "prompt_intro": prompt_intro,
                            "prompt_rag": rag_prompt,
                            "docs": st.session_state['top_docs'],
                            "reponse": answer,
                            "cout": cost,
                            "delai": elapsed
                        })
                    except Exception as e:
                        st.error(f"Erreur {model_name} : {e}")
# --- Étape 4 : Génération de la réponse LLM avec toute la documentation ---
if llm_all_btn and question and selected:
    with st.spinner("Génération de la réponse par ChatGPT à partir de toute la documentation..."):
        rag_corpus = corpus  # toute la documentation brute
        cols = st.columns(len(selected))
        for idx, model_name in enumerate(selected):
            with cols[idx]:
                try:
                    client = openai.OpenAI(api_key=openai_api_key)
                    rag_prompt = (
                        f"Voici la question d'un utilisateur :\n{question}\n\n"
                        f"Voici toute la documentation à ta disposition :\n{rag_corpus}\n\n"
                        "ATTENTION : ne fais pas d'invention. Ne réponds que si la réponse est clairement présente dans la documentation."
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
                    st.subheader(f"Réponse {model_name} (toute la doc) :")
                    st.write(answer)
                    st.info(f"Coût estimé : ${cost} | Temps de réponse : {elapsed:.2f}s")
                    # --- Sauvegarde historique ---
                    save_history({
                        "timestamp": datetime.now().isoformat(),
                        "type": "llm_all",
                        "question": question,
                        "model": model_name,
                        "params": {},
                        "prompt_intro": prompt_intro,
                        "prompt_rag": rag_prompt,
                        "reponse": answer,
                        "cout": cost,
                        "delai": elapsed
                    })
                except Exception as e:
                    st.error(f"Erreur {model_name} : {e}")

# --- Bouton pour afficher l'historique dans un nouvel onglet ---
def show_history():
    if not os.path.exists(HISTO_PATH):
        st.sidebar.info("Aucun historique disponible.")
        return
    with open(HISTO_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    st.title("Historique des échanges LLM")
    for line in lines[::-1]:  # affichage du plus récent au plus ancien
        entry = json.loads(line)
        with st.expander(f"{entry.get('timestamp', '')} | {entry.get('type', '')}"):
            st.markdown(f"**Question :** {entry.get('question','')}")
            if 'question_reformulee' in entry:
                st.markdown(f"**Question reformulée :** {entry['question_reformulee']}")
            st.markdown(f"**Paramètres :** {json.dumps(entry.get('params',{}) , ensure_ascii=False)}")
            if 'prompt_intro' in entry:
                st.markdown(f"**Prompt system :**\n```")
                st.markdown(entry['prompt_intro'])
                st.markdown("```")
            if 'prompt_rag' in entry:
                st.markdown(f"**Prompt user :**\n```")
                st.markdown(entry['prompt_rag'])
                st.markdown("```")
            if 'docs' in entry:
                st.markdown(f"**Documents pertinents :**")
                for i, doc in enumerate(entry['docs']):
                    st.text_area(f"Doc {i+1}", doc, height=80)
            if 'reponse' in entry:
                st.markdown(f"**Réponse :**\n{entry['reponse']}")
            st.markdown(f"**Coût :** ${entry.get('cout','')} | **Délai :** {entry.get('delai',''):.2f}s")

if st.sidebar.button("Afficher l'historique"):
    url = "/llm_history"
    js = f"window.open('{url}','_blank')"
    st.sidebar.markdown(f'<a href="{url}" target="_blank">Ouvrir l\'historique dans un nouvel onglet</a>', unsafe_allow_html=True)
    components.html(f"<script>{js}</script>", height=0)

# Affichage de l'historique si l'URL correspond
if st.query_params.get('page', [''])[0] == 'llm_history' or st.query_params.get('llm_history', [''])[0] == '':
    if st.query_params.get('llm_history', [''])[0] == '':
        # Cas d'accès direct via /llm_history (sans ?page=llm_history)
        st.set_page_config(page_title="Historique LLM", layout="centered")
        st.title("Historique des échanges LLM")
        show_history()
        st.stop()
    elif st.query_params.get('page', [''])[0] == 'llm_history':
        st.set_page_config(page_title="Historique LLM", layout="centered")
        st.title("Historique des échanges LLM")
        show_history()
        st.stop()
if st.session_state.get('show_llm_history'):
    show_history()
