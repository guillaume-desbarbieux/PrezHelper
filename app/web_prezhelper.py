# Importation des modules nécessaires
import streamlit as st
from llm_logic import generate_answer_ollama  # Fonction pour générer une réponse avec le LLM

# Configuration de la page Streamlit
st.set_page_config(page_title="RAG Prezevent - Recherche", layout="centered")
st.title("Recherche RAG Prezevent")

# Sidebar pour les paramètres avancés
st.sidebar.header("Paramètres avancés")

# Paramètre : introduction personnalisable du prompt envoyé au LLM
prompt_intro = st.sidebar.text_area(
    "Introduction du prompt (avant la question)",
    value="Tu es un assistant expert de la documentation Prezevent. Réponds uniquement en français, de façon claire et concise, en t'appuyant exclusivement sur la documentation ci-dessous. Si la réponse n'est pas présente, indique-le poliment."
)

# Chargement du corpus documentaire
@st.cache_data(show_spinner=False)
def load_corpus():
    with open("data/corpus_llm.txt", "r", encoding="utf-8") as f:
        return f.read()
corpus = load_corpus()

# Champ de saisie pour la question de l'utilisateur
question = st.text_input("Posez votre question :", "")

if question:
    if st.button("Générer une réponse LLM à partir de la documentation"):
        with st.spinner("Génération de la réponse par le LLM à partir de la documentation complète..."):
            # Construction du prompt complet
            full_prompt = (
                prompt_intro +
                "\n\nDOCUMENTATION:\n" +
                corpus +
                "\n\nQUESTION:\n" +
                question +
                "\n\nRéponds uniquement à partir de la documentation ci-dessus."
            )
            answer = generate_answer_ollama(question, full_prompt)
        st.success("Réponse générée par le LLM ci-dessous.")
        st.subheader("Réponse générée par LLM :")
        st.write(answer)
