# Importation des modules nécessaires
import streamlit as st
import openai

# Configuration de la page Streamlit
st.set_page_config(page_title="PrezHelper IA", layout="centered")
st.title("PrezHelper IA")

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
    with open("data/corpus_llm_gpt_optimized.txt", "r", encoding="utf-8") as f:
        return f.read()
corpus = load_corpus()

# Champ de saisie pour la question de l'utilisateur
question = st.text_input("Posez votre question :", "")

# Clé API OpenAI
openai_api_key = st.sidebar.text_input("Clé API OpenAI", type="password")

if question:
    if st.button("Générer une réponse LLM à partir de la documentation"):
        with st.spinner("Génération de la réponse par ChatGPT à partir de la documentation complète..."):
            # Construction du prompt complet
            full_prompt = (
                prompt_intro +
                "\n\n[QUESTION UTILISATEUR]:\n" +
                question +
                "\n\n[CONTEXTE DOCUMENTAIRE]:\n" +
                corpus +
                "\n\n[REPONSE ATTENDUE]:\n"
            )
            if not openai_api_key:
                st.error("Veuillez renseigner votre clé API OpenAI dans la sidebar.")
            else:
                try:
                    client = openai.OpenAI(api_key=openai_api_key)
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": prompt_intro},
                            {"role": "user", "content": full_prompt}
                        ],
                        max_tokens=800,
                        temperature=0.2
                    )
                    answer = response.choices[0].message.content
                    st.success("Réponse générée par ChatGPT ci-dessous.")
                    st.subheader("Réponse générée par ChatGPT :")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Erreur lors de l'appel à l'API OpenAI : {e}")
