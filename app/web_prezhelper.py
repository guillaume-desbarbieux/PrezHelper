# Importation des modules nécessaires
import streamlit as st
import openai

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
