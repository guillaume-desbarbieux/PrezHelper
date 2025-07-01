import streamlit as st
from rag_logic import ask_question
from llm_logic import generate_answer_ollama

st.set_page_config(page_title="RAG Prezevent - Recherche", layout="centered")
st.title("Recherche RAG Prezevent")

# 1. L'utilisateur pose une question
question = st.text_input("Posez votre question :", "")

# Pour stocker les résultats RAG entre deux interactions
if 'results' not in st.session_state:
    st.session_state['results'] = None
if 'last_question' not in st.session_state:
    st.session_state['last_question'] = None

if question:
    if question != st.session_state['last_question']:
        with st.spinner("Recherche en cours..."):
            results = ask_question(question)
        st.session_state['results'] = results
        st.session_state['last_question'] = question
    else:
        results = st.session_state['results']
    st.subheader("Passages pertinents extraits :")
    scores = results.get('scores', [[100.0]*len(results['documents'][0])])[0]
    for i, doc in enumerate(results['documents'][0]):
        meta = results['metadatas'][0][i]
        score = scores[i] if i < len(scores) else None
        title = meta.get('title', 'Sans titre')
        url = meta.get('source', 'N/A')
        with st.expander(f"{i+1}. Pertinence : {score:.1f}% | Titre : {title} | URL : {url}"):
            st.write(doc)

    # 2. L'utilisateur peut choisir de générer une réponse LLM à partir des passages
    if st.button("Générer une réponse LLM à partir de ces passages"):
        with st.spinner("Génération de la réponse par le LLM..."):
            # Ajout du score, titre, catégorie et url dans le contexte envoyé au LLM
            context = "\n\n".join([
                f"[Pertinence : {scores[i]:.1f}%]\n[Titre : {meta.get('title', 'Sans titre')}]\n[Catégorie : {meta.get('category_title', 'N/A')}]\n[URL : {meta.get('source', 'N/A')}]\n{doc}"
                for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0]))
            ])
            answer = generate_answer_ollama(question, context)
        st.subheader("Réponse générée par LLM :")
        st.write(answer)
