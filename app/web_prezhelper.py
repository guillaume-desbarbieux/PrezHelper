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
        st.markdown(f"**Passage {i+1}** — Pertinence : {score:.1f}%")
        st.write(doc)
        st.caption(f"Source : {meta.get('source', 'N/A')}")

    # 2. L'utilisateur peut choisir de générer une réponse LLM à partir des passages
    if st.button("Générer une réponse LLM à partir de ces passages"):
        with st.spinner("Génération de la réponse par le LLM..."):
            context = "\n\n".join(results['documents'][0])
            answer = generate_answer_ollama(question, context)
        st.subheader("Réponse générée par LLM :")
        st.write(answer)
