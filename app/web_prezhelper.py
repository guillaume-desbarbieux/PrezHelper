import streamlit as st
from rag_logic import ask_question

st.set_page_config(page_title="RAG Prezevent - Recherche", layout="centered")
st.title("Recherche RAG Prezevent")

question = st.text_input("Posez votre question :", "")

if question:
    with st.spinner("Recherche en cours..."):
        results = ask_question(question)
    st.subheader("Résultats :")
    for i, doc in enumerate(results['documents'][0]):
        meta = results['metadatas'][0][i]
        st.markdown(f"**Réponse {i+1} :**")
        st.write(doc)
        st.caption(f"Source : {meta.get('source', 'N/A')}")
