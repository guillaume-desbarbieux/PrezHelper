# Importation des modules nécessaires
import streamlit as st
from rag_logic import ask_question  # Fonction pour interroger la base documentaire (RAG)
from llm_logic import generate_answer_ollama  # Fonction pour générer une réponse avec le LLM

# Configuration de la page Streamlit
st.set_page_config(page_title="RAG Prezevent - Recherche", layout="centered")
st.title("Recherche RAG Prezevent")

# Sidebar pour les paramètres avancés
st.sidebar.header("Paramètres avancés")

# Paramètre : nombre de documents à afficher (top k)
top_k = st.sidebar.number_input(
    "Nombre de documents à afficher (top k)", min_value=1, max_value=20, value=5, step=1
)
# Paramètre : seuil de pertinence minimal pour filtrer les passages
seuil_pertinence = st.sidebar.slider(
    "Seuil de pertinence minimal (%)", min_value=0, max_value=100, value=50, step=1
)
# Paramètre : introduction personnalisable du prompt envoyé au LLM
prompt_intro = st.sidebar.text_area(
    "Introduction du prompt (avant la question)",
    value="Tu es un assistant expert de la documentation Prezevent. Réponds uniquement en français, de façon claire et concise, en t'appuyant exclusivement sur les passages suivants extraits de la documentation. Si la réponse n'est pas présente dans les extraits, indique-le poliment."
)

# Champ de saisie pour la question de l'utilisateur
question = st.text_input("Posez votre question :", "")

# Initialisation des variables de session pour stocker les résultats et la dernière question
if 'results' not in st.session_state:
    st.session_state['results'] = None
if 'last_question' not in st.session_state:
    st.session_state['last_question'] = None

# Si une question est posée
if question:
    # Si la question est nouvelle, on lance la recherche documentaire
    if question != st.session_state['last_question']:
        with st.spinner("Recherche des passages pertinents dans la base documentaire..."):
            results = ask_question(question, top_k=top_k)
        # DEBUG : Affichage brut des résultats Chroma
        # (debug retiré)
        if not results or not isinstance(results, dict):
            st.error("Erreur interne : la recherche n'a pas retourné de résultat exploitable.")
            st.stop()
        if 'error' in results:
            st.error(f"Erreur ChromaDB : {results['error']}")
        if not results or 'documents' not in results or not results['documents'] or not results['documents'][0]:
            st.error("Aucun résultat trouvé ou la base documentaire n'est pas accessible. Veuillez vérifier la base ou réessayer plus tard.")
            st.session_state['results'] = None
            st.session_state['last_question'] = question
            st.stop()
        st.success("Recherche terminée. Passages extraits affichés ci-dessous.")
        st.session_state['results'] = results
        st.session_state['last_question'] = question
    else:
        # Sinon, on réutilise les résultats précédents
        results = st.session_state['results']
        st.info("Affichage des résultats précédemment trouvés pour cette question.")
    st.subheader("Passages pertinents extraits :")
    passages = []  # Liste des passages pertinents à afficher
    scores = results.get('scores', [[100.0]*len(results['documents'][0])])[0]
    best_score = None  # Pour stocker le meilleur score trouvé
    # Parcours des documents retournés
    for i, doc in enumerate(results['documents'][0]):
        meta = results['metadatas'][0][i]
        score = scores[i] if i < len(scores) else None
        # Mise à jour du meilleur score
        if best_score is None or (score is not None and score > best_score):
            best_score = score
        # On ignore les passages sous le seuil de pertinence
        if score is not None and score < seuil_pertinence:
            continue
        passages.append((i, doc, meta, score))
        title = meta.get('title', 'Sans titre')
        url = meta.get('source', 'N/A')
        # Affichage du passage dans un expander avec infos
        with st.expander(f"{i+1}. Pertinence : {score:.1f}% | Titre : {title} | URL : {url}"):
            st.write(doc)
            if url != 'N/A':
                st.markdown(f"[Ouvrir la source dans un nouvel onglet]({url})", unsafe_allow_html=True)
            # Affichage des images associées à ce passage (dans l'expander)
            image_urls_str = meta.get('image_urls', '')
            image_urls = [url.strip() for url in image_urls_str.split(',') if url.strip()]
            for img_url in image_urls:
                if img_url.startswith('//'):
                    img_url = 'https:' + img_url
                st.image(img_url)

    # Si aucun passage pertinent n'est trouvé, message d'erreur et astuce
    if not passages:
        st.error("Aucun passage pertinent n'a été trouvé avec le seuil actuel. Veuillez contacter le support si le problème persiste.")
        if best_score is not None:
            st.warning(f"Astuce : essayez de diminuer le seuil de pertinence (meilleur score trouvé : {best_score:.1f}%)")
    else:
        # Si des passages sont trouvés, possibilité de générer une réponse LLM
        if st.button("Générer une réponse LLM à partir de ces passages"):
            with st.spinner("Génération de la réponse par le LLM à partir des passages extraits..."):
                # Construction du prompt structuré avec balises
                prompt = (
                    prompt_intro +
                    "\n<QUESTION>\n" +
                    f"{question}\n" +
                    "</QUESTION>\n" +
                    "\n<CONTEXT>\n"
                )
                # Construction du contexte documentaire à partir des passages filtrés
                context = "\n\n".join([
                    f"[Pertinence : {score:.1f}%]\n[Titre : {meta.get('title', 'Sans titre')}]\n[Catégorie : {meta.get('category_title', 'N/A')}]\n[URL : {meta.get('source', 'N/A')}]\n{doc}"
                    for (_, doc, meta, score) in passages
                ])
                full_prompt = prompt + context + "\n</CONTEXT>"
                # Appel au LLM pour générer la réponse
                answer = generate_answer_ollama(question, full_prompt)
            st.success("Réponse générée par le LLM ci-dessous.")
            st.subheader("Réponse générée par LLM :")
            st.write(answer)
