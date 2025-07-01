# PrezHelper – Pipeline RAG Documentation Prezevent

Ce projet permet de scraper, enrichir, chunker et indexer la documentation Prezevent (HTML) dans ChromaDB pour un usage RAG (Retrieval-Augmented Generation), avec gestion automatique des images et interfaces de questionnement (web et terminal).

## Fonctionnalités principales
- **Scraping** des catégories et articles Prezevent (HTML)
- **Génération automatique de descriptions d’images** (modèle BLIP)
- **Structuration avancée des métadonnées** (source, images, etc.)
- **Chunking** des articles (langchain)
- **Indexation** dans ChromaDB
- **Pipeline unitaire** automatisé (`full_scrape_and_index`)
- **Interface web** (Flask) et **interface terminal** pour interroger la base RAG
- **Code modulaire** : logique RAG centralisée dans `rag_logic.py`

## Structure du projet
- `app/ingest_prezevent_docs.py` : lance le pipeline complet (scraping → indexation)
- `app/rag_logic.py` : logique de questionnement RAG (fonction `ask_question`)
- `app/web_prezhelper.py` : interface web pour poser des questions
- `app/ask_prezevent.py` : interface terminal (CLI)
- `app/utils.py` : fonctions de scraping, enrichissement, chunking, indexation
- `requirements.txt` : dépendances Python (torch, transformers, chromadb, flask, etc.)

## Utilisation

### 1. Ingestion de la documentation
```bash
python app/ingest_prezevent_docs.py
```

### 2. Interface web (Streamlit)
```bash
pip install streamlit
streamlit run app/web_prezhelper.py
```
Accès sur http://localhost:8501

### 3. Interface terminal (CLI)
```bash
python app/ask_prezevent.py
```

## Personnalisation
- La logique RAG est centralisée dans `rag_logic.py` : toute nouvelle interface (API, chatbot, etc.) peut l’utiliser sans dupliquer le code.
- Le pipeline peut être adapté pour d’autres sources/documentations.

## Dépendances principales
- torch, transformers, Pillow, opencv-python, beautifulsoup4, chromadb, langchain, flask

## Notes
- Le scraping et l’indexation nécessitent un accès internet.
- Le pipeline gère automatiquement la génération de descriptions d’images.
