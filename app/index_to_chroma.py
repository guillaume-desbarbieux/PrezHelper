from app.utils import flatten_article
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Charger les articles scrappés
with open("data/prezevent_scraped.json", "r", encoding="utf-8") as f:
    all_articles = json.load(f)

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
all_chunks = []

for article in all_articles:
    full_text = flatten_article(article)
    metadatas = {"source": article["url"], "title": article["title"]}
    chunks = splitter.create_documents([full_text], metadatas=[metadatas])
    all_chunks.extend(chunks)

# Exemple d'affichage pour vérification
print(f"Nombre total de chunks : {len(all_chunks)}")
for i, chunk in enumerate(all_chunks[:5]):
    print(f"Chunk {i+1} (source: {chunk.metadata['source']}) :\n{chunk.page_content[:200]}\n---")

# À ce stade, tu peux indexer all_chunks dans Chroma
# (chroma.add_documents([chunk.page_content for chunk in all_chunks], [chunk.metadata for chunk in all_chunks]))
