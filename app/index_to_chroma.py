from app.utils import flatten_article
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb

# Charger les articles scrappés
with open("data/prezevent_scraped.json", "r", encoding="utf-8") as f:
    all_articles = json.load(f)

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
all_chunks = []

for article in all_articles:
    flat = flatten_article(article)
    text = flat["text"]
    metadata = flat["metadata"]
    chunks = splitter.create_documents([text], metadatas=[metadata])
    all_chunks.extend(chunks)

# Indexation dans Chroma
client = chromadb.Client()
collection = client.get_or_create_collection("prezevent_articles")

# Ajout des documents
collection.add_documents(
    documents=[chunk.page_content for chunk in all_chunks],
    metadatas=[chunk.metadata for chunk in all_chunks]
)

print(f"{len(all_chunks)} chunks indexés dans Chroma.")
