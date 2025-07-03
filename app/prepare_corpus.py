import json
import os

INPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "corpus_llm.json")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "corpus_ready.json")

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    corpus = json.load(f)

docs = []
for doc in corpus:
    title = doc.get("title", "").strip()
    url = doc.get("url", "").strip()
    text = doc.get("text", "")

    # Nettoyer le contenu : retirer toutes les lignes [Image URL: ...] et supprimer les deux premières lignes
    lines = text.splitlines()
    # Supprimer les deux premières lignes si elles existent
    lines = lines[2:] if len(lines) > 2 else []
    cleaned_lines = []
    for line in lines:
        if not line.strip().startswith("[Image URL:"):
            cleaned_lines.append(line)
    cleaned_text = "\n".join(cleaned_lines).strip()

    # Formater la sortie
    doc_str = (
        "[DOCUMENT]\n"
        f"Titre : {title}\n"
        f"URL : {url}\n"
        f"Contenu :\n{cleaned_text}\n"
        "[/DOCUMENT]"
    )
    docs.append(doc_str)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(docs, f, ensure_ascii=False, indent=2)

print(f"Conversion terminée. {len(docs)} documents écrits dans {OUTPUT_PATH}")