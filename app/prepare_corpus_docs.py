import re
import json

# Ce script lit le corpus brut et extrait chaque document individuellement
# Usage : python prepare_corpus_docs.py

INPUT_PATH = "data/corpus_llm.txt"
OUTPUT_PATH = "data/corpus_docs.json"

def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        txt = f.read()
    docs = re.split(r"\[\/DOCUMENT\]", txt)
    docs = [d.strip() for d in docs if d.strip()]
    # Optionnel : retirer le tag de fin si présent
    docs = [d if d.endswith("[/DOCUMENT]") else d+"\n[/DOCUMENT]" for d in docs]
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    print(f"{len(docs)} documents sauvegardés dans {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
