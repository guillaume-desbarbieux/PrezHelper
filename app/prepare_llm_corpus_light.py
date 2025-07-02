import re

INPUT_PATH = "data/corpus_llm.txt"
OUTPUT_PATH = "data/corpus_llm_light.txt"

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    corpus = f.read()

# Supprime les lignes contenant des images, descriptions d'images, ou l'URL de l'article
lines = corpus.splitlines()
filtered = []
for line in lines:
    if line.strip().startswith("[Image"):
        continue
    if line.strip().startswith("[Image URL"):
        continue
    if line.strip().startswith("URL :"):
        continue
    filtered.append(line)

# Réduit le séparateur d'article à une simple ligne
text = "\n---\n".join(
    part.strip() for part in re.split(r"=+", "\n".join(filtered)) if part.strip()
)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(text)

print(f"Corpus light exporté dans {OUTPUT_PATH}")
