import json
import re
import os

INPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "corpus_dict.json")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "list_images.json")

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    corpus = json.load(f)

results = []
img_id = 1

for doc in corpus:
    text = doc.get("text", "")
    # Trouve toutes les paires [Image URL: ...] suivies de [Image Description: ...]
    pattern = re.compile(
        r"\[Image URL: ([^\]\n]+)\]\s*\[Image Description: ([^\]]+)\]",
        re.MULTILINE
    )
    for match in pattern.finditer(text):
        url = match.group(1).strip()
        desc = match.group(2).strip()
        results.append({
            "id": f"img_{img_id}",
            "url": url,
            "description": desc
        })
        img_id += 1

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"{len(results)} images extraites dans {OUTPUT_PATH}")