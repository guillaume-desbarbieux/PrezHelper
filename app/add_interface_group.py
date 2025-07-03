import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Chemins d'entrée et de sortie
BASE_DIR = os.path.dirname(__file__)
INPUT_PATH = os.path.join(BASE_DIR, "data", "list_images.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "list_images_with_interfaces.json")

# Charger le corpus
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    documents = json.load(f)

# Extraire descriptions d’interface depuis le champ "description"
descriptions = []
description_meta = []

for i, doc in enumerate(documents):
    desc = doc.get("description", "").strip()
    if desc:
        descriptions.append(desc)
        description_meta.append({
            "doc_index": i,
            "id": doc.get("id"),
            "url": doc.get("url")
        })

# Embedding des descriptions
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(descriptions)

# Matrice de similarité
sim_matrix = cosine_similarity(embeddings)

# Grouper par similarité > 0.85
THRESHOLD = 0.85
groups = []
used = set()
for i in range(len(sim_matrix)):
    if i in used:
        continue
    group = [i]
    used.add(i)
    for j in range(i + 1, len(sim_matrix)):
        if j not in used and sim_matrix[i][j] >= THRESHOLD:
            group.append(j)
            used.add(j)
    groups.append(group)

# Assigner un identifiant à chaque groupe
interface_groups = {}
for idx, group in enumerate(groups):
    interface_name = f"interface_{idx+1:03d}"
    for j in group:
        interface_groups[j] = interface_name

# Annoter les documents
for j, meta in enumerate(description_meta):
    interface = interface_groups.get(j)
    doc = documents[meta["doc_index"]]
    doc["interface"] = interface

# Sauvegarde du corpus annoté
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(documents, f, ensure_ascii=False, indent=2)