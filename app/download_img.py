import os
import json
import requests
from urllib.parse import urlparse

# Chemin du fichier source
BASE_DIR = os.path.dirname(__file__)
INPUT_PATH = os.path.join(BASE_DIR, "data", "list_images_with_interfaces.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "images")

# Charger la liste des images
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    images = json.load(f)

for img in images:
    url = img["url"]
    interface = img.get("interface", "no_interface")
    img_id = img["id"]
    # Créer le dossier de l'interface si besoin
    interface_dir = os.path.join(OUTPUT_DIR, interface)
    os.makedirs(interface_dir, exist_ok=True)
    # Déduire l'extension du fichier depuis l'URL
    ext = os.path.splitext(urlparse(url).path)[1]
    if not ext:
        ext = ".jpg"
    # Nom du fichier
    filename = f"{img_id}{ext}"
    filepath = os.path.join(interface_dir, filename)
    # Télécharger si pas déjà présent
    if not os.path.exists(filepath):
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            with open(filepath, "wb") as out:
                out.write(r.content)
            print(f"Téléchargé : {filepath}")
        except Exception as e:
            print(f"Erreur pour {url} : {e}")
    else:
        print(f"Déjà présent : {filepath}")