import openai
import requests
import json
import re
from time import sleep

# 👉 Renseigne ta clé API ici (ou utilise une variable d'environnement)
openai.api_key = "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"

# 🔢 Modèle multimodal
MODEL = "gpt-4o"

# 🖼️ Liste des URLs d'images à analyser
with open("data/corpus_llm_test.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

# Dictionnaire pour stocker les descriptions générées par URL
image_descriptions = {}

# 📥 Prompt d’analyse de l’image
base_prompt = """
Tu es un expert UX assistant. Ton rôle est de décrire de manière détaillée l’interface visible dans cette image d’un outil web (Prezevent).
Fournis une description fonctionnelle précise, incluant :
- la structure de l’interface (menus, boutons, zones actives…)
- les actions disponibles
- l’état ou le contexte affiché
Utilise des puces pour chaque point.
Ne fais pas de résumé général. Sois utile à un assistant IA qui ne peut pas afficher l’image mais doit guider un utilisateur qui la voit.
"""

# 🖼️ Liste des URLs d'images à analyser
image_urls = []
for doc in corpus:
    matches = re.findall(r'\[Image URL: ([^\]\s]+)\]', doc.get('text', ''))
    image_urls.extend([m.strip() for m in matches])
image_urls = list(sorted(set(image_urls)))

# 🔁 Boucle de génération
for i, url in enumerate(image_urls):
    print(f"\n⏳ Traitement de l’image {i+1}/{len(image_urls)} : {url}")
    try:
        response = openai.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Tu es un assistant expert en interfaces web."},
                {"role": "user", "content": base_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": url,
                                "detail": "auto"
                            }
                        }
                    ]
                }
            ],
            max_tokens=800,
            temperature=0.3
        )
        content = response.choices[0].message.content
        print(f"\n📄 Description générée :\n{content}")
        image_descriptions[url] = content
    except Exception as e:
        print(f"❌ Erreur pour {url} : {e}")
    sleep(1.5)

# Ajout des descriptions dans le corpus
for doc in corpus:
    def insert_descriptions(text):
        def replacer(match):
            url = match.group(1).strip()
            desc = image_descriptions.get(url)
            if desc:
                return f"[Image URL: {url}]\n[Image Description: {desc}]"
            else:
                return match.group(0)
        return re.sub(r'\[Image URL: ([^\]\s]+)\]', replacer, text)
    doc['text'] = insert_descriptions(doc.get('text', ''))

with open("data/corpus_llm_test.json", "w", encoding="utf-8") as f:
    json.dump(corpus, f, ensure_ascii=False, indent=2)

