import openai
import requests
from time import sleep

# 👉 Renseigne ta clé API ici (ou utilise une variable d'environnement)
openai.api_key = "sk-..."

# 🔢 Modèle multimodal
MODEL = "gpt-4o"

# 🖼️ Liste des URLs d'images à analyser
image_urls = [
    "https://help.prezevent.com/images/mes-campagnes.png",
    # Ajoute ici d'autres URLs...
]

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
        
    except Exception as e:
        print(f"❌ Erreur pour {url} : {e}")
    
    sleep(1.5)  # pause recommandée pour éviter les dépassements de quota
