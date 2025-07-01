from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import torch
import cv2
import numpy as np
from typing import List

# Chargement du modèle BLIP (à faire une seule fois)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def describe_image_from_url(url: str) -> str:
    """
    Télécharge une image et retourne une description générée par BLIP.
    """
    try:
        image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
        inputs = processor(image, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"[Erreur description image: {e}]"

def describe_video_from_url(url: str, frame_interval: int = 15) -> str:
    """
    Télécharge une vidéo depuis une URL, extrait une frame toutes les `frame_interval` secondes,
    génère une description pour chaque frame avec BLIP, puis fusionne les descriptions.
    """
    import tempfile, os
    import requests
    from PIL import Image

    try:
        # Télécharger la vidéo dans un fichier temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            r = requests.get(url, stream=True)
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    tmp.write(chunk)
            video_path = tmp.name

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps else 0
        frame_times = np.arange(0, duration, frame_interval)
        descriptions: List[str] = []

        for t in frame_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if not ret:
                continue
            # Convertir la frame en image PIL
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # Utiliser BLIP pour décrire la frame
            desc = describe_image_pil(img)
            descriptions.append(desc)
        cap.release()
        os.remove(video_path)
        if not descriptions:
            return "[Aucune frame n'a pu être décrite]"
        # Fusionner les descriptions
        return " ".join(descriptions)
    except Exception as e:
        return f"[Erreur description vidéo: {e}]"

def describe_image_pil(img):
    """
    Décrit une image PIL avec BLIP (utilisé pour la vidéo).
    """
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    inputs = processor(img, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def scrape_articles(urls, scraper_func, category_title=None, category_description=None):
    """
    Prend une liste d'URLs et une fonction de scraping (ex: scrape_article),
    ajoute le titre et la description de la catégorie à chaque résultat si fournis.
    """
    results = []
    for url in urls:
        res = scraper_func(url)
        if res:
            if category_title:
                res['category_title'] = category_title
            if category_description:
                res['category_description'] = category_description
            results.append(res)
    return results

def scrape_category_page(category_url, scraper_func):
    """
    Scrappe une page de catégorie : récupère le titre, la description et la liste des URLs d'articles,
    puis appelle scrape_articles avec ces informations.
    """
    import requests
    from bs4 import BeautifulSoup
    from urllib.parse import urljoin

    response = requests.get(category_url)
    if response.status_code != 200:
        print(f"Erreur lors du chargement de {category_url}")
        return []
    soup = BeautifulSoup(response.text, 'html.parser')

    content = soup.find('div', class_='contentWrapper')
    if not content:
        print("Pas de bloc contentWrapper trouvé.")
        return []

    title = content.find('h1').get_text(strip=True) if content.find('h1') else 'Sans titre'
    desc_tag = content.find('p', class_='descrip')
    description = desc_tag.get_text(strip=True) if desc_tag else ''

    article_list = content.find('ul', class_='articleList')
    if not article_list:
        print("Pas de liste d'articles trouvée.")
        return []
    article_links = article_list.find_all('a')
    article_urls = [urljoin(category_url, a['href']) for a in article_links if a.has_attr('href')]

    return scrape_articles(article_urls, scraper_func, category_title=title, category_description=description)
