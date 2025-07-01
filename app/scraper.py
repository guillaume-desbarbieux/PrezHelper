import requests
from bs4 import BeautifulSoup
from .utils import describe_image_from_url, describe_video_from_url

def scrape_article(url):
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Erreur lors du chargement de {url}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')

    # Titre principal
    title = soup.find('h1').get_text(strip=True) if soup.find('h1') else 'Sans titre'

    # Corps de l'article
    article_body = soup.find('div', class_='fullArticle')  # À adapter selon la classe exacte

    if not article_body:
        print("Pas de contenu trouvé dans cette page.")
        return None

    elements = []

    for elem in article_body.find_all(['h3', 'p', 'img', 'iframe']):
        if elem.name == 'p':
            elements.append({'type': 'paragraph', 'content': elem.get_text(strip=True)})

        elif elem.name == 'h3':
            elements.append({'type': 'heading', 'content': elem.get_text(strip=True)})

        elif elem.name == 'img':
            img_url = elem.get('src')
            alt = elem.get('alt', '')
            # Générer une description automatique de l'image
            description = describe_image_from_url(img_url)
            elements.append({'type': 'image', 'src': img_url, 'alt': alt, 'description': description})

        elif elem.name == 'iframe':
            src = elem.get('src')
            # Générer une description automatique de la vidéo si c'est un lien vidéo direct
            description = describe_video_from_url(src)
            elements.append({'type': 'iframe', 'src': src, 'description': description})

    return {
        'url': url,
        'title': title,
        'content': elements
    }

# Exemple d'utilisation
article_url = 'https://help.prezevent.com/article/18-site-web-evenementiel'  # remplace avec une vraie URL
result = scrape_article(article_url)

# Affichage
import json
print(json.dumps(result, indent=2, ensure_ascii=False))
if result:
    print(f"Titre: {result['title']}")
    for element in result['content']:
        if element['type'] == 'paragraph':
            print(f"Paragraphe: {element['content']}")
        elif element['type'] == 'heading':
            print(f"Titre: {element['content']}")
        elif element['type'] == 'image':
            print(f"Image: {element['src']} (alt: {element['alt']})")
        elif element['type'] == 'iframe':
            print(f"Iframe: {element['src']}")


