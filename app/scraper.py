import requests
from bs4 import BeautifulSoup

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
            elements.append({'type': 'image', 'src': img_url, 'alt': alt})

        elif elem.name == 'iframe':
            src = elem.get('src')
            elements.append({'type': 'iframe', 'src': src})

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
