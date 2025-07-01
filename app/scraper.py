from app.utils import scrape_category_page, scrape_article
import json
import requests
from bs4 import BeautifulSoup

# Récupération automatique des URLs de catégories
homepage = "https://help.prezevent.com/"
resp = requests.get(homepage)
soup = BeautifulSoup(resp.text, 'html.parser')
category_urls = [a['href'] for a in soup.find_all('a', class_='category') if a.has_attr('href')]

all_results = []
for url in category_urls:
    results = scrape_category_page(url, scrape_article)
    all_results.extend(results)

# Stockage pour usage futur (exemple : sauvegarde JSON, à adapter pour DB réelle)
with open("data/prezevent_scraped.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"{len(all_results)} articles scrappés et stockés.")

# Affichage d'un résumé pour vérification
print("\nRésumé des articles :")
print(f"Nombre d'articles : {len(all_results)}")
for art in all_results:
    print(f"- URL : {art.get('url')}")
    print(f"  Titre : {art.get('title')}")
    print(f"  Nombre d'éléments : {len(art.get('content', []))}")


