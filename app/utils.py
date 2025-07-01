from PIL import Image
import requests
from typing import List
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime

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
    Scrappe une page de catégorie : récupère le titre, la description et la liste des URLs d'articles,
    puis appelle scrape_articles avec ces informations.
    """
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

def scrape_article(url):
    """
    Scrappe un article individuel et retourne un dictionnaire structuré (titre, contenu, images, etc).
    """
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
            # On ne génère plus de description automatique, on stocke juste l'image et l'alt
            elements.append({'type': 'image', 'src': img_url, 'alt': alt})

        elif elem.name == 'iframe':
            src = elem.get('src')
            elements.append({'type': 'iframe', 'src': src, 'description': ''})  # Pas de description pour l'indexation RAG

    return {
        'url': url,
        'title': title,
        'content': elements
    }

def flatten_article(article):
    """
    Retourne un objet JSON avec le texte concaténé (titres, paragraphes, images sans description)
    et les métadonnées utiles (source, titre, date, urls images).
    """
    lines = []
    image_urls = []
    for block in article.get("content", []):
        if block["type"] in ["heading", "paragraph"]:
            lines.append(block["content"])
        elif block["type"] == "image":
            url = block.get("src", "")
            alt = block.get("alt", "")
            if alt:
                lines.append(f"<image-alt>{alt}</image-alt>")
            if url:
                image_urls.append(url)
    text = "\n\n".join(lines)
    metadata = {
        "source": article.get("url"),
        "title": article.get("title"),
        "scraped_at": datetime.utcnow().isoformat() + "Z",
        "image_urls": image_urls
    }
    return {"text": text, "metadata": metadata}

def full_scrape_and_index(collection_name="prezevent_articles"):
    """
    Pipeline complet : scrape toutes les catégories, articles, prépare les chunks et indexe dans Chroma.
    """
    import requests
    import json
    from bs4 import BeautifulSoup
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    import chromadb
    from utils import scrape_category_page, scrape_article, flatten_article

    print("[INFO] Récupération des URLs de catégories...")
    # 1. Récupère les URLs de catégories depuis la page d'accueil
    homepage = "https://help.prezevent.com/"
    resp = requests.get(homepage)
    soup = BeautifulSoup(resp.text, 'html.parser')
    category_urls = [a['href'] for a in soup.find_all('a', class_='category') if a.has_attr('href')]
    print(f"[INFO] {len(category_urls)} catégories trouvées.")

    # 2. Scrape tous les articles de chaque catégorie
    all_articles = []
    for i, url in enumerate(category_urls):
        print(f"[INFO] Scraping catégorie {i+1}/{len(category_urls)}: {url}")
        results = scrape_category_page(url, scrape_article)
        print(f"[INFO]  -> {len(results)} articles extraits.")
        all_articles.extend(results)
    print(f"[INFO] Total articles extraits : {len(all_articles)}")

    # 3. Découpe les articles en chunks pour l'indexation
    print("[INFO] Découpage des articles en chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    all_chunks = []
    for idx, article in enumerate(all_articles):
        flat = flatten_article(article)
        text = flat["text"]
        metadata = flat["metadata"]
        chunks = splitter.create_documents([text], metadatas=[metadata])
        all_chunks.extend(chunks)
        if (idx+1) % 10 == 0 or idx == len(all_articles)-1:
            print(f"[INFO]  -> {idx+1}/{len(all_articles)} articles chunkés. Total chunks: {len(all_chunks)}")
    print(f"[INFO] Total chunks à indexer : {len(all_chunks)}")

    # 4. Indexe les chunks dans ChromaDB
    print(f"[INFO] Indexation dans ChromaDB (collection '{collection_name}')...")
    client = chromadb.Client()
    collection = client.get_or_create_collection(collection_name)
    collection.add_documents(
        documents=[chunk.page_content for chunk in all_chunks],
        metadatas=[chunk.metadata for chunk in all_chunks]
    )
    print(f"[INFO] {len(all_chunks)} chunks indexés dans Chroma (collection '{collection_name}').")
    return all_chunks
