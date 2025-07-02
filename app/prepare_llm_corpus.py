import json
import os

INPUT_PATH = "data/scraped_articles_20250701_142521.json"
OUTPUT_TXT = "data/corpus_llm.txt"
OUTPUT_JSON = "data/corpus_llm.json"

def article_to_text(article):
    lines = []
    title = article.get("title", "Sans titre")
    url = article.get("url", "")
    lines.append(f"Titre : {title}")
    if url:
        lines.append(f"URL : {url}")
    for block in article.get("content", []):
        if block["type"] in ["heading", "paragraph"]:
            lines.append(block["content"])
        elif block["type"] == "list":
            prefix = "- " if not block.get('ordered', False) else "1. "
            for item in block["items"]:
                lines.append(f"{prefix}{item}")
        elif block["type"] == "list_item":
            lines.append(f"- {block['content']}")
        elif block["type"] == "image":
            img_url = block.get("src", "")
            alt = block.get("alt", "")
            if alt:
                lines.append(f"[Image: {alt}]")
            if img_url:
                lines.append(f"[Image URL: {img_url}]")
    return "\n".join(lines)

def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        articles = json.load(f)
    corpus = []
    for article in articles:
        text = article_to_text(article)
        corpus.append({"title": article.get("title", "Sans titre"), "url": article.get("url", ""), "text": text})
    # Export TXT (concaténé)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        for entry in corpus:
            f.write(entry["text"] + "\n\n" + ("="*40) + "\n\n")
    # Export JSON (un article par entrée)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    print(f"Corpus exporté dans {OUTPUT_TXT} et {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
