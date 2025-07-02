import json
import os

INPUT_PATH = "data/scraped_articles_20250701_142521.json"
OUTPUT_TXT = "data/corpus_llm.txt"


def article_to_text(article):
    lines = []
    title = article.get("title", "Sans titre")
    url = article.get("url", "")
    lines.append("[DOCUMENT]")
    lines.append(f"Titre : {title}")
    lines.append(f"URL : {url}")
    lines.append("Contenu :")
    for block in article.get("content", []):
        if block["type"] in ["heading", "paragraph"]:
            lines.append(block["content"])
        elif block["type"] == "list":
            prefix = "- " if not block.get('ordered', False) else "1. "
            for item in block["items"]:
                lines.append(f"{prefix}{item}")
        elif block["type"] == "list_item":
            lines.append(f"- {block['content']}")
    lines.append("[/DOCUMENT]")
    return "\n".join(lines)


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        articles = json.load(f)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        for article in articles:
            text = article_to_text(article)
            f.write(text + "\n\n")
    print(f"Corpus exporté dans {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
