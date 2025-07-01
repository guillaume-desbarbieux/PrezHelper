# app/main.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.scraper import scrape_prezevent
from app.rag import ask_question

# Exécution de test : scrape un article + réponse à une question
if __name__ == "__main__":
    print("Scraping Prezevent...")
    scrape_prezevent()

    print("\nAssistant IA prêt.")
    while True:
        query = input("\nPose une question : ")
        answer = ask_question(query)
        print("\nRéponse :", answer)
