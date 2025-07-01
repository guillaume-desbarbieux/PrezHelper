from utils import full_scrape_and_index

if __name__ == "__main__":
    print("[INFO] Démarrage du pipeline d'ingestion Prezevent...")
    chunks = full_scrape_and_index(collection_name="prezevent_articles")
    print(f"[INFO] Pipeline terminé. Nombre total de chunks indexés : {len(chunks)}")
