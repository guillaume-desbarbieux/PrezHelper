# Importe la fonction d'ingestion complète depuis utils
from utils import full_scrape_and_index
import os

# Point d'entrée du script
if __name__ == "__main__":
    # Force le dossier de persistance Chroma sur la machine hôte
    os.environ["CHROMA_DB_DIR"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chroma"))
    print(f"[INFO] Dossier de persistance ChromaDB : {os.environ['CHROMA_DB_DIR']}")
    print("[INFO] Démarrage du pipeline d'ingestion Prezevent...")  # Message de début
    chunks = full_scrape_and_index(collection_name="prezevent_articles")  # Lance l'ingestion et l'indexation
    print(f"[INFO] Pipeline terminé. Nombre total de chunks indexés : {len(chunks)}")  # Affiche le nombre de chunks indexés
