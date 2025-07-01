# Importe la fonction d'ingestion complète depuis utils
from utils import full_scrape_and_index

# Point d'entrée du script
if __name__ == "__main__":
    print("[INFO] Démarrage du pipeline d'ingestion Prezevent...")  # Message de début
    chunks = full_scrape_and_index(collection_name="prezevent_articles")  # Lance l'ingestion et l'indexation
    print(f"[INFO] Pipeline terminé. Nombre total de chunks indexés : {len(chunks)}")  # Affiche le nombre de chunks indexés
