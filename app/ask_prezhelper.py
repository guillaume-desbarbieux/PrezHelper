from rag_logic import ask_question

if __name__ == "__main__":
    print("--- Interface de question RAG Prezevent ---")
    while True:
        question = input("\nVotre question (ou 'exit'): ")
        if question.lower() in ("exit", "quit", "q"):
            break
        results = ask_question(question)
        print("\n[Réponses les plus pertinentes] :\n")
        for i, doc in enumerate(results['documents'][0]):
            meta = results['metadatas'][0][i]
            print(f"--- Réponse {i+1} ---")
            print(doc)
            print(f"[Source: {meta.get('source', 'N/A')}]\n")
