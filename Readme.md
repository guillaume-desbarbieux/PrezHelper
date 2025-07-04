# PrezHelper IA

PrezHelper IA est un assistant intelligent pour l'aide à l'utilisation de la plateforme Prezevent, basé sur l'IA générative et la recherche sémantique dans la documentation.

## Fonctionnalités principales

- **Reformulation automatique** : Reformule la question utilisateur dans le style de la documentation technique pour améliorer la recherche.
- **Recherche sémantique (RAG)** : Trouve les documents les plus pertinents de la documentation Prezevent à partir d'une question utilisateur, grâce à des embeddings de phrases et un scoring mixte (titre/contenu).
- **Génération de réponse LLM** : Génère une réponse synthétique à partir des documents trouvés, ou de toute la documentation, en utilisant OpenAI GPT-4o.
- **Interface Streamlit** : Application web simple, avec sidebar de configuration et affichage des sources utilisées.
- **Gestion des images et descriptions** : Extraction, description automatique et regroupement des interfaces visuelles issues de la documentation.


## Choix technologiques

- **Python 3.10+**
- **Streamlit** : Interface utilisateur rapide et interactive.
- **SentenceTransformers** : Embeddings sémantiques pour la recherche documentaire (modèle par défaut : `msmarco-MiniLM-L6-cos-v5`).
- **scikit-learn** : Calcul de similarité cosinus.
- **OpenAI API** : Génération de texte (GPT-4o).
- **Docker** : Conteneurisation pour reproductibilité et déploiement facile.

## Préparation et structuration du corpus

1. **Collecte** : Récupération de la documentation Prezevent
    - scraping html automatique depuis help.prezevent.com
    - export des titres, paragraphes, listes, images et iframes.
2. **Mise en forme** :
    - mise en forme en dictionnaire : un objet par article.
    - Suppression des balises iframes
3. **Description image**
    - Génération description technique des images via l'API OpenAI
    - Ajout de la description au sein de chaque article
4. **Formatage** :
    - Pour convenir au mieux aux attentes du LLM :
    - Chaque document est encapsulé dans `[DOCUMENT] ... [/DOCUMENT]` avec les champs `Titre :`, `URL :`, `Contenu :`.
    - Les URL des images sont supprimées. Leur description se trouve dans le flux du contenu dans une balise `[Interface affichée: ...]`
   - Stockage dans un fichier JSON `corpus_list.json`

## Fonctionnement détaillé

### 1. Reformulation de la question

- **But** : Adapter la question utilisateur au style de la documentation Prezevent pour maximiser la pertinence de la recherche documentaire.
- **Prompt utilisé** :
  > Tu es un assistant qui reformule des questions posées en langage naturel par des utilisateurs, pour les adapter au style de la documentation de l’application Prezevent. La documentation utilise un style clair, direct, sous forme de questions techniques. Les titres des articles commencent souvent par 'Comment' et se concentrent sur une action précise, comme 'Comment exporter une liste de contacts ?' ou 'Comment créer une campagne de mail ?'. Ta tâche est de reformuler les questions utilisateur dans ce style.
- **Modèle** : GPT-4o (OpenAI)
- **Résultat** : La question reformulée est utilisée pour la recherche documentaire.

### 2. Recherche documentaire (RAG)

- **But** : Trouver les documents les plus pertinents dans la base documentaire à partir de la question reformulée.
- **Pipeline** :
  - Embedding de la question, des titres et contenus des documents avec SentenceTransformers.
  - Calcul d'un score mixte :
    - `score_mixte = alpha * score_titre + (1-alpha) * score_contenu`
  - Sélection des documents selon :
    - `top_k` : nombre de documents à retourner
    - `min_score` : score minimal de pertinence (cosinus)
    - `relative_margin` : écart autorisé par rapport au meilleur score
    - `alpha` : poids du titre dans le score mixte
- **Intérêt de chaque paramètre** :
  - `top_k` : contrôle la quantité de contexte fourni au LLM
    Trop peu de contexte ne lui permettrait pas de trouver de réponse.
    Trop de contexte risquerait de diluer les informations pertinentes et engendre un surcoût inutile.
  - `min_score` : filtre les documents peu pertinents
  - `relative_margin` : permet d'élargir ou de restreindre la sélection autour du meilleur score
  - `alpha` : permet de donner plus d'importance au titre (souvent très informatif dans la doc Prezevent)

### 3. Génération de la réponse LLM

- **But** : Générer une réponse synthétique et sourcée à partir des documents sélectionnés (ou de toute la documentation).
- **Préparation du corpus pour le LLM**
    - Les documents pertinents sont concaténés et fournis en texte brut avec leurs balises spécifiques :
    `[DOCUMENT] Titre :... , URL :... , Contenu : ... [Interface affichée: ...] ... [/DOCUMENT]`

- **Prompt système** :
  > Tu es un assistant expert de Prezevent. Tu dois répondre uniquement en utilisant les documents fournis dans chaque requête. Ces documents peuvent inclure des descriptions détaillées d’interfaces visuelles (captures d’écran) sous forme de texte. Ces descriptions remplacent l'image et doivent être prises en compte pour comprendre les boutons, menus ou options affichés. Tu ne dois jamais mentionner d'images. Si aucune réponse claire n’est présente dans les documents, réponds simplement que tu ne sais pas. En fin de réponse, indique toujours les articles utilisés sous la forme : 📄 Source : [Titre de l’article](URL)
- **Prompt user** :
  > Voici la question d'un utilisateur :\n`{question}`\n\nVoici les documents pertinents à ta disposition :\nCertains passages décrivent l’interface visuelle (ex : boutons, onglets, textes affichés). Ces descriptions textuelles remplacent les captures d’écran et sont à interpréter comme si tu voyais l’écran.\n\n`{corpus}`\n\nATTENTION : ne fais pas d'invention. Ne réponds que si la réponse est clairement présente.

- **Modèle** : GPT-4o (OpenAI)
- **Affichage** : La réponse est affichée avec le coût estimé et le temps de génération.

## Pistes d'amélioration / TODO

- **Pipeline complète** Mise en place du processus complet sans intervention utilisateur en utilisant les paramètres les plus pertinents :
    Reformulation   ->  Recherche documentaire  ->  Génération de la réponse
- **Gestion des questions multiples** (Détection et découpage en plusieurs requêtes via la phase de reformulation)
- **Affinage du scoring RAG** (poids dynamiques, prise en compte du contexte, nouvelle recherche documentaire avec ajustement des paramètres avant génération si necessaire).
- **Ajout d'un mode conversationnel** (mémoire de session, suivi de contexte).
- **Automatisation de la mise à jour du corpus** (scraping régulier, UI to text).