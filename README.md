# ⚖️ Assistant AI Act — Chatbot RAG (100% gratuit)

Chatbot capable de répondre à des questions sur le règlement européen sur l'intelligence
artificielle (AI Act) à partir de documents officiels, en citant systématiquement ses sources
(document + numéro de page).

**Stack :** Python, Streamlit, RAG (Retrieval-Augmented Generation), embeddings locaux
(sentence-transformers), FAISS, LLM gratuit (Ollama local ou Groq cloud).

---

## 1. Installation

```bash
# Cloner / se placer dans le dossier du projet
cd ai-act-rag

# Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate        # sous Windows : venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

## 2. Récupérer le document officiel de l'AI Act (gratuit)

Télécharge le texte officiel du règlement (PDF) depuis une source officielle, par exemple :
- EUR-Lex : https://eur-lex.europa.eu/eli/reg/2024/1689/oj (Règlement (UE) 2024/1689)

Place le(s) fichier(s) PDF dans le dossier `data/` :

```
ai-act-rag/
└── data/
    └── AI_Act.pdf
```

Tu peux mettre plusieurs PDF (par exemple le règlement + des guides explicatifs officiels de
la Commission européenne) : ils seront tous indexés avec leur nom de fichier comme source.

## 3. Construire l'index vectoriel

```bash
python ingest.py
```

Ce script :
1. Extrait le texte de chaque page des PDF
2. Découpe le texte en chunks (avec chevauchement pour ne pas perdre le contexte)
3. Calcule des embeddings avec un modèle local et gratuit (`all-MiniLM-L6-v2`)
4. Sauvegarde un index FAISS + les métadonnées (fichier source, page) dans `index/`

## 4. Choisir le modèle de génération (gratuit) — deux options

### Option A — Ollama (recommandé : 100% local, aucune clé, aucune limite)

1. Installer Ollama : https://ollama.com
2. Télécharger un modèle léger :
   ```bash
   ollama pull llama3.2
   ```
3. Laisser Ollama tourner en arrière-plan (il démarre automatiquement un serveur local).

### Option B — Groq (cloud, très rapide, free tier généreux)

1. Créer un compte gratuit sur https://console.groq.com
2. Générer une clé API gratuite : https://console.groq.com/keys
3. Tu la renseigneras directement dans l'interface Streamlit (barre latérale).

Aucune carte bancaire n'est nécessaire pour l'une ou l'autre option.

## 5. Lancer l'application

```bash
streamlit run app.py
```

L'interface s'ouvre dans le navigateur. Choisis ton backend (Ollama ou Groq) dans la barre
latérale, pose ta question, et la réponse s'affiche avec un encart "Sources utilisées"
indiquant les extraits exacts (document + page) qui ont servi à générer la réponse.

---

## Comment fonctionne le RAG (pour ton entretien / ta soutenance)

1. **Indexation** (`ingest.py`) : les PDF sont découpés en chunks, chaque chunk est transformé
   en vecteur numérique (embedding) qui capture son sens, puis stocké dans un index FAISS.
2. **Recherche** (`rag.py` → `search`) : la question de l'utilisateur est elle aussi transformée
   en vecteur, puis on recherche par similarité cosinus les chunks les plus proches sémantiquement.
3. **Génération augmentée** (`rag.py` → `generate_*`) : les chunks récupérés sont injectés dans
   le prompt envoyé au LLM, qui doit répondre en se basant uniquement sur ce contexte et en
   citant ses sources — ce qui limite les hallucinations et rend la réponse traçable/explicable.

Cette architecture illustre concrètement : IA générative, recherche documentaire vectorielle,
et explicabilité/conformité (traçabilité des sources), ce qui correspond directement aux
compétences visées par ce projet.

## Idées d'amélioration (à mentionner en entretien si tu as le temps de les coder)

- Ajouter un score de confiance ou un seuil de similarité en dessous duquel le bot répond
  "je ne sais pas" plutôt que d'halluciner.
- Découper par article/paragraphe plutôt que par taille fixe de caractères (chunking sémantique).
- Ajouter un ré-ordonnancement (reranking) des résultats avec un cross-encoder.
- Déployer l'app gratuitement sur Streamlit Community Cloud pour avoir une démo en ligne à
  mettre en lien direct sur le CV/LinkedIn.

## Exemple de ligne pour le CV

> Développement d'un assistant RAG (Python, Streamlit, FAISS, sentence-transformers, LLM
> open-source) permettant d'interroger la documentation officielle de l'AI Act avec génération
> de réponses sourcées et traçables — projet personnel, 100% gratuit et open-source.
