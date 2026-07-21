"""
Module RAG : charge l'index FAISS, recherche les passages pertinents
pour une question, construit un prompt avec citations, puis interroge
un LLM gratuit (Ollama local ou Groq cloud) pour générer la réponse.
"""

import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

INDEX_DIR = "index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

SYSTEM_PROMPT = """Tu es un assistant expert de l'AI Act (règlement européen sur l'intelligence artificielle).
Réponds UNIQUEMENT à partir des extraits de documents fournis dans le contexte ci-dessous.
Si l'information ne se trouve pas dans le contexte, dis clairement que tu ne peux pas répondre avec les documents disponibles.
Cite systématiquement tes sources en indiquant le nom du document et le numéro de page entre parenthèses, par exemple : (AI_Act.pdf, p.12).
Réponds en français, de façon claire et précise.
"""


class RagPipeline:
    def __init__(self):
        index_path = os.path.join(INDEX_DIR, "faiss.index")
        meta_path = os.path.join(INDEX_DIR, "metadata.pkl")
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(
                "Index introuvable. Lance d'abord 'python ingest.py' après avoir "
                "placé un PDF de l'AI Act dans le dossier data/."
            )
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.metadata = data["metadata"]
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

    def search(self, query: str, top_k: int = 5):
        """Retourne les top_k chunks les plus pertinents avec leurs métadonnées."""
        q_emb = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)
        scores, indices = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "text": self.chunks[idx],
                "source": self.metadata[idx]["source"],
                "page": self.metadata[idx]["page"],
                "score": float(score),
            })
        return results

    def build_prompt(self, query: str, contexts: list):
        context_block = "\n\n".join(
            f"[Extrait {i+1} — {c['source']}, page {c['page']}]\n{c['text']}"
            for i, c in enumerate(contexts)
        )
        user_prompt = f"""Contexte (extraits de l'AI Act) :
{context_block}

Question : {query}

Réponds à la question en te basant uniquement sur le contexte ci-dessus, en citant tes sources (document + page)."""
        return user_prompt

    # ---------- Génération avec Ollama (100% local, gratuit, aucune clé) ----------
    def generate_ollama(self, query: str, contexts: list, model_name: str = "llama3.2"):
        import ollama
        user_prompt = self.build_prompt(query, contexts)
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response["message"]["content"]

    # ---------- Génération avec Groq (cloud, free tier généreux, clé gratuite) ----------
    def generate_groq(self, query: str, contexts: list, api_key: str, model_name: str = "llama-3.1-8b-instant"):
        from groq import Groq
        client = Groq(api_key=api_key)
        user_prompt = self.build_prompt(query, contexts)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    def answer(self, query: str, backend: str, top_k: int = 5, **kwargs):
        contexts = self.search(query, top_k=top_k)
        if not contexts:
            return "Aucun passage pertinent trouvé dans les documents indexés.", []

        if backend == "ollama":
            text = self.generate_ollama(query, contexts, model_name=kwargs.get("model_name", "llama3.2"))
        elif backend == "groq":
            text = self.generate_groq(
                query, contexts,
                api_key=kwargs["api_key"],
                model_name=kwargs.get("model_name", "llama-3.1-8b-instant"),
            )
        else:
            raise ValueError(f"Backend inconnu : {backend}")

        return text, contexts
