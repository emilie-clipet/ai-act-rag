"""
Script d'ingestion : lit les PDF du dossier data/, les découpe en chunks,
calcule les embeddings (modèle local gratuit) et sauvegarde un index FAISS
+ les métadonnées (source, page) sur disque.

Usage :
    python ingest.py
"""

import os
import pickle
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
INDEX_DIR = "index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # modèle local gratuit, léger et efficace
CHUNK_SIZE = 800       # nombre de caractères par chunk
CHUNK_OVERLAP = 150    # chevauchement entre chunks pour ne pas couper le contexte


def extract_pages(pdf_path: str):
    """Retourne une liste de (texte, numero_page) pour un PDF donné."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append((text, i + 1))
    return pages


def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Découpe un texte en chunks avec chevauchement."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def build_index():
    if not os.path.isdir(DATA_DIR):
        raise SystemExit(f"Le dossier '{DATA_DIR}/' n'existe pas.")

    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise SystemExit(
            f"Aucun PDF trouvé dans '{DATA_DIR}/'. "
            "Place-y le texte officiel de l'AI Act (ex: depuis EUR-Lex) puis relance ce script."
        )

    print(f"PDF détectés : {pdf_files}")
    print("Chargement du modèle d'embeddings (local, gratuit)...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    all_chunks = []      # texte des chunks
    all_metadata = []    # {"source": ..., "page": ...}

    for pdf_file in pdf_files:
        path = os.path.join(DATA_DIR, pdf_file)
        print(f"Lecture de {pdf_file}...")
        pages = extract_pages(path)
        for text, page_num in pages:
            for chunk in chunk_text(text):
                if len(chunk.strip()) < 30:
                    continue
                all_chunks.append(chunk)
                all_metadata.append({"source": pdf_file, "page": page_num})

    print(f"Total de {len(all_chunks)} chunks à indexer.")
    print("Calcul des embeddings...")
    embeddings = model.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    # normalisation pour utiliser la similarité cosinus avec un index "Inner Product"
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))
    with open(os.path.join(INDEX_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump({"chunks": all_chunks, "metadata": all_metadata}, f)

    print(f"Index sauvegardé dans '{INDEX_DIR}/'. Terminé !")


if __name__ == "__main__":
    build_index()
