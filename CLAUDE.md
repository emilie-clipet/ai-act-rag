# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A free, local-first RAG (Retrieval-Augmented Generation) chatbot that answers questions about the
EU AI Act (Regulation (EU) 2024/1689) from official PDF documents, always citing its sources
(document name + page number). Built as a personal/portfolio project — code and UI text (including
comments and Streamlit strings) are in French.

## Commands

```bash
# Setup
python -m venv venv
venv\Scripts\activate          # Windows (this repo is developed on Windows/PowerShell)
pip install -r requirements.txt

# Build the vector index (run after adding/changing PDFs in data/)
python ingest.py

# Run the app
streamlit run app.py
```

There is no test suite, linter, or CI configured in this repo.

To index a document, place a PDF in `data/` (e.g. the official AI Act text from EUR-Lex) before
running `ingest.py`. Currently `data/` contains `OJ_L_202401689_FR_TXT.pdf` (the French official
text). `ingest.py` will fail fast (`SystemExit`) if `data/` is missing or has no PDFs.

## Architecture

Three-file pipeline, no framework beyond Streamlit for the UI:

1. **`ingest.py`** (offline, run manually): reads every PDF in `data/`, extracts text per page
   (`pypdf`), splits each page into overlapping character chunks (`CHUNK_SIZE=800`,
   `CHUNK_OVERLAP=150`), embeds chunks with a local `sentence-transformers` model
   (`all-MiniLM-L6-v2`), L2-normalizes the embeddings, and writes a FAISS `IndexFlatIP` (cosine
   similarity via inner product) plus a metadata pickle to `index/` (`faiss.index`,
   `metadata.pkl`). Metadata parallels chunks by list index: `metadata[i]` = `{"source", "page"}`
   for `chunks[i]`.

2. **`rag.py`** (`RagPipeline` class, imported by `app.py`): loads the FAISS index + metadata at
   construction (raises `FileNotFoundError` if `ingest.py` hasn't been run yet), and re-embeds
   queries with the *same* embedding model so vector spaces match.
   - `search(query, top_k)` → cosine-similarity nearest chunks with source/page/score.
   - `build_prompt(query, contexts)` → assembles a context block with per-source-per-page labels.
   - `generate_ollama(...)` / `generate_groq(...)` → two interchangeable free LLM backends, both
     driven by the same `SYSTEM_PROMPT` that instructs the model to answer only from the supplied
     context, cite sources as `(filename, p.N)`, admit when the answer isn't in the documents, and
     respond in French.
   - `answer(query, backend, top_k, **kwargs)` is the single entry point `app.py` calls: it runs
     `search` then dispatches to the right `generate_*` method based on `backend` (`"ollama"` or
     `"groq"`).

3. **`app.py`**: Streamlit UI. Sidebar picks the backend (Ollama local model name, or Groq model
   name + API key) and `top_k`. `RagPipeline` is built once via `@st.cache_resource`. Chat history
   lives in `st.session_state.messages`, each message optionally carrying its `sources` list so
   past answers keep their citations expandable in the UI.

**Key invariant:** the embedding model name (`EMBEDDING_MODEL = "all-MiniLM-L6-v2"`) is duplicated
as a constant in both `ingest.py` and `rag.py` — if you change one, change the other and rebuild
the index, or search/query vectors will be incompatible.

**LLM backends** are both "free tier" options and swappable per-request rather than at startup:
Ollama (local, no key, needs `ollama pull <model>` and the Ollama server running) and Groq (cloud,
needs a free API key entered in the sidebar, not stored anywhere — `.env`/`GROQ_API_KEY` is
supported via `python-dotenv` but not currently read in code, only documented in `.env.example`).
