"""
Chatbot RAG sur l'AI Act — interface Streamlit.

Lancement :
    streamlit run app.py
"""

import streamlit as st
from rag import RagPipeline

st.set_page_config(page_title="Assistant AI Act", page_icon="⚖️", layout="wide")

st.title("⚖️ Assistant AI Act — Chatbot RAG")
st.caption(
    "Posez une question sur le règlement européen sur l'intelligence artificielle. "
    "Les réponses sont générées à partir des documents officiels indexés, avec citation des sources."
)

# ---------- Sidebar : configuration du backend LLM ----------
with st.sidebar:
    st.header("⚙️ Configuration")
    backend = st.selectbox(
        "Modèle de génération (gratuit)",
        options=["ollama", "groq"],
        format_func=lambda x: "Ollama (local, 100% gratuit)" if x == "ollama" else "Groq (cloud, free tier)",
    )

    if backend == "ollama":
        model_name = st.text_input("Nom du modèle Ollama", value="llama3.2")
        st.info(
            "Nécessite Ollama installé et lancé en local (ollama.com), "
            "et le modèle téléchargé via : `ollama pull llama3.2`"
        )
        api_key = None
    else:
        model_name = st.text_input("Nom du modèle Groq", value="llama-3.1-8b-instant")
        try:
            secret_api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            secret_api_key = None

        if secret_api_key:
            api_key = secret_api_key
            st.success("Clé API Groq chargée automatiquement depuis les secrets.")
        else:
            api_key = st.text_input("Clé API Groq (gratuite)", type="password")
            st.info("Clé gratuite disponible sur https://console.groq.com/keys")

    top_k = st.slider("Nombre de passages récupérés (top_k)", min_value=2, max_value=10, value=5)

    st.divider()
    st.caption(
        "Avant la première utilisation : place un PDF officiel de l'AI Act dans le dossier "
        "`data/` puis lance `python ingest.py`."
    )

# ---------- Chargement du pipeline RAG (mis en cache) ----------
@st.cache_resource
def load_pipeline():
    return RagPipeline()

try:
    pipeline = load_pipeline()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# ---------- Historique de conversation ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources utilisées"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['source']} — page {s['page']}** (score : {s['score']:.2f})")
                    st.caption(s["text"][:300] + "...")

# ---------- Zone de saisie ----------
question = st.chat_input("Pose ta question sur l'AI Act...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche dans les documents et génération de la réponse..."):
            try:
                if backend == "groq" and not api_key:
                    st.warning("Merci de renseigner ta clé API Groq dans la barre latérale.")
                    st.stop()

                kwargs = {"model_name": model_name}
                if backend == "groq":
                    kwargs["api_key"] = api_key

                answer_text, sources = pipeline.answer(question, backend=backend, top_k=top_k, **kwargs)
                st.markdown(answer_text)

                if sources:
                    with st.expander("📚 Sources utilisées"):
                        for s in sources:
                            st.markdown(f"**{s['source']} — page {s['page']}** (score : {s['score']:.2f})")
                            st.caption(s["text"][:300] + "...")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "sources": sources,
                })
            except Exception as e:
                st.error(f"Erreur : {e}")
