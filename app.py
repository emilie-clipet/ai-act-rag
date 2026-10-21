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

GROQ_MODEL = "llama-3.1-8b-instant"

EXAMPLE_QUESTIONS = [
    "Quelles sont les obligations pour les systèmes d'IA à haut risque ?",
    "Qu'est-ce qu'un système d'IA selon l'AI Act ?",
    "Quelles sont les pratiques d'IA interdites ?",
    "Quelles sanctions en cas de non-conformité ?",
]

# ---------- Pop-up de bienvenue (une seule fois par session) ----------
@st.dialog("Bienvenue sur l'Assistant AI Act ⚖️")
def welcome_dialog():
    st.write(
        "Ce chatbot répond à vos questions sur le règlement européen sur l'intelligence "
        "artificielle (AI Act), en se basant uniquement sur le texte officiel indexé. "
        "Il cite systématiquement ses sources (document + page) pour chaque réponse."
    )
    st.caption("Projet personnel réalisé par Emilie Clipet")
    if st.button("Commencer"):
        st.rerun()

if "welcome_shown" not in st.session_state:
    st.session_state.welcome_shown = False

if not st.session_state.welcome_shown:
    st.session_state.welcome_shown = True
    welcome_dialog()

# ---------- Clé API Groq (depuis les secrets Streamlit) ----------
try:
    api_key = st.secrets.get("GROQ_API_KEY")
except Exception:
    api_key = None

if not api_key:
    st.error("Clé API Groq introuvable. Ajoute `GROQ_API_KEY` dans les secrets de l'application.")
    st.stop()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Configuration")
    top_k = st.slider(
        "Nombre d'extraits du règlement consultés",
        min_value=2,
        max_value=10,
        value=5,
        help=(
            "Le chatbot analyse ce nombre d'extraits avant de formuler sa réponse. "
            "Plus ce nombre est élevé, plus il dispose de contexte. Toutefois, un trop "
            "grand nombre d'extraits peut mélanger des informations et réduire la "
            "précision de la réponse."
        ),
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

# ---------- Questions d'exemple (avant la première question posée) ----------
if not st.session_state.messages:
    st.caption("💡 Exemples de questions :")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, example in zip(cols, EXAMPLE_QUESTIONS):
        with col:
            if st.button(example, use_container_width=True):
                st.session_state.pending_question = example

# ---------- Zone de saisie ----------
question = st.chat_input("Pose ta question sur l'AI Act...")
if not question:
    question = st.session_state.pop("pending_question", None)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche dans les documents et génération de la réponse..."):
            try:
                answer_text, sources = pipeline.answer(
                    question,
                    backend="groq",
                    top_k=top_k,
                    model_name=GROQ_MODEL,
                    api_key=api_key,
                )
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
                st.rerun()
            except Exception as e:
                st.error(f"Erreur : {e}")
