import os
import streamlit as st

# — Page config
st.set_page_config(
    page_title="RAG‑Powered Multi‑Agent Q&A",
    layout="centered",
    initial_sidebar_state="expanded"
)
st.title("RAG‑Powered Multi‑Agent Q&A")

# — Ensure FAISS index exists
from ingestion import build_faiss_index, INDEX_FILE, CHUNKS_FILE
if not (os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE)):
        build_faiss_index()

# — Import the agent
from agent import handle_query

# — Session state for logs & history
if "logs" not in st.session_state:
    st.session_state.logs = []
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts {q, branch, snippets, answer}

# — Display all past results first
if st.session_state.history:
    st.header("Batch Results")
    for entry in st.session_state.history:
        st.subheader(f"Q: {entry['q']}")
        st.write("**Branch:**", entry["branch"].upper())
        if entry["branch"] == "rag":
            st.write("**Retrieved Context:**")
            for i, snip in enumerate(entry["snippets"], 1):
                st.markdown(f"> Snippet {i}: {snip}")
        st.write("**Answer:**", entry["answer"])
        st.markdown("---")

# — Then show the input area at the bottom
batch = st.text_area(
    "Ask Questions", 
    height=50
)
if st.button("Submit") and batch.strip():
    questions = [q.strip() for q in batch.splitlines() if q.strip()]
    for q in questions:
        res = handle_query(q)
        st.session_state.logs.append(f"Q: {q} | {res['log']}")
        st.session_state.history.append({
            "q": q,
            "branch": res["branch"],
            "snippets": res["snippets"],
            "answer": res["answer"],
        })
    # after submission, clear the text area
    st.experimental_rerun()

# — Sidebar: full agent log
with st.sidebar:
    st.header("Agent Log")
    for line in st.session_state.logs:
        st.write(line)
