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

# — If requested, clear the input box and rerun before rendering widgets
if st.session_state.get("clear_input", False):
    st.session_state["batch_input"] = ""
    st.session_state["clear_input"] = False
    st.rerun()

# — Multi‑line input area for batch questions
batch = st.text_area(
    "Ask Questions", 
    height=68
)

if st.button("Submit") and batch.strip():
    # split lines, filter out empty
    questions = [q.strip() for q in batch.splitlines() if q.strip()]
    # process each
    for q in questions:
        res = handle_query(q)
        # record in session history & logs
        st.session_state.logs.append(f"Q: {q} | {res['log']}")
        st.session_state.history.append({
            "q": q, 
            "branch": res["branch"], 
            "snippets": res["snippets"], 
            "answer": res["answer"]
        })
    # schedule the input box to be cleared on next run
    st.session_state["clear_input"] = True

#— Display all results
if st.session_state.history:
    st.markdown("---")
    st.header("Batch Results")
    for entry in st.session_state.history:
        st.subheader(f"Q: {entry['q']}")
        st.write("**Branch:**", entry["branch"].upper())
        if entry["branch"] == "rag" and entry["snippets"]:
            st.write("**Source Snippet:**")
            st.markdown(f"> {entry['snippets'][0]}")
        st.write("**Answer:**", entry["answer"])
        st.markdown("---")

# — Sidebar: full agent log
with st.sidebar:
    st.header("Agent Log")
    for line in st.session_state.logs:
        st.write(line)
