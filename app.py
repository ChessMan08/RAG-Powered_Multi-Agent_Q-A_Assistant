import os
import streamlit as st
import sys
from streamlit.runtime.scriptrunner import RerunException
from streamlit.runtime.state.session_state_proxy import SessionState

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

# — Check if rerun was scheduled
if st.session_state.get("trigger_rerun", False):
    st.session_state["trigger_rerun"] = False
    st.rerun()

# — Input clearing logic
if st.session_state.get("clear_input"):
    st.session_state["batch_input"] = ""
    st.session_state["clear_input"] = False
    st.session_state["trigger_rerun"] = True
    st.stop()

# — Input area BELOW the results, in a form
with st.form("question_form"):
    batch = st.text_area(
        "Ask Questions", 
        height=68,
        key="batch_input"
    )
    submitted = st.form_submit_button("Submit")

if submitted and batch.strip():
    questions = [q.strip() for q in batch.splitlines() if q.strip()]
    for q in questions:
        res = handle_query(q)
        st.session_state.logs.append(f"Q: {q} | {res['log']}")
        st.session_state.history.append({
            "q": q,
            "branch": res["branch"],
            "snippets": res["snippets"],
            "answer": res["answer"]
        })
    # mark input for clearing
    st.session_state["clear_input"] = True
    st.stop()


# — Sidebar: full agent log
with st.sidebar:
    st.header("Agent Log")
    for line in st.session_state.logs:
        st.write(line)
