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
    with st.spinner("Building RAG index… this happens only once"):
        build_faiss_index()
    st.success("Index built — you can now ask questions!")

# — Import the agent
from agent import handle_query

# — Session state
if "logs" not in st.session_state:
    st.session_state.logs = []
if "history" not in st.session_state:
    st.session_state.history = []

# — Input
query = st.text_input("Ask a question:")
if st.button("Submit") and query:
    res = handle_query(query)
    # record
    st.session_state.logs.append(f"Q: {query} | {res['log']}")
    st.session_state.history.append({"q": query, **res})

# — Show latest result
if st.session_state.history:
    last = st.session_state.history[-1]
    st.subheader("Tool / Agent Branch")
    st.write(last["branch"].upper())

    if last["branch"] == "rag":
        st.subheader("Retrieved Context Snippets")
        for i, snip in enumerate(last["snippets"], start=1):
            st.markdown(f"**Snippet {i}:**")
            st.write(snip)

    st.subheader("Answer")
    st.write(last["answer"])

# — Sidebar: full agent log
with st.sidebar:
    st.header("Agent Log")
    for entry in st.session_state.logs:
        st.write(entry)
