import streamlit as st

# Page config
st.set_page_config(page_title="RAG Q&A", layout="centered")

st.title("RAG‑Powered Multi‑Agent Q&A")

# One‑time build flag
if 'built' not in st.session_state:
    st.session_state.built = False

# ensure index is built automatically on startup
from ingestion import build_faiss_index, INDEX_FILE, CHUNKS_FILE
import os
if not (os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE)):
    build_faiss_index()

# At this point built==True, so UI always shows
from agent import handle_query

query = st.text_input("Ask a question:")
if st.button("Submit") and query:
    res = handle_query(query)
    st.sidebar.header("Agent Log")
    st.sidebar.write(f"Q: {query} | {res['log']}")
    st.subheader("Answer")
    st.write(res["answer"])
