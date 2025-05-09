import streamlit as st

# Page config
st.set_page_config(page_title="RAG Q&A", layout="centered")

st.title("RAG‑Powered Multi‑Agent Q&A")

# One‑time build flag
if 'built' not in st.session_state:
    st.session_state.built = False

# If not built, show build button and then stop
if not st.session_state.built:
    if st.button("Build RAG Index (one‑time)"):
        from ingestion import build_faiss_index
        build_faiss_index()
        st.session_state.built = True
        st.success("Index built! You can now ask questions below.")
        # After building, do NOT st.stop() so that on this same run the UI appears
    else:
        # If button not clicked yet, do stop here
        st.stop()

# At this point built==True, so UI always shows
from agent import handle_query

query = st.text_input("Ask a question:")
if st.button("Submit") and query:
    res = handle_query(query)
    st.sidebar.header("Agent Log")
    st.sidebar.write(res["log"])
    st.subheader("Answer")
    st.write(res["answer"])
