import streamlit as st
from agent import handle_query

st.set_page_config(page_title="RAG Q&A", layout="centered")
st.title("RAG-Powered Multi-Agent Q&A")

if 'logs' not in st.session_state:
    st.session_state.logs = []

query = st.text_input("Ask a question:")
if st.button("Submit") and query:
    res = handle_query(query)
    # record log
    st.session_state.logs.append(res['log'])
    # show only answer
    st.subheader("Answer")
    st.write(res['answer'])
    # show context in expander if RAG
    if res['branch']=='rag':
        with st.expander("Show Retrieved Context Snippets"):
            for s in res['snippets']:
                st.markdown(f"> {s}")
    # show agent log
    with st.sidebar:
        st.header("Agent Log")
        for l in st.session_state.logs:
            st.write(l)
