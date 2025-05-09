import streamlit as st
from agent import handle_query

# Set page configuration
st.set_page_config(
    page_title="RAG-Powered Multi-Agent Q&A",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("RAG-Powered Multi-Agent Q&A (Open-Source LLM)")

# Initialize session state for conversation history and agent logs
if 'history' not in st.session_state:
    st.session_state.history = []
if 'logs' not in st.session_state:
    st.session_state.logs = []

# User input
query = st.text_input("Ask a question:", key="input_query")

# Submit button triggers query handling
if st.button("Submit") and query:
    # Handle the user query through our multi-agent system
    res = handle_query(query)

    # Log the agent decision
    st.session_state.logs.append(res['log'])

    # Display agent log in sidebar
    with st.sidebar:
        st.header("Agent Log")
        for log_entry in st.session_state.logs:
            st.write(log_entry)

    # Display retrieved context for RAG branch
    if res['branch'] == 'rag':
        st.subheader("Retrieved Context Snippets")
        for snippet in res['snippets']:
            st.markdown(f"> {snippet}")

    # Display the final answer
    st.subheader("Answer")
    st.write(res['answer'])

    # Save conversation history
    st.session_state.history.append({"q": query, "a": res['answer']})

# Show conversation history
if st.session_state.history:
    st.markdown("---")
    st.header("Conversation History")
    for turn in st.session_state.history:
        st.markdown(f"**Q:** {turn['q']}")
        st.markdown(f"**A:** {turn['a']}")
