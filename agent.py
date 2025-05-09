# agent.py
from retrieval import retrieve
from tools.calculator import calculate
from tools.dictionary import define
from transformers import pipeline

# Generative model
LLM_MODEL = "google/flan-t5-small"  # smaller model to avoid TorchScript issues

# Initialize text-generation pipeline
# We set `device=0` (GPU) if available; remove device arg if CPU-only
generator = pipeline(
    "text2text-generation", 
    model=LLM_MODEL,
    device=-1  # use CPU; change to 0 if GPU available
)

# In-memory log of agent decisions
log = []

def handle_query(query: str) -> dict:
    """
    Route the query to calculator, dictionary, or RAG pipeline based on keywords.
    Returns a dict with keys: branch, snippets, answer, log.
    """
    q_low = query.lower()

    # Calculator branch
    if "calculate" in q_low:
        expr = q_low.replace("calculate", '').strip()
        result = calculate(expr)
        entry = f"Calculator branch: expr={expr} -> {result}"
        log.append(entry)
        return {"branch": "calculator", "snippets": [], "answer": result, "log": entry}

    # Dictionary branch
    if "define" in q_low:
        term = q_low.replace("define", '').strip()
        result = define(term)
        entry = f"Dictionary branch: term={term} -> {result}"
        log.append(entry)
        return {"branch": "dictionary", "snippets": [], "answer": result, "log": entry}

    # RAG branch
    snippets = retrieve(query)
    prompt = (
        "Use the following context to answer the question.
" +
        "
---
".join(snippets) +
        f"
Question: {query}
Answer:"
    )
    # Generate answer
    outputs = generator(prompt, max_length=200, num_return_sequences=1)
    ans = outputs[0].get('generated_text', '').strip()
    entry = f"RAG branch: retrieved {len(snippets)} snippets"
    log.append(entry)
    return {"branch": "rag", "snippets": snippets, "answer": ans, "log": entry}
```## app.py

```python
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
