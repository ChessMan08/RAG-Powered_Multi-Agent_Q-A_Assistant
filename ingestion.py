import os
import glob
import pickle
import numpy as np
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI

def build_faiss_index(docs_path: str = "docs/",
                      index_file: str = "faiss_index.bin",
                      chunks_file: str = "chunks.pkl"):
    """
    Load text files, chunk them, embed with OpenAI, and build & save a FAISS index.
    """
    # Ensure API key is set
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("OPENAI_API_KEY not set in environment.")
    client = OpenAI()

    # Load raw text documents
    files = glob.glob(os.path.join(docs_path, "*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {docs_path}")
    texts = [open(f, encoding='utf-8').read() for f in files]

    # Chunk texts
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for txt in texts:
        chunks.extend(splitter.split_text(txt))

    # Embed chunks
    embeddings = []
    for chunk in chunks:
        emb = client.embeddings.create(input=chunk, model="text-embedding-ada-002")["data"][0]["embedding"]
        embeddings.append(emb)
    embeddings = np.array(embeddings).astype('float32')

    # Build FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Persist index and chunks
    faiss.write_index(index, index_file)
    with open(chunks_file, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Index built: {index_file}, chunks saved: {chunks_file}")


if __name__ == "__main__":
    build_faiss_index()

import os
import pickle
import numpy as np
import faiss
from openai import OpenAI
from ingestion import build_faiss_index

# File paths
INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.pkl"

# Ensure index & chunks exist, otherwise build
if not os.path.exists(INDEX_FILE) or not os.path.exists(CHUNKS_FILE):
    print("FAISS index or chunks not found—building now...")
    build_faiss_index()

# Load index and chunks
index = faiss.read_index(INDEX_FILE)
with open(CHUNKS_FILE, "rb") as f:
    chunks = pickle.load(f)

# Initialize OpenAI client
en_api_key = os.getenv('OPENAI_API_KEY')
if not en_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is missing.")
os.environ.setdefault('OPENAI_API_KEY', en_api_key)
client = OpenAI()


def retrieve(query: str, k: int = 3) -> list[str]:
    """
    Embed the user query, perform a FAISS search, and return the top-k text chunks.
    """
    # Embed the query text
    q_emb = client.embeddings.create(
        input=query,
        model="text-embedding-ada-002"
    )["data"][0]["embedding"]

    # Convert to numpy array for FAISS
    q_vec = np.array([q_emb]).astype('float32')

    # Search the FAISS index
    distances, indices = index.search(q_vec, k)

    # Return the corresponding chunks
    return [chunks[i] for i in indices[0]]
```

**app.py** (no change needed except to ensure ingestion import works)

```python
import streamlit as st
from agent import handle_query

st.title("RAG-Powered Multi-Agent Q&A")

if 'history' not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask a question:")
if st.button("Submit") and query:
    res = handle_query(query)
    st.sidebar.write("**Agent Log**")
    st.sidebar.write(res['log'])
    if res['branch'] == 'RAG':
        st.write("**Retrieved Context Snippets:**")
        for snip in res['snippets']:
            st.markdown(f"> {snip}")
    st.write("**Answer:**")
    st.write(res['answer'])
    st.session_state.history.append((query, res['answer']))

if st.session_state.history:
    st.write("---")
    for q, a in st.session_state.history:
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
```
