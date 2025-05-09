import os, logging

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["SENTENCE_TRANSFORMERS_VERBOSE"] = "0"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from ingestion import build_faiss_index

# Paths
INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.pkl"

# Ensure index exists
if not os.path.exists(INDEX_FILE) or not os.path.exists(CHUNKS_FILE):
    print("Index or chunks missing—building...")
    build_faiss_index()

# Load index & chunks
index = faiss.read_index(INDEX_FILE)
with open(CHUNKS_FILE, "rb") as f:
    chunks = pickle.load(f)

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve(query: str, k: int = 3) -> list[str]:
    """
    Return top-k text chunks similar to the query.
    """
    q_vec = embedder.encode([query], convert_to_numpy=True).astype('float32')
    _, indices = index.search(q_vec, k)
    return [chunks[i] for i in indices[0]]
