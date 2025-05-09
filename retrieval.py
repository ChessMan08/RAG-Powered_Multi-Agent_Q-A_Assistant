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
