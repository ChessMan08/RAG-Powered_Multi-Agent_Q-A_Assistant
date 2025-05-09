import os
import pickle
import faiss
import numpy as np
from openai import OpenAI

# Load index and chunks
index = faiss.read_index("faiss_index.bin")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Initialize OpenAI client (ensuring API key is read from environment)
os.environ.setdefault('OPENAI_API_KEY', os.getenv('OPENAI_API_KEY', ''))
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
