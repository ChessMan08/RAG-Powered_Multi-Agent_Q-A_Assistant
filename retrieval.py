import os
import pickle
import faiss
import numpy as np
from openai import OpenAI

# Load index and chunks
index = faiss.read_index("faiss_index.bin")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

client = OpenAI()

def retrieve(query, k=3):
    # Embed query
ios.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    q_emb = client.embeddings.create(input=query, model="text-embedding-ada-002")["data"][0]["embedding"]
    q_vec = np.array([q_emb]).astype('float32')
    # Search
    D, I = index.search(q_vec, k)
    return [chunks[i] for i in I[0]]