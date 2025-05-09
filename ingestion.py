import os
import glob
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
import faiss
import numpy as np

# Initialize OpenAI Embedding client
ios.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
client = OpenAI()

# Load raw text documents
dir_path = "docs/"
files = glob.glob(os.path.join(dir_path, "*.txt"))
texts = [open(f, encoding='utf-8').read() for f in files]

# Chunk texts
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = []
for txt in texts:
    chunks.extend(splitter.split_text(txt))

# Embed chunks
tokens = [client.embeddings.create(input=c, model="text-embedding-ada-002")["data"][0]["embedding"] for c in chunks]
embeddings = np.array(tokens).astype('float32')

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Persist index and chunks
faiss.write_index(index, "faiss_index.bin")
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)
print("Ingestion complete: index and chunks saved.")