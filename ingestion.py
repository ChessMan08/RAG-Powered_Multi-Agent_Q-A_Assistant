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
