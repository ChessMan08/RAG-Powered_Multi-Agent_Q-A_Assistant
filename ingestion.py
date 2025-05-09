import os, logging

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["SENTENCE_TRANSFORMERS_VERBOSE"] = "0"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

import glob
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

import nltk
nltk.download('wordnet')
# Model for embeddings
EMBED_MODEL = "all-MiniLM-L6-v2"

def build_faiss_index(docs_path: str = "docs/",
                      index_file: str = "faiss_index.bin",
                      chunks_file: str = "chunks.pkl"):
    """
    Load .txt files, chunk them, embed with SentenceTransformer, build & save FAISS index.
    """
    # Load documents
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
    embedder = SentenceTransformer(EMBED_MODEL)
    embeddings = embedder.encode(chunks, convert_to_numpy=True)

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save index and chunks
    faiss.write_index(index, index_file)
    with open(chunks_file, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Built FAISS index ({index_file}) with {len(chunks)} chunks.")

if __name__ == "__main__":
    build_faiss_index()
