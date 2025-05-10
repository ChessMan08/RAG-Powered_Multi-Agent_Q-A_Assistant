# agent.py

from retrieval import retrieve
from tools.calculator import calculate
from tools.dictionary import define
from sentence_transformers import SentenceTransformer
import numpy as np

# Embedding model (same as retrieval)
EMBED_MODEL = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL)

def handle_query(query: str) -> dict:
    """
    - “calculate” → calculator
    - “define” → dictionary
    - otherwise RAG: split chunks into Q/A pairs, embed each question segment,
      pick the answer whose question embedding is closest to the query embedding.
    """
    q = query.lower().strip()

    # Calculator
    if "calculate" in q:
        expr = q.replace("calculate", "").strip()
        ans = calculate(expr)
        return {"branch":"calculator","snippets":[],"answer":ans,"log":f"Calculated '{expr}'→{ans}"}

    # Dictionary
    if "define" in q:
        term = q.replace("define", "").strip()
        ans = define(term)
        return {"branch":"dictionary","snippets":[],"answer":ans,"log":f"Defined '{term}'→{ans}"}

    # RAG‑semantic‑match branch
    snippets = retrieve(query, k=3)

    # Embed the user query once
    q_vec = embedder.encode([query], convert_to_numpy=True).astype("float32")

    best_score = -1.0
    best_answer = None

    # For each chunk, split into QA pairs and compare embeddings
    for chunk in snippets:
        # find all Q:/A: occurrences
        parts = chunk.split("Q:")
        for seg in parts:
            if "A:" not in seg:
                continue
            # separate question vs answer
            ques_text, ans_text = seg.split("A:", 1)
            ques_text = ques_text.strip()
            ans_text = ans_text.strip().split("Q:",1)[0].strip()

            # embed this question segment
            seg_vec = embedder.encode([ques_text], convert_to_numpy=True).astype("float32")
            # cosine similarity
            sim = float(np.dot(q_vec, seg_vec.T) / (np.linalg.norm(q_vec)*np.linalg.norm(seg_vec)))
            if sim > best_score:
                best_score = sim
                best_answer = ans_text

    # fallback: first chunk’s whole answer
    if not best_answer:
        top = snippets[0]
        if "A:" in top:
            best_answer = top.split("A:",1)[1].split("Q:",1)[0].strip()
        else:
            best_answer = top.strip()

    return {
        "branch": "rag",
        "snippets": snippets,
        "answer": best_answer,
        "log": f"RAG-sematic-match picked answer with score {best_score:.3f}"
    }
