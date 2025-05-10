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
    Route the query to calculator, dictionary, or RAG‑semantic‑match branch.
    Returns a dict with:
      - branch: which tool was used
      - snippets: list of retrieved chunks
      - answer: the extracted answer or a fallback message
      - log: a short description of the action
    """
    q = query.lower().strip()

    # 1) Calculator branch
    if "calculate" in q:
        expr = q.replace("calculate", "").strip()
        ans = calculate(expr)
        return {
            "branch": "calculator",
            "snippets": [],
            "answer": ans,
            "log": f"Calculated '{expr}' → {ans}"
        }

    # 2) Dictionary branch
    if "define" in q:
        term = q.replace("define", "").strip()
        ans = define(term)
        return {
            "branch": "dictionary",
            "snippets": [],
            "answer": ans,
            "log": f"Defined '{term}' → {ans}"
        }

    # 3) RAG‑semantic‑match branch
    snippets = retrieve(query, k=3)

    # Embed the user query once
    q_vec = embedder.encode([query], convert_to_numpy=True).astype("float32")

    best_score = -1.0
    best_answer = None
    best_snippet = None

    # For each chunk, split into QA pairs and compare embeddings
    for chunk in snippets:
        parts = chunk.split("Q:")
        for seg in parts:
            if "A:" not in seg:
                continue
            ques_text, ans_text = seg.split("A:", 1)
            ques_text = ques_text.strip()
            ans_text = ans_text.strip().split("Q:", 1)[0].strip()

            # embed this question segment
            seg_vec = embedder.encode([ques_text], convert_to_numpy=True).astype("float32")
            # cosine similarity
            sim = float(np.dot(q_vec, seg_vec.T) / (np.linalg.norm(q_vec) * np.linalg.norm(seg_vec)))
            if sim > best_score and ans_text:
                best_score = sim
                best_answer = ans_text
                best_snippet = chunk

    # Fallback: if no good match or score below threshold
    if not best_answer or best_score < 0.2:
        best_answer = (
            "I’m sorry, I don’t have enough information to answer that. "
            "Please try asking in a different way or about another topic."
        )
        log_entry = "RAG‑semantic‑match found no suitable answer"
    else:
        log_entry = f"RAG‑semantic‑match picked answer with score {best_score:.3f}"

    return {
        "branch": "rag",
        "snippets": [best_snippet] if best_snippet else [],
        "answer": best_answer,
        "log": log_entry
    }
