# agent.py

from retrieval import retrieve
from tools.calculator import calculate
from tools.dictionary import define

def handle_query(query: str) -> dict:
    """
    Routes to calculator, dictionary, or RAG‑QA-per‑snippet.
    For RAG: runs a QA model on each snippet, picks the highest‑score answer.
    Returns { branch, snippets, answer, log }.
    """
    q = query.lower().strip()

    # Calculator
    if "calculate" in q:
        expr = q.replace("calculate", "").strip()
        ans = calculate(expr)
        return {
            "branch": "calculator",
            "snippets": [],
            "answer": ans,
            "log": f"Calculated '{expr}' → {ans}"
        }

    # Dictionary
    if "define" in q:
        term = q.replace("define", "").strip()
        ans = define(term)
        return {
            "branch": "dictionary",
            "snippets": [],
            "answer": ans,
            "log": f"Defined '{term}' → {ans}"
        }

    # RAG‑QA-per‑snippet
    snippets = retrieve(query, k=3)

    # Lazy‑load QA pipeline
    from transformers import pipeline
    qa = pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad",
        device=-1
    )

    best_ans = None
    best_score = -1.0

    # Run QA on each chunk, pick highest score
    for i, chunk in enumerate(snippets):
        try:
            out = qa(question=query, context=chunk)
            ans = out.get("answer", "").strip()
            score = float(out.get("score", 0))
            if score > best_score and ans:
                best_score = score
                best_ans = ans
        except Exception:
            continue

    # Fallback: if QA failed everywhere, fall back to manual scan Option 1
    if not best_ans:
        for chunk in snippets:
            for seg in chunk.split("Q:"):
                if query.lower().strip("?") in seg.lower():
                    if "A:" in seg:
                        best_ans = seg.split("A:",1)[1].split("Q:",1)[0].strip()
                    else:
                        best_ans = seg.strip()
                    break
            if best_ans:
                break

    # Last fallback: entire first chunk
    if not best_ans:
        first = snippets[0]
        if "A:" in first:
            best_ans = first.split("A:",1)[1].split("Q:",1)[0].strip()
        else:
            best_ans = first.strip()

    return {
        "branch": "rag",
        "snippets": snippets,
        "answer": best_ans,
        "log": f"RAG‑QA checked {len(snippets)} chunks, best_score={best_score:.2f}"
    }
