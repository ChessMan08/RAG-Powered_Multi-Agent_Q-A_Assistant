# agent.py

from retrieval import retrieve
from tools.calculator import calculate
from tools.dictionary import define

def handle_query(query: str) -> dict:
    """
    Route the query to calculator, dictionary, or RAG‑QA pipeline.
    Returns a dict with:
      - branch: which tool ran
      - snippets: the retrieved text chunks
      - answer: the final answer string
      - log: a short description of what happened
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

    # 3) RAG‑QA branch
    #    Retrieve top‑3 relevant chunks, then run QA to extract the precise answer.
    snippets = retrieve(query, k=3)

    # Combine all snippets into one context string
    context = "\n\n".join(snippets)

    # Lazy‑import the QA pipeline
    from transformers import pipeline
    qa = pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad",
        device=-1  # CPU; change to 0 for GPU
    )

    # Run the QA model
    result = qa(question=query, context=context)
    ans = result.get("answer", "").strip()

    return {
        "branch": "rag",
        "snippets": snippets,
        "answer": ans,
        "log": f"RAG‑QA extracted answer (score={result.get('score', 0):.2f})"
    }
