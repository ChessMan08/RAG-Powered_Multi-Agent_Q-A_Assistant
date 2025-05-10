from retrieval import retrieve
from tools.calculator import calculate
from tools.dictionary import define
from transformers import pipeline

LLM_MODEL = "t5-small"

def handle_query(query: str) -> dict:
    q = query.lower()
    if "calculate" in q:
        expr = q.replace("calculate", "").strip()
        ans = calculate(expr)
        return {"branch": "calculator", "snippets": [], "answer": ans, "log": f"Calculated {expr} → {ans}"}

    if "define" in q:
        term = q.replace("define", "").strip()
        ans = define(term)
        return {"branch": "dictionary", "snippets": [], "answer": ans, "log": f"Defined {term} → {ans}"}

    snippets = retrieve(query)

    from transformers import pipeline
    qa = pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad",
        device=-1
    )
    context = "\n".join(snippets)
    out = qa(question=query, context=context)
    ans = out["answer"].strip()
    return {"branch": "rag", "snippets": snippets, "answer": ans, "log": f"RAG retrieved {len(snippets)} chunks"}
