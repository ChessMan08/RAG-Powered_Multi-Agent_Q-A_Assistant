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

    # RAG
    snippets = retrieve(query)
    prompt = (
        "Answer the question using only the context below.\n\n"
        + "\n---\n".join(snippets)
        + f"\n\nQuestion: {query}\nAnswer:"
    )
    prompt = f"""\
Answer the question below concisely using only the information in the context. \
Do NOT repeat the question or list the context in your answer.

Context:
{"\n".join(snippets)}

Question: {query}
Answer:"""

    gen = pipeline("text2text-generation", model=LLM_MODEL, device=-1, framework="pt")
    out = gen(prompt, max_length=200, num_return_sequences=1)[0]["generated_text"].strip()
    return {"branch": "rag", "snippets": snippets, "answer": out, "log": f"RAG retrieved {len(snippets)} chunks"}
