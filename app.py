from retrieval import retrieve
from tools.calculator import calculate
from tools.dictionary import define

log = []

def handle_query(query: str) -> dict:
    # Decide branch
    if "calculate" in query.lower():
        branch = "calculator"
        # extract expression
        expr = query.lower().replace("calculate", "").strip()
        result = calculate(expr)
        log_entry = f"Branch=Calculator, expr={expr}, result={result}"
        log.append(log_entry)
        return {"branch": branch, "snippets": [], "answer": result, "log": log_entry}
    if "define" in query.lower():
        branch = "dictionary"
        term = query.lower().replace("define", "").strip()
        result = define(term)
        log_entry = f"Branch=Dictionary, term={term}, definition={result}"
        log.append(log_entry)
        return {"branch": branch, "snippets": [], "answer": result, "log": log_entry}
    # RAG branch
    branch = "RAG"
    snippets = retrieve(query)
    prompt = "Context:\n" + "\n---\n".join(snippets) + f"\n\nQuestion: {query}\nAnswer:"
    # call OpenAI ChatCompletion
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}]
    )
    ans = resp.choices[0].message.content
    log_entry = f"Branch=RAG, retrieved={len(snippets)} snippets"
    log.append(log_entry)
    return {"branch": branch, "snippets": snippets, "answer": ans, "log": log_entry}