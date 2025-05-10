# agent.py
from retrieval import retrieve
from tools.calculator import calculate
from tools.dictionary import define

def handle_query(query: str) -> dict:
    """
    Route to calculator, dictionary, or RAG. 
    RAG: return the full answer verbatim from the top chunk.
    """
    q = query.lower()

    # calculator
    if "calculate" in q:
        expr = q.replace("calculate", "").strip()
        ans = calculate(expr)
        return {"branch":"calculator","snippets":[],"answer":ans,"log":f"Calculated {expr}→{ans}"}

    # dictionary
    if "define" in q:
        term = q.replace("define", "").strip()
        ans = define(term)
        return {"branch":"dictionary","snippets":[],"answer":ans,"log":f"Defined {term}→{ans}"}

    # RAG branch: pull chunks
    snippets = retrieve(query)

    # take the first chunk that contains an "A:" and return everything after "A:"
    top = snippets[0]
    if "A:" in top:
        # split on the first "A:" and strip
        ans = top.split("A:",1)[1].strip()
    else:
        # fallback: return the entire chunk
        ans = top.strip()

    return {
      "branch": "rag",
      "snippets": snippets,
      "answer": ans,
      "log": f"RAG retrieved {len(snippets)} chunks"
    }
