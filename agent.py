from retrieval import retrieve
from tools.calculator import calculate
from tools.dictionary import define

def handle_query(query: str) -> dict:
    """
    Route the query to calculator, dictionary, or RAG.
    Returns:
      - branch: which tool was used
      - snippets: list of retrieved chunks
      - answer: final answer text
      - log: a short log entry
    """
    q = query.lower().strip()

    # Calculator branch
    if "calculate" in q:
        expr = q.replace("calculate", "").strip()
        ans = calculate(expr)
        return {
            "branch": "calculator",
            "snippets": [],
            "answer": ans,
            "log": f"Calculated '{expr}' → {ans}"
        }

    # Dictionary branch
    if "define" in q:
        term = q.replace("define", "").strip()
        ans = define(term)
        return {
            "branch": "dictionary",
            "snippets": [],
            "answer": ans,
            "log": f"Defined '{term}' → {ans}"
        }

    # RAG branch
    snippets = retrieve(query)

    # Extract the exact A:… for this question from the top chunk
    top = snippets[0]
    ans = None
    for seg in top.split("Q:"):
        if not seg.strip():
            continue
        # match this segment's question
        if query.lower().strip("?") in seg.lower():
            if "A:" in seg:
                # take text after A: up to next Q:
                ans = seg.split("A:", 1)[1].split("Q:", 1)[0].strip()
            else:
                ans = seg.strip()
            break

    # fallback to first answer in chunk
    if ans is None:
        if "A:" in top:
            ans = top.split("A:", 1)[1].split("Q:", 1)[0].strip()
        else:
            ans = top.strip()

    return {
        "branch": "rag",
        "snippets": snippets,
        "answer": ans,
        "log": f"RAG retrieved {len(snippets)} chunks"
    }
