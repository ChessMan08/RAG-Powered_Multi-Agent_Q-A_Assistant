from retrieval import retrieve
from tools.calculator import calculate
from tools.dictionary import define

def handle_query(query: str) -> dict:
    """
    Route the query to calculator, dictionary, or RAG (manual scan) branch.
    Returns a dict with:
      - branch: which tool was used
      - snippets: list of retrieved chunks
      - answer: the extracted answer
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

    # 3) RAG branch (Option 1: manual scan across snippets)
    snippets = retrieve(query, k=3)
    ans = None

    # Scan each snippet for the matching Q:/A: pair
    for chunk in snippets:
        # Each chunk may contain multiple "Q: ... A: ..." pairs
        parts = chunk.split("Q:")
        for seg in parts:
            if not seg.strip():
                continue
            # If this segment's question matches the user query
            if query.lower().strip("?") in seg.lower():
                # Extract everything after "A:" up to the next "Q:"
                if "A:" in seg:
                    ans = seg.split("A:", 1)[1].split("Q:", 1)[0].strip()
                else:
                    ans = seg.strip()
                break
        if ans:
            break

    # Fallback: if no matching segment found, return first chunk's answer
    if not ans:
        top = snippets[0]
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
