from retrieval import retrieve
from tools.calculator import calculate
from tools.dictionary import define
from transformers import pipeline

# Generative model
LLM_MODEL = "google/flan-t5-base"
# Initialize text-generation pipeline
generator = pipeline("text2text-generation", model=LLM_MODEL)

log = []

def handle_query(query: str) -> dict:
    # Route based on keywords
    q_low = query.lower()
    if "calculate" in q_low:
        expr = q_low.replace("calculate", '').strip()
        result = calculate(expr)
        entry = f"Calculator: expr={expr} -> {result}"
        log.append(entry)
        return {"branch":"calculator","snippets":[],"answer":result,"log":entry}
    if "define" in q_low:
        term = q_low.replace("define", '').strip()
        result = define(term)
        entry = f"Dictionary: term={term} -> {result}"
        log.append(entry)
        return {"branch":"dictionary","snippets":[],"answer":result,"log":entry}

    # RAG branch
    snippets = retrieve(query)
    prompt = "Use the context to answer the question.\n" + "\n---\n".join(snippets) + f"\nQuestion: {query}"
    # Generate answer
    gen = generator(prompt, max_length=200)
    ans = gen[0]['generated_text']
    entry = f"RAG: retrieved {len(snippets)} snippets"
    log.append(entry)
    return {"branch":"rag","snippets":snippets,"answer":ans,"log":entry}
