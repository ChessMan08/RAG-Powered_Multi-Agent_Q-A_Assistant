from retrieval import retrieve
from tools.calculator import calculate
from tools.dictionary import define
from transformers import pipeline

# Generative model
LLM_MODEL = "google/flan-t5-small"  # smaller model to avoid TorchScript issues

# Initialize text-generation pipeline
generator = pipeline(
    "text2text-generation", 
    model=LLM_MODEL,
    device=-1
)

log = []

def handle_query(query: str) -> dict:
    q_low = query.lower()

    if "calculate" in q_low:
        expr = q_low.replace("calculate", '').strip()
        result = calculate(expr)
        entry = f"Calculator branch: expr={expr} -> {result}"
        log.append(entry)
        return {"branch": "calculator", "snippets": [], "answer": result, "log": entry}

    if "define" in q_low:
        term = q_low.replace("define", '').strip()
        result = define(term)
        entry = f"Dictionary branch: term={term} -> {result}"
        log.append(entry)
        return {"branch": "dictionary", "snippets": [], "answer": result, "log": entry}

    snippets = retrieve(query)
    context = "\n---\n".join(snippets)
    prompt = f"Use the following context to answer the question.\n{context}\nQuestion: {query}\nAnswer:"
    outputs = generator(prompt, max_length=200, num_return_sequences=1)
    ans = outputs[0].get('generated_text', '').strip()
    entry = f"RAG branch: retrieved {len(snippets)} snippets"
    log.append(entry)
    return {"branch": "rag", "snippets": snippets, "answer": ans, "log": entry}
