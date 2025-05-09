from retrieval import retrieve
from tools.calculator import calculate
from tools.dictionary import define
from transformers import pipeline

# Generative model
LLM_MODEL = "google/flan-t5-small"
# Initialize text-generation pipeline on CPU
generator = pipeline(
    "text2text-generation", model=LLM_MODEL, device=-1
)
log = []

def handle_query(query: str) -> dict:
    """
    Route to calculator, dictionary, or RAG. Returns branch, snippets, answer, log.
    """
    q_low = query.lower()
    # calculator
    if "calculate" in q_low:
        expr = q_low.replace("calculate", '').strip()
        result = calculate(expr)
        entry = f"Calculator branch: expr={expr} -> {result}"
        log.append(entry)
        return {"branch":"calculator","snippets":[],"answer":result,"log":entry}
    # dictionary
    if "define" in q_low:
        term = q_low.replace("define", '').strip()
        result = define(term)
        entry = f"Dictionary branch: term={term} -> {result}"
        log.append(entry)
        return {"branch":"dictionary","snippets":[],"answer":result,"log":entry}
    # RAG branch
    snippets = retrieve(query)
    # build prompt: context then direct instruction
    context = "\n".join([f"- {s}" for s in snippets])
    prompt = (
        f"You are given the following context snippets:\n{context}\n"
        f"Answer the question below concisely and only output the answer (no extra text).\n"
        f"Question: {query}\nAnswer:"
    )
    outputs = get_generator()(prompt, max_length=100, num_return_sequences=1)
    raw = outputs[0].get('generated_text', '')
    # extract answer after 'Answer:' if present
    answer = raw.split('Answer:')[-1].strip()
    entry = f"RAG branch: retrieved {len(snippets)} snippets"
    log.append(entry)
    return {"branch":"rag","snippets":snippets,"answer":answer,"log":entry}
