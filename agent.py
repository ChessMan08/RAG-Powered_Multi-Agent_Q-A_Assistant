import os, logging

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["SENTENCE_TRANSFORMERS_VERBOSE"] = "0"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

def handle_query(query: str) -> dict:
    from transformers import pipeline
    from retrieval import retrieve
    from tools.calculator import calculate
    from tools.dictionary import define

    generator = pipeline(
        "text2text-generation",
        model="t5-small",
        device=-1,
        framework="pt"
    )

    log = []
    q_low = query.lower()

    if "calculate" in q_low:
        expr = q_low.replace("calculate", "").strip()
        result = calculate(expr)
        log.append(f"Calculator: {expr} → {result}")
        return {"branch": "calculator", "snippets": [], "answer": result, "log": log[-1]}

    if "define" in q_low:
        term = q_low.replace("define", "").strip()
        result = define(term)
        log.append(f"Dictionary: {term} → {result}")
        return {"branch": "dictionary", "snippets": [], "answer": result, "log": log[-1]}

    snippets = retrieve(query)
    prompt = (
        "Answer the question below as clearly as possible using only the provided context.\n\n"
        + "\n---\n".join(snippets)
        + f"\n\nQuestion: {query}\nAnswer:"
    )
    output = generator(prompt, max_length=200, num_return_sequences=1)
    answer = output[0]["generated_text"].strip()
    log.append(f"RAG: retrieved {len(snippets)} chunks")
    return {"branch": "rag", "snippets": snippets, "answer": answer, "log": log[-1]}
