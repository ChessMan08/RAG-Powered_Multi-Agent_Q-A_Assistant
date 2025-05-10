# RAG-Powered Multi-Agent Q&A Assistant

This Streamlit app is a multi-agent Q&A assistant that uses Retrieval-Augmented Generation (RAG) to answer user questions from a small document collection. It reads text documents from `docs/`, indexes them with **FAISS** using SentenceTransformers embeddings. 

It nswers queries by 
- (a) retrieving relevant text chunks.
- (b) using a generative LLM to form the answer.

If a query contains the word `“calculate”`, the app routes it to a `simple calculator agent`; if it contains `“define”`, it routes to a `WordNet dictionary agent`. Otherwise it performs RAG: it embeds the query (with all-MiniLM-L6-v2) and finds the top relevant chunks from the FAISS index, then combines those snippets with the query to prompt the **Flan-T5-small** LLM for an answer. The interface shows the question, which agent branch was taken, the answer, and (for RAG queries) a button to reveal the retrieved context snippets.
