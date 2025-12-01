# src/pipeline.py
from src.embedder import embed_texts
from src.vectorstore import add_documents, query_embeddings
from src.agents import guardrail_agent, generator_agent, evaluator_agent
from src.config import MAX_RETRIEVAL
from src.config import EMBED_MODEL, GOOGLE_API_KEY

import uuid

def ingest_contents(contents, metadatas=None):
    ids = [f"mq_{i}" for i in range(len(contents))]
    embeddings = embed_texts(contents, batch_size=128, sleep_secs=0.15)
    add_documents(docs=contents, metadatas=metadatas, ids=ids, embeddings=embeddings)
    return len(contents)

def rag_answer(query, top_k=MAX_RETRIEVAL, max_regens=2):
    q_emb = embed_texts([query], batch_size=1)[0]
    res = query_embeddings([q_emb], n_results=top_k)
    docs = []
    for i, d in enumerate(res['documents'][0]):
        docs.append({"id": res['ids'][0][i], "document": d, "distance": res['distances'][0][i]})
    guard = guardrail_agent(query, docs)
    if guard.get("decision") == "UNSAFE":
        return {"answer": "UNSAFE: seek immediate medical attention", "score":0.0, "source_ids":[]}
    if guard.get("decision") == "INSUFFICIENT":
        return {"answer": "INSUFFICIENT: Not enough relevant data found.", "score":0.0, "source_ids":[]}
    selected = guard.get("selected", [])
    # generator + evaluator loop
    for attempt in range(max_regens+1):
        gen = generator_agent(query, selected)
        evalr = evaluator_agent(query, selected, gen)
        score = evalr.get("score", 0.0)
        if score >= 0.8:
            return {"answer": gen, "score": score, "source_ids":[s['id'] for s in selected]}
        # otherwise tighten selected
        if len(selected) > 1:
            selected = selected[:max(1, len(selected)//2)]
        else:
            break
    return {"answer": gen, "score": score, "source_ids":[s['id'] for s in selected]}
