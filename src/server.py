# src/server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.pipeline import rag_answer
from src.vectorstore import create_collection
from src.telemetry import log_interaction
from src.preprocess import load_and_clean
import uvicorn

app = FastAPI(title="MedHelp RAG API")

class Query(BaseModel):
    query: str

@app.on_event("startup")
def startup():
    create_collection("medquad")

@app.post("/ask")
def ask(q: Query):
    if not q.query.strip():
        raise HTTPException(400, "Query empty")
    res = rag_answer(q.query)
    log_interaction(q.query, res.get("answer",""), res.get("score",0.0), len(res.get("source_ids",[])))
    return res

if __name__ == "__main__":
    uvicorn.run("src.server:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
