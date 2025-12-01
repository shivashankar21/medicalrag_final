# run_ingest.py
from src.preprocess import load_and_clean
from src.pipeline import ingest_contents
import os

if __name__ == "__main__":
    df = load_and_clean()
    contents = df['content'].tolist()
    metadatas = [{"question": q, "answer": a} for q,a in zip(df['question'], df['answer'])]
    print(f"Ingesting {len(contents)} docs...")
    count = ingest_contents(contents, metadatas=metadatas)
    print("Ingested:", count)
