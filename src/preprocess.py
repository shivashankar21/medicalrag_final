# src/preprocess.py
import pandas as pd
import os

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "medquad.csv")

def load_and_clean(csv_path=DATA_PATH):
    df = pd.read_csv(csv_path)
    # find question/answer columns heuristically
    qcol = None
    acol = None
    for c in df.columns:
        lc = c.lower()
        if 'question' in lc:
            qcol = c
        if 'answer' in lc or 'response' in lc:
            acol = c
    if not qcol or not acol:
        # fallback: take first two columns as question/answer
        qcol, acol = df.columns[0], df.columns[1]
    df = df[[qcol, acol] + [c for c in df.columns if c not in (qcol, acol)]]
    df = df.rename(columns={qcol: 'question', acol: 'answer'})
    df = df.dropna(subset=['question','answer'])
    df['content'] = df['question'].astype(str).str.strip() + "\n\n" + df['answer'].astype(str).str.strip()
    df = df.drop_duplicates(subset=['content']).reset_index(drop=True)
    return df
