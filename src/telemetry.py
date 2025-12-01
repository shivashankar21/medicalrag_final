# src/telemetry.py
import csv, os
from datetime import datetime
FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "telemetry.csv")

def log_interaction(query, answer, score, retrieval_count):
    write_header = not os.path.exists(FILE)
    with open(FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp","query","answer","score","retrieval_count"])
        writer.writerow([datetime.utcnow().isoformat(), query, answer, score, retrieval_count])
