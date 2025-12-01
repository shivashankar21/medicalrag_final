# src/config.py
from dotenv import load_dotenv
import os

load_dotenv()  # loads .env


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

os.makedirs(CHROMA_DIR, exist_ok=True)


CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
GEN_MODEL_MAIN = os.getenv("GENERATOR_MODEL", "gemini-2.5-pro")
GEN_MODEL_GUARD = os.getenv("GUARDRAIL_MODEL", "gemini-2.5-flash")
GEN_MODEL_EVAL = os.getenv("EVALUATOR_MODEL", GEN_MODEL_GUARD)
MAX_RETRIEVAL = int(os.getenv("MAX_RETRIEVAL", "5"))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not set. Populate it via environment variable or .env file.")
PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID", "tranquil-symbol-470922-v7")
LOCATION = os.getenv("GOOGLE_LOCATION", "us-central1")
