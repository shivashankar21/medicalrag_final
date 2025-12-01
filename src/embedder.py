import time
import google.generativeai as genai
from src.config import GOOGLE_API_KEY, EMBED_MODEL

genai.configure(api_key=GOOGLE_API_KEY)

def embed_texts(texts: list, batch_size=64, sleep_secs=0.1):
    """
    Embeds a list of texts using Google's text-embedding-004 model.
    Compatible with free tier and stable batching.
    """
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        response = genai.embed_content(
            model=EMBED_MODEL,
            content=batch,
            task_type="retrieval_document"
        )

        batch_embeddings = response["embedding"]
        embeddings.extend(batch_embeddings)

        time.sleep(sleep_secs)

    return embeddings
