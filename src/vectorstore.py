# src/vectorstore.py

from typing import Iterable, Optional

from chromadb import PersistentClient  # pyright: ignore[reportMissingImports]

from src.config import CHROMA_DIR

_client: Optional[PersistentClient] = None


def get_client():
    """Return a singleton Chroma PersistentClient backed by CHROMA_DIR."""
    global _client
    if _client is None:
        _client = PersistentClient(path=CHROMA_DIR)
    return _client


def create_collection(name="medquad"):
    """
    Creates or retrieves a persistent ChromaDB collection.
    """
    client = get_client()

    # Get OR Create collection safely
    try:
        col = client.get_collection(name)
    except Exception:
        col = client.create_collection(name)

    return col


def _batched(seq: Iterable, size: int):
    chunk = []
    for item in seq:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def add_documents(
    docs,
    metadatas=None,
    ids=None,
    embeddings=None,
    collection_name="medquad",
    batch_size=2048,
):
    """Adds documents + metadata + embeddings into the vector store in batches."""
    col = create_collection(collection_name)

    mt = metadatas or [None] * len(docs)
    ds = ids or [None] * len(docs)
    embs = embeddings or [None] * len(docs)

    for doc_batch, meta_batch, id_batch, emb_batch in zip(
        _batched(docs, batch_size),
        _batched(mt, batch_size),
        _batched(ds, batch_size),
        _batched(embs, batch_size),
    ):
        col.add(
            documents=doc_batch,
            metadatas=meta_batch,
            ids=id_batch,
            embeddings=emb_batch,
        )

    return True


def query_embeddings(
    embeddings,
    n_results=5,
    collection_name="medquad"
):
    """
    Searches the vector DB using embedding vectors.
    """
    col = create_collection(collection_name)

    return col.query(
        query_embeddings=embeddings,
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
