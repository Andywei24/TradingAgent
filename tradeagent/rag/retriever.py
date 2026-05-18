from __future__ import annotations

from tradeagent.rag.embedder import embed_one
from tradeagent.rag.vectorstore import search


def retrieve(query: str, k: int = 5) -> list[dict]:
    vec = embed_one(query)
    if vec.size == 0:
        return []
    return search(vec, k=k)
