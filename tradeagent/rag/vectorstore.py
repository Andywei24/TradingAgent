from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tradeagent.config import get_settings
from tradeagent.rag.embedder import embed_texts
from tradeagent.rag.loader import Chunk

INDEX_FILENAME = "index.faiss"
META_FILENAME = "meta.jsonl"


def _index_path() -> Path:
    return Path(get_settings().rag_index_dir) / INDEX_FILENAME


def _meta_path() -> Path:
    return Path(get_settings().rag_index_dir) / META_FILENAME


def build_index(chunks: list[Chunk]) -> int:
    import faiss

    if not chunks:
        return 0
    vecs = embed_texts([c.text for c in chunks])
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    _index_path().parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(_index_path()))
    with _meta_path().open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(
                json.dumps(
                    {"text": c.text, "source": c.source, "title": c.title, "section": c.section}
                )
                + "\n"
            )
    return len(chunks)


def load_index():
    import faiss

    p = _index_path()
    if not p.exists():
        return None, []
    index = faiss.read_index(str(p))
    meta = []
    with _meta_path().open(encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    return index, meta


def search(query_vec: np.ndarray, k: int = 5) -> list[dict]:
    index, meta = load_index()
    if index is None:
        return []
    if query_vec.ndim == 1:
        query_vec = query_vec[None, :]
    scores, ids = index.search(query_vec.astype("float32"), k)
    hits: list[dict] = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0 or idx >= len(meta):
            continue
        m = meta[idx]
        hits.append(
            {
                "score": float(score),
                "text": m["text"],
                "source": m["source"],
                "title": m["title"],
                "section": m.get("section"),
            }
        )
    return hits
