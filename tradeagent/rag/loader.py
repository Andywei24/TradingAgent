from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CHUNK_TOKENS = 800
CHUNK_OVERLAP = 100
APPROX_CHARS_PER_TOKEN = 4


@dataclass
class Chunk:
    text: str
    source: str
    title: str
    section: str | None = None


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join((p.extract_text() or "") for p in reader.pages)


def _read_html(path: Path) -> str:
    import trafilatura

    raw = path.read_text(encoding="utf-8", errors="ignore")
    extracted = trafilatura.extract(raw) or raw
    return extracted


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _chunk(text: str, source: str, title: str) -> list[Chunk]:
    chunk_chars = CHUNK_TOKENS * APPROX_CHARS_PER_TOKEN
    overlap_chars = CHUNK_OVERLAP * APPROX_CHARS_PER_TOKEN
    chunks: list[Chunk] = []
    i = 0
    while i < len(text):
        piece = text[i : i + chunk_chars].strip()
        if piece:
            chunks.append(Chunk(text=piece, source=source, title=title))
        i += chunk_chars - overlap_chars
    return chunks


def load_file(path: Path) -> list[Chunk]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = _read_pdf(path)
    elif suffix in {".html", ".htm"}:
        text = _read_html(path)
    elif suffix in {".md", ".txt"}:
        text = _read_text(path)
    else:
        return []
    return _chunk(text, source=str(path), title=path.stem)


def load_directory(root: Path) -> list[Chunk]:
    root = Path(root)
    all_chunks: list[Chunk] = []
    for path in root.rglob("*"):
        if path.is_file():
            all_chunks.extend(load_file(path))
    return all_chunks
