from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

from config import get_collections, validate_metadata, CEFR_LEVELS

BATCH = 128
MAX_CHARS = 1500  # ~350-400 tokens: focused chunks retrieve better than large ones
OVERLAP = 150     # carried across splits so a rule isn't severed at the seam


def _id(text: str, *prefix: str) -> str:
    """Deterministic content-hash ID so re-running ingestion updates rather than duplicates."""
    h = hashlib.sha1(("|".join(prefix) + "|" + text).encode("utf-8")).hexdigest()
    return f"{'_'.join(prefix)}_{h[:16]}" if prefix else h[:16]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:60]


def _infer_level(text: str, default: str = "A1") -> str:
    """Pulls a CEFR tag like 'level: B1' or '[B2]' out of a heading if present."""
    m = re.search(r"\b(A1|A2|B1|B2|C1|C2)\b", text)
    return m.group(1) if m else default


def chunk_markdown_grammar(md: str) -> list[tuple[str, str]]:
    """Splits markdown into (heading, body) chunks on # / ## / ### headers.
    Each chunk keeps its heading so a rule stays attached to its examples.
    Oversized sections are windowed with overlap and the heading repeated on each piece."""
    lines = md.splitlines()
    chunks: list[tuple[str, str]] = []
    cur_head = "general"
    buf: list[str] = []

    def _split_oversized(head: str, body: str) -> list[tuple[str, str]]:
        full = f"{head}\n\n{body}"
        if len(full) <= MAX_CHARS:
            return [(head, full)]
        pieces, start = [], 0
        while start < len(body):
            window = body[start: start + MAX_CHARS]
            pieces.append((head, f"{head}\n\n{window.strip()}"))
            start += MAX_CHARS - OVERLAP
        return pieces

    def flush():
        body = "\n".join(buf).strip()
        if body:
            chunks.extend(_split_oversized(cur_head, body))

    for line in lines:
        if re.match(r"^#{1,3}\s+", line):
            flush()
            cur_head = re.sub(r"^#{1,3}\s+", "", line).strip()
            buf = []
        else:
            buf.append(line)
    flush()
    return chunks


def load_pdf_text(path: Path) -> str:
    """Extracts all text from a PDF as a single string. Prefers Markdown for clean headers."""
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def iter_grammar_docs(root: Path) -> Iterable[Path]:
    for ext in ("*.md", "*.markdown", "*.pdf"):
        yield from root.rglob(ext)


def ingest_grammar(root: str, language: str, default_level: str = "A1") -> int:
    """Ingests all .md/.pdf files under root into the grammar collection.
    Returns the number of chunks upserted."""
    grammar, _ = get_collections()
    ids, docs, metas = [], [], []
    root_path = Path(root)

    for path in iter_grammar_docs(root_path):
        raw = load_pdf_text(path) if path.suffix == ".pdf" else path.read_text("utf-8")
        for heading, chunk in chunk_markdown_grammar(raw):
            meta = validate_metadata(
                {
                    "language": language,
                    "cefr_level": _infer_level(heading, default_level),
                    "topic": _slug(heading),
                    "type": "grammar",
                    "source": path.name,
                },
                kind="grammar",
            )
            ids.append(_id(chunk, language, "gr"))
            docs.append(chunk)
            metas.append(meta)

    _flush(grammar, ids, docs, metas)
    print(f"[grammar] upserted {len(ids)} chunks from {root}")
    return len(ids)


def ingest_vocab(json_path: str, language: str) -> int:
    """Ingests a vocabulary JSON file into the vocab collection. Each entry becomes one document.
    Expected entry shape: {term, definition, examples, cefr_level, pos, gender, topic}.
    Returns the number of entries upserted."""
    _, vocab = get_collections()
    entries = json.loads(Path(json_path).read_text("utf-8"))
    ids, docs, metas = [], [], []

    for e in entries:
        examples = " ".join(e.get("examples", []))
        doc = f"{e['term']} — {e.get('definition', '')}\n{examples}".strip()
        meta = validate_metadata(
            {
                "language": language,
                "cefr_level": e.get("cefr_level", "A1"),
                "topic": e.get("topic", "vocab"),
                "type": "vocab",
                "source": Path(json_path).name,
                "lemma": e.get("term", ""),
                "pos": e.get("pos", ""),
                "gender": e.get("gender", ""),
            },
            kind="vocab",
        )
        ids.append(_id(doc, language, "vo"))
        docs.append(doc)
        metas.append(meta)

    _flush(vocab, ids, docs, metas)
    print(f"[vocab] upserted {len(ids)} entries from {json_path}")
    return len(ids)


def _flush(collection, ids, docs, metas) -> None:
    for i in range(0, len(ids), BATCH):
        collection.upsert(
            ids=ids[i: i + BATCH],
            documents=docs[i: i + BATCH],
            metadatas=metas[i: i + BATCH],
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Ingest curriculum into ChromaDB")
    p.add_argument("--grammar", help="dir of .md/.pdf grammar docs")
    p.add_argument("--vocab", help="path to vocab .json")
    p.add_argument("--lang", required=True, help="target language code, e.g. fr")
    p.add_argument("--level", default="A1", choices=CEFR_LEVELS,
                   help="fallback CEFR level for grammar docs without a tag")
    args = p.parse_args()

    if args.grammar:
        ingest_grammar(args.grammar, args.lang, args.level)
    if args.vocab:
        ingest_vocab(args.vocab, args.lang)
    if not (args.grammar or args.vocab):
        p.error("pass --grammar and/or --vocab")
