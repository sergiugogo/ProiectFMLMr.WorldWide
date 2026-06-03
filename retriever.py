from __future__ import annotations

import os
import re
from dataclasses import dataclass

from config import CEFR_LEVELS, get_collections

_VOCAB_CUES = (
    r"\bwhat does\b", r"\bwhat is the (?:word|french)\b", r"\bhow do (?:you|i) say\b",
    r"\bmeaning of\b", r"\btranslate\b", r"\bvocab(?:ulary)?\b", r"\bword for\b",
    r"\bgive me (?:some )?words\b", r"\bquiz me\b", r"\btest me\b", r"\bflashcards?\b",
    r"\bque (?:veut dire|signifie)\b", r"\bcomment (?:dit-on|dire)\b", r"\bvocabulaire\b",
)
_VOCAB_RE = re.compile("|".join(_VOCAB_CUES), re.IGNORECASE)


def decide_mode(student_input: str, *, forced: str | None = None) -> str:
    """Keyword heuristic router. Returns "vocab" or "error".
    Pass forced="vocab" or forced="error" to bypass the heuristic."""
    if forced in ("vocab", "error"):
        return forced
    return "vocab" if _VOCAB_RE.search(student_input or "") else "error"


ROUTER_MODEL = os.getenv("ROUTER_MODEL", "gpt-4o-mini")

_ROUTER_SYSTEM = (
    "You route a French learner's message to one of two retrieval modes. "
    "Answer with exactly one word, lowercase, no punctuation:\n"
    "  vocab  — the student asks for word meanings, translations, vocabulary, "
    "or a vocabulary quiz/test.\n"
    "  error  — the student wrote a French sentence (possibly with mistakes) to "
    "be checked, or asks about grammar.\n"
    "If unsure, answer error."
)


def llm_route(student_input: str, *, forced: str | None = None,
              model: str = ROUTER_MODEL) -> str:
    """LLM-backed router. Drop-in replacement for decide_mode (same signature).
    Falls back to the keyword heuristic on any failure so it is always safe to call."""
    if forced in ("vocab", "error"):
        return forced
    try:
        from openai import OpenAI

        resp = OpenAI().chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=1,
            messages=[
                {"role": "system", "content": _ROUTER_SYSTEM},
                {"role": "user", "content": student_input or ""},
            ],
        )
        answer = (resp.choices[0].message.content or "").strip().lower()
        if answer.startswith("vocab"):
            return "vocab"
        if answer.startswith("error"):
            return "error"
    except Exception:
        pass
    return decide_mode(student_input)


def _levels_up_to(level: str) -> list[str]:
    """All CEFR levels at or below level, so a B1 student can still see A1/A2 content."""
    if level not in CEFR_LEVELS:
        return list(CEFR_LEVELS)
    return list(CEFR_LEVELS[: CEFR_LEVELS.index(level) + 1])


@dataclass
class Retrieved:
    mode: str           # "error" | "vocab"
    documents: list[str]
    metadatas: list[dict]
    distances: list[float]

    def is_empty(self) -> bool:
        return not self.documents


def retrieve(
    student_input: str,
    *,
    mode: str,
    cefr_level: str = "A1",
    n_results: int = 4,
    scope_by_level: bool = True,
) -> Retrieved:
    """Embeds student_input and pulls the most relevant chunks from the grammar or
    vocabulary collection, optionally scoped to the student's CEFR level."""
    grammar, vocab = get_collections()
    collection = vocab if mode == "vocab" else grammar

    where = None
    if scope_by_level:
        where = {"cefr_level": {"$in": _levels_up_to(cefr_level)}}

    res = collection.query(
        query_texts=[student_input],
        n_results=n_results,
        where=where,
    )
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    return Retrieved(mode=mode, documents=docs, metadatas=metas, distances=dists)


def format_context(retrieved: Retrieved) -> str:
    """Renders retrieved chunks into the [CONTEXT]...[/CONTEXT] block the model expects.
    Returns an empty string when retrieval is empty."""
    if retrieved.is_empty():
        return ""
    body = "\n\n".join(doc.strip() for doc in retrieved.documents if doc.strip())
    return f"[CONTEXT]\n{body}\n[/CONTEXT]"
