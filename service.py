from __future__ import annotations

import os

import retriever
import state_store
import tutor

# Set USE_LLM_ROUTER=1 to classify with retriever.llm_route instead of the keyword heuristic.
_ROUTER = retriever.llm_route if os.getenv("USE_LLM_ROUTER") == "1" else retriever.decide_mode

HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "6"))

state_store.init_db()


def history_from_db(user_id: str) -> list[dict]:
    """Returns the last HISTORY_TURNS turns as a ChatML history list, oldest first."""
    rows = list(reversed(state_store.recent_interactions(user_id, limit=HISTORY_TURNS)))
    history: list[dict] = []
    for r in rows:
        history.append({"role": "user", "content": r["student_input"]})
        history.append({"role": "assistant", "content": r["response"]})
    return history


def resolve_level(user_id: str, level: str | None) -> str:
    """Creates the user if they don't exist. If level is given it is persisted;
    otherwise the stored level is returned."""
    user = state_store.get_or_create_user(user_id, default_level=level or "A1")
    if level and level != user["cefr_level"]:
        state_store.set_level(user_id, level)
        return level
    return user["cefr_level"]


def _setup(user_id: str, message: str, cefr: str,
           forced_mode: str | None) -> tuple[str, list[str], list[dict]]:
    """Route → retrieve → build messages. Returns (mode, sources, messages)."""
    mode = _ROUTER(message, forced=forced_mode)
    retrieved = retriever.retrieve(message, mode=mode, cefr_level=cefr)
    context_block = retriever.format_context(retrieved)
    sources = [m.get("source", "") for m in retrieved.metadatas]
    messages = tutor.build_messages(context_block, message, cefr,
                                    history=history_from_db(user_id))
    return mode, sources, messages


def _finish(user_id: str, mode: str, message: str,
            sources: list[str], raw: str) -> tuple[str, str]:
    """Parse raw model output and persist the interaction. Returns (reasoning, response)."""
    reasoning, response = tutor.parse_output(raw)
    state_store.log_interaction(
        user_id, mode=mode, student_input=message, sources=sources,
        reasoning=reasoning, response=response,
    )
    return reasoning, response


def chat_blocking(user_id: str, message: str, level: str | None = None,
                  forced_mode: str | None = None) -> dict:
    cefr = resolve_level(user_id, level)
    mode, sources, messages = _setup(user_id, message, cefr, forced_mode)
    raw = tutor.generate(messages)
    reasoning, response = _finish(user_id, mode, message, sources, raw)
    return {
        "response": response,
        "reasoning": reasoning,
        "mode": mode,
        "sources": sources,
        "cefr_level": cefr,
    }


def chat_stream(user_id: str, message: str, level: str | None = None,
                forced_mode: str | None = None):
    """Generator. Yields {"type":"token","text":...} while generating, then
    {"type":"done","mode":...,"sources":...,"reasoning":...,"cefr_level":...}."""
    cefr = resolve_level(user_id, level)
    mode, sources, messages = _setup(user_id, message, cefr, forced_mode)

    filt = tutor.VisibleStreamFilter()
    raw_parts: list[str] = []
    for chunk in tutor.generate_stream(messages):
        raw_parts.append(chunk)
        visible = filt.feed(chunk)
        if visible:
            yield {"type": "token", "text": visible}
    tail = filt.finalize()
    if tail:
        yield {"type": "token", "text": tail}

    reasoning, response = _finish(user_id, mode, message, sources, "".join(raw_parts))
    yield {"type": "done", "mode": mode, "sources": sources,
           "reasoning": reasoning, "cefr_level": cefr}
