from __future__ import annotations

import operator
from typing import Annotated, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

import retriever
import state_store
import tutor


class TutorState(TypedDict, total=False):
    user_id: str
    cefr_level: str
    forced_mode: str | None
    student_input: str
    mode: str
    context_block: str
    sources: list[str]
    raw_output: str
    reasoning: str
    response: str
    history: Annotated[list[dict], operator.add]


def _make_route(router_fn: Callable[..., str]):
    def _route(state: TutorState) -> dict:
        mode = router_fn(state["student_input"], forced=state.get("forced_mode"))
        return {"mode": mode}
    return _route


def _retrieve(state: TutorState) -> dict:
    r = retriever.retrieve(
        state["student_input"],
        mode=state["mode"],
        cefr_level=state.get("cefr_level", "A1"),
    )
    sources = [m.get("source", "") for m in r.metadatas]
    return {"context_block": retriever.format_context(r), "sources": sources}


def _make_generate(generate_fn: Callable[[list[dict]], str]):
    def _generate(state: TutorState) -> dict:
        messages = tutor.build_messages(
            state.get("context_block", ""),
            state["student_input"],
            state.get("cefr_level", "A1"),
            history=state.get("history", []),
        )
        return {"raw_output": generate_fn(messages)}
    return _generate


def _parse(state: TutorState) -> dict:
    reasoning, response = tutor.parse_output(state["raw_output"])
    return {"reasoning": reasoning, "response": response}


def _persist(state: TutorState) -> dict:
    state_store.log_interaction(
        state["user_id"],
        mode=state.get("mode", ""),
        student_input=state["student_input"],
        sources=state.get("sources", []),
        reasoning=state.get("reasoning", ""),
        response=state.get("response", ""),
    )
    turn = [
        {"role": "user", "content": state["student_input"]},
        {"role": "assistant", "content": state.get("response", "")},
    ]
    return {"history": turn}


def build_graph(generate_fn: Callable[[list[dict]], str] = tutor.generate,
                router_fn: Callable[..., str] = retriever.decide_mode,
                checkpointer=None):
    """Compiles the tutor graph: route → retrieve → generate → parse → persist.
    Pass generate_fn to swap in a stub for offline testing.
    Pass a LangGraph checkpointer (e.g. MemorySaver) to enable multi-turn history
    via LangGraph state instead of the user-state DB."""
    state_store.init_db()

    g = StateGraph(TutorState)
    g.add_node("route", _make_route(router_fn))
    g.add_node("retrieve", _retrieve)
    g.add_node("generate", _make_generate(generate_fn))
    g.add_node("parse", _parse)
    g.add_node("persist", _persist)

    g.add_edge(START, "route")
    g.add_edge("route", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "parse")
    g.add_edge("parse", "persist")
    g.add_edge("persist", END)

    return g.compile(checkpointer=checkpointer)
