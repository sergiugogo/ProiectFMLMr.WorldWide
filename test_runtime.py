from graph import build_graph
from langgraph.checkpoint.memory import MemorySaver
from tutor import parse_output


def fake_generate(messages):
    """Pretend to be DeepSeek: echo that it saw the context, emit both tags."""
    user_turn = messages[-1]["content"]
    saw_context = "[CONTEXT]" in user_turn
    return (
        "<think>internal scratch</think>"
        "<reasoning>The context was "
        + ("present" if saw_context else "MISSING")
        + ". I will hint, not correct.</reasoning>\n"
        "Belle tentative ! Quel verbe utilise-t-on pour l'âge en français ?"
    )


def test_parser():
    r, v = parse_output("<think>a</think><reasoning>b</reasoning>\nVisible text")
    assert "Visible text" == v, v
    assert "a" in r and "b" in r, r
    # unterminated reasoning
    r2, v2 = parse_output("<reasoning>only opener, truncated")
    assert v2 == "", repr(v2)
    print("parser OK")


def main():
    test_parser()
    graph = build_graph(generate_fn=fake_generate, checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "tester"}}

    # 1) error mode: a sentence with a mistake -> grammar collection
    out = graph.invoke(
        {"user_id": "tester", "cefr_level": "A1",
         "student_input": "Je suis 20 ans et je suis un étudiant."},
        config=cfg,
    )
    print(f"\n[turn 1] mode={out['mode']} sources={out['sources']}")
    print("context retrieved:", bool(out["context_block"]))
    assert out["mode"] == "error", out["mode"]
    assert out["context_block"], "expected grammar context"
    assert "MISSING" not in out["reasoning"], "model did not receive context!"
    assert out["response"].startswith("Belle"), out["response"]

    # 2) vocab mode: an explicit vocabulary ask -> vocab collection
    out2 = graph.invoke(
        {"user_id": "tester", "cefr_level": "A2",
         "student_input": "How do you say 'breakfast' in French? Give me some food words."},
        config=cfg,
    )
    print(f"[turn 2] mode={out2['mode']} sources={out2['sources'][:3]}...")
    assert out2["mode"] == "vocab", out2["mode"]
    assert out2["context_block"], "expected vocab context"

    # 3) history accumulated across turns (checkpointer working)
    assert len(out2["history"]) == 4, f"expected 4 history msgs, got {len(out2['history'])}"
    print(f"[history] {len(out2['history'])} messages persisted across turns")

    print("\nALL CHECKS PASSED — pipeline wired correctly.")


if __name__ == "__main__":
    main()
