import json
import os
import tempfile

# Use a throwaway DB so the test never touches real user state.
os.environ["USER_STATE_DB"] = os.path.join(tempfile.gettempdir(), "mrww_test_api.db")

import service
import tutor
from fastapi.testclient import TestClient

# --- mock the model (both blocking and streaming) -------------------------- #
def fake_generate(messages):
    saw = "[CONTEXT]" in messages[-1]["content"]
    return (f"<think>x</think><reasoning>ctx={'yes' if saw else 'no'}</reasoning>\n"
            "Belle tentative ! Quel verbe pour l'âge ?")

def fake_stream(messages):
    # Deliberately split tags across chunk boundaries to exercise the filter.
    for piece in ["<think>scr", "atch</thi", "nk><reaso", "ning>anal",
                  "yze</reasoning>\nBon", "jour ! Quel ", "verbe ?"]:
        yield piece

tutor.generate = fake_generate
tutor.generate_stream = fake_stream

import api
client = TestClient(api.app)


def _sse_events(text):
    return [json.loads(line[len("data: "):])
            for line in text.splitlines() if line.startswith("data: ")]


def main():
    assert client.get("/health").json()["status"] == "ok"

    # set level
    r = client.post("/users/u1/level", json={"level": "A1"})
    assert r.json()["ok"] is True, r.json()

    # blocking, no debug -> reasoning hidden
    r = client.post("/chat", json={"user_id": "u1", "message": "Je suis 20 ans."})
    body = r.json()
    print("[blocking]", body["mode"], body["sources"][:1], "| resp:", body["response"][:40])
    assert body["mode"] == "error", body
    assert body["sources"], "expected retrieved grammar sources"
    assert body["reasoning"] is None, "reasoning must be hidden without debug"
    assert body["response"].startswith("Belle"), body

    # blocking, debug -> reasoning present and confirms context reached the model
    r = client.post("/chat?debug=true", json={"user_id": "u1", "message": "Je suis 20 ans."})
    body = r.json()
    assert body["reasoning"] and "ctx=yes" in body["reasoning"], body
    print("[blocking+debug] reasoning:", body["reasoning"])

    # streaming, no debug
    with client.stream("POST", "/chat/stream",
                       json={"user_id": "u1", "message": "How do you say breakfast?"}) as resp:
        text = "".join(resp.iter_text())
    events = _sse_events(text)
    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    done = [e for e in events if e["type"] == "done"][0]
    print("[stream] visible:", repr(tokens), "| mode:", done["mode"])
    assert "<" not in tokens and "reasoning" not in tokens, f"tag leaked: {tokens!r}"
    assert tokens.strip() == "Bonjour ! Quel verbe ?", repr(tokens)
    assert done["mode"] == "vocab", done
    assert "reasoning" not in done, "reasoning must be hidden in stream without debug"

    # streaming, debug -> done event carries reasoning
    with client.stream("POST", "/chat/stream?debug=true",
                       json={"user_id": "u1", "message": "How do you say breakfast?"}) as resp:
        text = "".join(resp.iter_text())
    done = [e for e in _sse_events(text) if e["type"] == "done"][0]
    assert "reasoning" in done and "analyze" in done["reasoning"], done
    print("[stream+debug] reasoning:", done["reasoning"])

    print("\nALL API CHECKS PASSED.")


if __name__ == "__main__":
    main()
