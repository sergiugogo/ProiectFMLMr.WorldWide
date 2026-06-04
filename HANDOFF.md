# MrWorldwide — Person C → Person B Handoff

## What's done (frontend, no backend changes needed)

The Streamlit UI is complete and live on **port 8001**.

```
uvicorn api:app --port 8000   # your backend, as before
streamlit run frontend/app.py  # frontend, port set in frontend/.streamlit/config.toml
```

### Login / Auth — frontend-only, nothing for you to do

Students log in with a username + password stored in `frontend/.streamlit/secrets.toml`
(gitignored). The username becomes the `user_id` sent to every backend API call — so all
your existing state, history, and level logic works unchanged.

```toml
# frontend/.streamlit/secrets.toml  (add one line per student)
[users]
mihnea = "parola"
student1 = "bonjour"
```

No JWT, no session endpoint, no schema change. The backend never sees the password.

---

## One thing you need to implement — live reasoning streaming

### Why

The frontend has a "🧠 Thinking…" panel that should show the model's `<reasoning>` block
streaming in real time, before the visible response appears. Right now `VisibleStreamFilter`
silently discards reasoning tokens, so the panel only populates from the `done` event after
generation finishes. Adding a `reasoning_token` event type to the SSE stream is all that's
needed.

### Exact event spec

Emit this **before** any `token` events (reasoning precedes the visible response):

```json
{"type": "reasoning_token", "text": "...chunk of reasoning text..."}
```

The frontend handles it automatically — no changes needed on the UI side.

Once these events are flowing, you can drop the `reasoning` field from the `done` event
(the frontend falls back to it when no `reasoning_token` events arrived, so removing it is
optional and can be done at any point).

### Where to make the change

**`tutor.py`** — add a `ReasoningStreamFilter` class (inverse of `VisibleStreamFilter`):

```python
class ReasoningStreamFilter:
    """Passes through only the content inside <think>/<reasoning> blocks.
    Mirror of VisibleStreamFilter — feed the same raw chunks into both."""

    _OPEN  = re.compile(r"<(think|reasoning)\b[^>]*>", re.IGNORECASE)
    _CLOSE = re.compile(r"</(think|reasoning)>",       re.IGNORECASE)
    _HOLD  = 12

    def __init__(self):
        self.buf = ""
        self.in_block = False

    def feed(self, chunk: str) -> str:
        self.buf += chunk
        out: list[str] = []
        while True:
            if self.in_block:
                m = self._CLOSE.search(self.buf)
                if m:
                    out.append(self.buf[: m.start()])
                    self.in_block = False
                    self.buf = self.buf[m.end():]
                    continue
                # still inside block — emit all but the hold window
                if len(self.buf) > self._HOLD:
                    out.append(self.buf[: -self._HOLD])
                    self.buf = self.buf[-self._HOLD:]
                break
            else:
                m = self._OPEN.search(self.buf)
                if m:
                    self.buf = self.buf[m.end():]
                    self.in_block = True
                    continue
                # not in a block — discard but keep hold window to catch opening tag
                if len(self.buf) > self._HOLD:
                    self.buf = self.buf[-self._HOLD:]
                break
        return "".join(out)

    def finalize(self) -> str:
        out = self.buf if self.in_block else ""
        self.buf = ""
        return out
```

**`service.py`** — update `chat_stream` to run both filters and emit `reasoning_token`:

```python
def chat_stream(user_id, message, level=None, forced_mode=None):
    cefr = resolve_level(user_id, level)
    mode, sources, messages = _setup(user_id, message, cefr, forced_mode)

    vis_filt  = tutor.VisibleStreamFilter()
    reas_filt = tutor.ReasoningStreamFilter()          # ← new
    raw_parts: list[str] = []

    for chunk in tutor.generate_stream(messages):
        raw_parts.append(chunk)

        visible   = vis_filt.feed(chunk)
        reasoning = reas_filt.feed(chunk)              # ← new

        if reasoning:                                  # ← new
            yield {"type": "reasoning_token", "text": reasoning}
        if visible:
            yield {"type": "token", "text": visible}

    tail_vis  = vis_filt.finalize()
    tail_reas = reas_filt.finalize()                   # ← new

    if tail_reas:                                      # ← new
        yield {"type": "reasoning_token", "text": tail_reas}
    if tail_vis:
        yield {"type": "token", "text": tail_vis}

    reasoning_full, response = _finish(user_id, mode, message, sources, "".join(raw_parts))
    yield {"type": "done", "mode": mode, "sources": sources,
           "reasoning": reasoning_full, "cefr_level": cefr}
```

> **Note on ordering:** the model generates `<reasoning>...</reasoning>` *before* the
> visible response, so `reasoning_token` events will naturally precede `token` events.
> The `VisibleStreamFilter` and `ReasoningStreamFilter` are fed the same raw chunks in
> the same loop — no buffering or coordination needed.

### Testing the change

```bash
# With the model running, send a request with debug=true and watch the SSE stream:
curl -N -X POST http://localhost:8000/chat/stream?debug=true \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test","message":"Je suis 20 ans","level":"A1","forced_mode":null}'

# You should now see reasoning_token events before the token events:
# data: {"type": "reasoning_token", "text": "The student used être instead of avoir..."}
# data: {"type": "token", "text": "Belle tentative !"}
# ...
# data: {"type": "done", ...}
```

---

## Quick reference — full SSE event sequence after the change

```
data: {"type": "reasoning_token", "text": "..."}   ← reasoning, streamed live
data: {"type": "reasoning_token", "text": "..."}   ← (one per raw chunk)
data: {"type": "token",           "text": "..."}   ← visible response
data: {"type": "token",           "text": "..."}
data: {"type": "done",  "mode": "error",
       "sources": ["generated.md"],
       "reasoning": "...",                          ← still present as fallback
       "cefr_level": "A1"}
```

## Running the full stack locally

```bash
# Terminal 1 — backend
uvicorn api:app --port 8000 --reload

# Terminal 2 — frontend
cd frontend
streamlit run app.py
# → http://localhost:8001
```
