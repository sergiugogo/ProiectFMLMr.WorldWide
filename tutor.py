from __future__ import annotations

import os
import re

SYSTEM_PROMPT = (
    "You are MrWorldwide, a Socratic French language tutor. You help students "
    "learn French by guiding them to discover correct answers themselves — never "
    "by giving direct corrections or translations.\n\n"
    "Rules:\n"
    "1. NEVER give the correct answer directly. Use hints, leading questions, or "
    "mini-exercises.\n"
    "2. Reason about the student's error inside <reasoning>...</reasoning> tags "
    "(hidden from student).\n"
    "3. Your visible response (after </reasoning>) should be encouraging and end "
    "with a guiding question.\n"
    "4. Match language to the student's level: A1 = mostly English, A2 = mixed, "
    "B1 = mostly French, B2 = all French.\n"
    "5. If the input has no errors, praise and give a follow-up exercise."
)

# INFERENCE_BACKEND=lmstudio uses LM Studio's local OpenAI-compatible server.
# INFERENCE_BACKEND=ollama (default) uses the Ollama daemon.
INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "ollama").lower()

# Ollama settings
OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL", "hf.co/MihaiRusu/mrworldwide-deepseek-r1-7b-socratic-fr:Q4_K_M"
)

# LM Studio settings — model name must match what is loaded in LM Studio.
# Leave LM_STUDIO_MODEL unset to use whatever model is currently loaded.
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "local-model")

GEN_OPTIONS = {
    "temperature": float(os.getenv("TUTOR_TEMPERATURE", "0.7")),
    "top_p": float(os.getenv("TUTOR_TOP_P", "0.95")),
    "max_tokens": int(os.getenv("TUTOR_MAX_TOKENS", "400")),
}


def _lm_studio_client():
    from openai import OpenAI
    return OpenAI(base_url=LM_STUDIO_BASE_URL, api_key="lm-studio")


def build_user_turn(context_block: str, student_input: str, cefr_level: str) -> str:
    parts = []
    if context_block:
        parts.append(context_block)
    parts.append(f"(Student level: {cefr_level})")
    parts.append(student_input)
    return "\n\n".join(parts)


def build_messages(context_block: str, student_input: str, cefr_level: str,
                   history: list[dict] | None = None) -> list[dict]:
    """Returns a ChatML message list: system prompt + prior turns + current turn."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append(
        {"role": "user", "content": build_user_turn(context_block, student_input, cefr_level)}
    )
    return messages


def generate(messages: list[dict]) -> str:
    """Calls the configured backend and returns the full raw assistant text."""
    if INFERENCE_BACKEND == "lmstudio":
        client = _lm_studio_client()
        resp = client.chat.completions.create(
            model=LM_STUDIO_MODEL,
            messages=messages,
            temperature=GEN_OPTIONS["temperature"],
            top_p=GEN_OPTIONS["top_p"],
            max_tokens=GEN_OPTIONS["max_tokens"],
        )
        msg = resp.choices[0].message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None) or ""
        if content and reasoning:
            # Normal case: LM Studio separated them — recombine for parse_output.
            return f"<reasoning>{reasoning}</reasoning>\n{content}"
        if content:
            return content
        # LM Studio put the entire model output into reasoning_content (happens when
        # the model's full response was inside <think> tags). Use it as-is — it already
        # contains any <reasoning> blocks that parse_output knows how to split.
        return reasoning

    import ollama
    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        options={
            "temperature": GEN_OPTIONS["temperature"],
            "top_p": GEN_OPTIONS["top_p"],
            "num_predict": GEN_OPTIONS["max_tokens"],
        },
    )
    return resp["message"]["content"]


def generate_stream(messages: list[dict]):
    """Yields raw assistant text chunks from the configured backend.
    Chunks are unfiltered — pass through VisibleStreamFilter to hide <think>/<reasoning>."""
    if INFERENCE_BACKEND == "lmstudio":
        client = _lm_studio_client()
        # LM Studio may put the full output in reasoning_content with empty content
        # (when the model wrapped everything in <think> tags), or split them normally.
        # Buffer reasoning chunks until we know whether content is also coming.
        reasoning_buf: list[str] = []
        got_content = False
        for chunk in client.chat.completions.create(
            model=LM_STUDIO_MODEL,
            messages=messages,
            temperature=GEN_OPTIONS["temperature"],
            top_p=GEN_OPTIONS["top_p"],
            max_tokens=GEN_OPTIONS["max_tokens"],
            stream=True,
        ):
            delta = chunk.choices[0].delta
            reasoning_piece = getattr(delta, "reasoning_content", None) or ""
            content_piece = delta.content or ""
            if content_piece:
                if not got_content and reasoning_buf:
                    yield "<reasoning>" + "".join(reasoning_buf) + "</reasoning>\n"
                    reasoning_buf = []
                got_content = True
                yield content_piece
            elif reasoning_piece:
                reasoning_buf.append(reasoning_piece)
        if reasoning_buf:
            # No content arrived — full output is in reasoning_content, yield it raw.
            for piece in reasoning_buf:
                yield piece
        return

    import ollama
    for part in ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        options={
            "temperature": GEN_OPTIONS["temperature"],
            "top_p": GEN_OPTIONS["top_p"],
            "num_predict": GEN_OPTIONS["max_tokens"],
        },
        stream=True,
    ):
        piece = part.get("message", {}).get("content", "")
        if piece:
            yield piece


_TAG_BLOCK = re.compile(r"<(think|reasoning)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_STRAY_TAGS = re.compile(r"</?(think|reasoning)\b[^>]*>", re.IGNORECASE)


def parse_output(raw: str) -> tuple[str, str]:
    """Splits raw generation into (reasoning_trace, visible_response).
    Extracts any <think>/<reasoning> block content as the trace and strips tags
    from the visible text. Handles unterminated tags from truncated output."""
    raw = raw or ""
    reasoning_parts = [m.group(0) for m in _TAG_BLOCK.finditer(raw)]
    visible = _TAG_BLOCK.sub("", raw)

    m = re.search(r"<(think|reasoning)\b[^>]*>", visible, re.IGNORECASE)
    if m:
        reasoning_parts.append(visible[m.start():])
        visible = visible[: m.start()]

    visible = _STRAY_TAGS.sub("", visible).strip()
    reasoning = _STRAY_TAGS.sub("", "\n".join(reasoning_parts)).strip()
    return reasoning, visible


class VisibleStreamFilter:
    """Strips <think>/<reasoning> blocks from a live token stream incrementally.
    Feed raw chunks via feed(); call finalize() at the end to flush the buffer.
    Holds back _HOLD bytes each step to avoid emitting a partial opening tag
    that spans a chunk boundary (e.g. '<reaso' + 'ning>')."""

    _OPEN = re.compile(r"<(think|reasoning)\b[^>]*>", re.IGNORECASE)
    _CLOSE = re.compile(r"</(think|reasoning)>", re.IGNORECASE)
    _HOLD = 12  # len("</reasoning>")

    def __init__(self):
        self.buf = ""
        self.in_hidden = False

    def feed(self, chunk: str) -> str:
        self.buf += chunk
        out: list[str] = []
        while True:
            if self.in_hidden:
                m = self._CLOSE.search(self.buf)
                if not m:
                    if len(self.buf) > self._HOLD:
                        self.buf = self.buf[-self._HOLD:]
                    break
                self.in_hidden = False
                self.buf = self.buf[m.end():]
                continue
            m = self._OPEN.search(self.buf)
            if m:
                out.append(self.buf[: m.start()])
                self.in_hidden = True
                self.buf = self.buf[m.end():]
                continue
            if len(self.buf) > self._HOLD:
                out.append(self.buf[: -self._HOLD])
                self.buf = self.buf[-self._HOLD:]
            break
        return "".join(out)

    def finalize(self) -> str:
        if self.in_hidden:
            self.buf = ""
            return ""
        out = _STRAY_TAGS.sub("", self.buf)
        self.buf = ""
        return out
