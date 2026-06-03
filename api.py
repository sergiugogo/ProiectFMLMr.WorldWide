from __future__ import annotations

import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import service
import state_store
from config import CEFR_LEVELS

app = FastAPI(title="MrWorldwide", description="Socratic French tutor (RAG + SLM)")

# Set CORS_ORIGINS to a comma-separated list of allowed origins in production.
_cors_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    user_id: str = Field(..., description="student identifier / conversation key")
    message: str = Field(..., description="the student's text")
    level: str | None = Field(None, description="override CEFR level (A1..C2)")
    forced_mode: str | None = Field(None, description="force retrieval mode: 'error' | 'vocab'")


class ChatResponse(BaseModel):
    response: str
    mode: str
    sources: list[str]
    cefr_level: str
    reasoning: str | None = None  # only populated when debug=true


class LevelRequest(BaseModel):
    level: str = Field(..., description=f"one of {CEFR_LEVELS}")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, debug: bool = False) -> ChatResponse:
    result = service.chat_blocking(
        req.user_id, req.message, level=req.level, forced_mode=req.forced_mode)
    return ChatResponse(
        response=result["response"],
        mode=result["mode"],
        sources=result["sources"],
        cefr_level=result["cefr_level"],
        reasoning=result["reasoning"] if debug else None,
    )


@app.post("/chat/stream")
def chat_stream(req: ChatRequest, debug: bool = False) -> StreamingResponse:
    """SSE stream. Each event is `data: <json>\\n\\n`.
    Emits {"type":"token","text":"..."} while generating, then
    {"type":"done","mode":...,"sources":[...],"cefr_level":...} to close.
    Reasoning is included in the done event only when debug=true.
    """
    def event_source():
        for ev in service.chat_stream(
            req.user_id, req.message, level=req.level, forced_mode=req.forced_mode
        ):
            if ev["type"] == "done" and not debug:
                ev = {k: v for k, v in ev.items() if k != "reasoning"}
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/users/{user_id}/level")
def set_level(user_id: str, req: LevelRequest) -> dict:
    if req.level not in CEFR_LEVELS:
        raise HTTPException(status_code=422,
                            detail=f"bad level; use {list(CEFR_LEVELS)}")
    state_store.get_or_create_user(user_id)
    state_store.set_level(user_id, req.level)
    return {"ok": True, "user_id": user_id, "cefr_level": req.level}


@app.get("/users/{user_id}/history")
def history(user_id: str, limit: int = 10) -> dict:
    return {"user_id": user_id,
            "interactions": state_store.recent_interactions(user_id, limit=limit)}
