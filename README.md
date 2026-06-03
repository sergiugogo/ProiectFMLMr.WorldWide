# MrWorldwide — Socratic French Tutor

A RAG-powered language tutor that teaches French through the Socratic method — guiding students to discover correct answers themselves rather than giving direct corrections. Built on a fine-tuned DeepSeek-R1 7B model, ChromaDB for curriculum retrieval, and FastAPI for the HTTP interface.

## How it works

A student sends a message. A query router classifies it as either a grammar correction request (`error` mode) or a vocabulary query (`vocab` mode). The relevant chunks are retrieved from ChromaDB, injected as context into the model prompt, and the model responds in character as a Socratic tutor — never giving the answer outright, always asking a guiding question back.

```
student message
    → query router (keyword heuristic or GPT-4o-mini classifier)
    → ChromaDB retrieval (grammar_rules or vocabulary collection)
    → [CONTEXT] block + CEFR level hint
    → DeepSeek-R1 7B (fine-tuned, served via LM Studio or Ollama)
    → <reasoning> stripped, visible response returned
    → interaction logged to SQLite / Postgres
```

## Project structure

```
api.py                  FastAPI endpoints
service.py              Pipeline orchestration (shared by blocking + streaming)
retriever.py            Query router + ChromaDB retrieval
tutor.py                Model interface (LM Studio / Ollama), streaming filter
graph.py                LangGraph graph (used in offline tests with checkpointing)
config.py               ChromaDB client, embedding functions, CEFR constants
state_store.py          DB backend selector (SQLite or Postgres)
store_sqlite.py         SQLite backend
store_postgres.py       Postgres backend
ingest.py               Curriculum ingestion (offline, run once)
generate_curriculum.py  AI-generated curriculum authoring (offline, run once)
test_api.py             HTTP smoke tests (mocked model)
test_runtime.py         Pipeline integration tests (mocked model)
```

## Setup

**1. Install dependencies**
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

**2. Configure environment**
```bash
cp .env.example .env
# Edit .env — set your OPENAI_API_KEY and choose an inference backend
```

**3. Generate and ingest the curriculum**
```bash
# Generate grammar docs and vocabulary (calls OpenAI — runs once)
python generate_curriculum.py --lang fr --out ./curriculum/fr

# Ingest into ChromaDB (runs once, idempotent)
python ingest.py --grammar ./curriculum/fr/grammar --lang fr
python ingest.py --vocab ./curriculum/fr/vocab.json --lang fr
```

**4. Start the model**

**Option A — LM Studio (recommended, no C drive space needed)**
- Download [LM Studio](https://lmstudio.ai) and install it to any drive
- Search for `MihaiRusu/mrworldwide-deepseek-r1-7b-socratic-fr`, download the `Q4_K_M` file (~4.7 GB)
- Load the model → Local Server tab → Start Server
- Set `INFERENCE_BACKEND=lmstudio` in `.env`

**Option B — Ollama**
```bash
ollama run hf.co/MihaiRusu/mrworldwide-deepseek-r1-7b-socratic-fr:Q4_K_M
```
- Set `INFERENCE_BACKEND=ollama` in `.env`

**5. Run the API**
```bash
uvicorn api:app --reload --port 8000
```

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/chat` | Blocking chat — returns full response as JSON |
| `POST` | `/chat/stream` | Streaming chat — Server-Sent Events, token by token |
| `POST` | `/users/{user_id}/level` | Set a student's CEFR level (A1–C2) |
| `GET` | `/users/{user_id}/history` | Recent interactions for a student |

**Chat request**
```json
{
  "user_id": "student1",
  "message": "Je suis 20 ans.",
  "level": "A1",
  "forced_mode": null
}
```

**Chat response**
```json
{
  "response": "Belle tentative ! Pour exprimer l'âge en français, quel verbe utilise-t-on ?",
  "mode": "error",
  "sources": ["generated.md"],
  "cefr_level": "A1",
  "reasoning": null
}
```

Add `?debug=true` to either chat endpoint to include the model's hidden `<reasoning>` trace in the response.

**Streaming** (`/chat/stream`) emits SSE events:
```
data: {"type": "token", "text": "Belle "}
data: {"type": "token", "text": "tentative"}
data: {"type": "done", "mode": "error", "sources": [...], "cefr_level": "A1"}
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required for embeddings and curriculum generation |
| `INFERENCE_BACKEND` | `ollama` | `lmstudio` or `ollama` |
| `LM_STUDIO_MODEL` | `local-model` | Model identifier shown in LM Studio's API tab |
| `LM_STUDIO_BASE_URL` | `http://localhost:1234/v1` | LM Studio server URL |
| `OLLAMA_MODEL` | `hf.co/MihaiRusu/...` | Ollama model tag |
| `EMBEDDER` | `openai` | `openai` or `bge-m3` (local, no API cost) |
| `CHROMA_DIR` | `./chroma_db` | ChromaDB persistence directory |
| `USER_STATE_BACKEND` | `sqlite` | `sqlite` or `postgres` |
| `USER_STATE_DB` | `./user_state.db` | SQLite file path |
| `DATABASE_URL` | — | Postgres DSN (only when backend=postgres) |
| `USE_LLM_ROUTER` | `0` | Set to `1` to use GPT-4o-mini for query routing |
| `HISTORY_TURNS` | `6` | Number of prior turns replayed to the model |
| `GEN_MODEL` | `gpt-5.5` | Model used by `generate_curriculum.py` |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins for production |

## Running tests

Both test suites mock the model so they run without LM Studio or Ollama:
```bash
python test_runtime.py   # pipeline + graph wiring
python test_api.py       # HTTP endpoints
```

## Model

Fine-tuned from `DeepSeek-R1-Distill-Qwen-7B` via QLoRA on a Socratic French tutoring dataset. Covers CEFR levels A1–B2. The model is instructed to reason inside `<reasoning>...</reasoning>` tags (logged server-side, never shown to students) and respond with a guiding question rather than a direct correction.

Model card: [MihaiRusu/mrworldwide-deepseek-r1-7b-socratic-fr](https://huggingface.co/MihaiRusu/mrworldwide-deepseek-r1-7b-socratic-fr)
