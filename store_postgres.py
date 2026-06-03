from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager

from config import CEFR_LEVELS

# Set DATABASE_URL to e.g. postgresql://user:pass@localhost:5432/mrworldwide
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/mrworldwide")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id      TEXT PRIMARY KEY,
    cefr_level   TEXT NOT NULL DEFAULT 'A1',
    stage        TEXT,
    created_at   DOUBLE PRECISION NOT NULL,
    updated_at   DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS interactions (
    id            BIGSERIAL PRIMARY KEY,
    user_id       TEXT NOT NULL,
    ts            DOUBLE PRECISION NOT NULL,
    mode          TEXT,
    student_input TEXT,
    sources       TEXT,
    reasoning     TEXT,
    response      TEXT
);
CREATE INDEX IF NOT EXISTS ix_interactions_user ON interactions(user_id, ts);
"""


@contextmanager
def _conn(dsn: str = DATABASE_URL):
    import psycopg
    from psycopg.rows import dict_row

    con = psycopg.connect(dsn, row_factory=dict_row)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db(dsn: str = DATABASE_URL) -> None:
    with _conn(dsn) as con:
        con.execute(_SCHEMA)


def get_or_create_user(user_id: str, default_level: str = "A1",
                       dsn: str = DATABASE_URL) -> dict:
    now = time.time()
    with _conn(dsn) as con:
        row = con.execute(
            "SELECT * FROM users WHERE user_id=%s", (user_id,)
        ).fetchone()
        if row is None:
            con.execute(
                "INSERT INTO users(user_id, cefr_level, stage, created_at, updated_at) "
                "VALUES (%s,%s,%s,%s,%s) ON CONFLICT (user_id) DO NOTHING",
                (user_id, default_level, None, now, now),
            )
            return {"user_id": user_id, "cefr_level": default_level,
                    "stage": None, "created_at": now, "updated_at": now}
        return dict(row)


def set_level(user_id: str, cefr_level: str, dsn: str = DATABASE_URL) -> None:
    if cefr_level not in CEFR_LEVELS:
        raise ValueError(f"bad cefr_level {cefr_level!r}; use {CEFR_LEVELS}")
    with _conn(dsn) as con:
        con.execute(
            "UPDATE users SET cefr_level=%s, updated_at=%s WHERE user_id=%s",
            (cefr_level, time.time(), user_id),
        )


def set_stage(user_id: str, stage: str | None, dsn: str = DATABASE_URL) -> None:
    with _conn(dsn) as con:
        con.execute(
            "UPDATE users SET stage=%s, updated_at=%s WHERE user_id=%s",
            (stage, time.time(), user_id),
        )


def log_interaction(user_id: str, *, mode: str, student_input: str,
                    sources: list[str], reasoning: str, response: str,
                    dsn: str = DATABASE_URL) -> None:
    with _conn(dsn) as con:
        con.execute(
            "INSERT INTO interactions(user_id, ts, mode, student_input, sources, "
            "reasoning, response) VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (user_id, time.time(), mode, student_input,
             json.dumps(sources, ensure_ascii=False), reasoning, response),
        )


def recent_interactions(user_id: str, limit: int = 10,
                        dsn: str = DATABASE_URL) -> list[dict]:
    with _conn(dsn) as con:
        rows = con.execute(
            "SELECT * FROM interactions WHERE user_id=%s ORDER BY ts DESC LIMIT %s",
            (user_id, limit),
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        try:
            d["sources"] = json.loads(d.get("sources") or "[]")
        except (json.JSONDecodeError, TypeError):
            d["sources"] = []
        result.append(d)
    return result
