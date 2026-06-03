from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager

from config import CEFR_LEVELS

DB_PATH = os.getenv("USER_STATE_DB", "./user_state.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id      TEXT PRIMARY KEY,
    cefr_level   TEXT NOT NULL DEFAULT 'A1',
    stage        TEXT,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS interactions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       TEXT NOT NULL,
    ts            REAL NOT NULL,
    mode          TEXT,
    student_input TEXT,
    sources       TEXT,
    reasoning     TEXT,
    response      TEXT
);
CREATE INDEX IF NOT EXISTS ix_interactions_user ON interactions(user_id, ts);
"""


@contextmanager
def _conn(db_path: str = DB_PATH):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db(db_path: str = DB_PATH) -> None:
    with _conn(db_path) as con:
        con.executescript(_SCHEMA)


def get_or_create_user(user_id: str, default_level: str = "A1",
                       db_path: str = DB_PATH) -> dict:
    now = time.time()
    with _conn(db_path) as con:
        row = con.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
        if row is None:
            con.execute(
                "INSERT INTO users(user_id, cefr_level, stage, created_at, updated_at) "
                "VALUES (?,?,?,?,?)",
                (user_id, default_level, None, now, now),
            )
            return {"user_id": user_id, "cefr_level": default_level,
                    "stage": None, "created_at": now, "updated_at": now}
        return dict(row)


def set_level(user_id: str, cefr_level: str, db_path: str = DB_PATH) -> None:
    if cefr_level not in CEFR_LEVELS:
        raise ValueError(f"bad cefr_level {cefr_level!r}; use {CEFR_LEVELS}")
    with _conn(db_path) as con:
        con.execute(
            "UPDATE users SET cefr_level=?, updated_at=? WHERE user_id=?",
            (cefr_level, time.time(), user_id),
        )


def set_stage(user_id: str, stage: str | None, db_path: str = DB_PATH) -> None:
    with _conn(db_path) as con:
        con.execute(
            "UPDATE users SET stage=?, updated_at=? WHERE user_id=?",
            (stage, time.time(), user_id),
        )


def log_interaction(user_id: str, *, mode: str, student_input: str,
                    sources: list[str], reasoning: str, response: str,
                    db_path: str = DB_PATH) -> None:
    with _conn(db_path) as con:
        con.execute(
            "INSERT INTO interactions(user_id, ts, mode, student_input, sources, "
            "reasoning, response) VALUES (?,?,?,?,?,?,?)",
            (user_id, time.time(), mode, student_input,
             json.dumps(sources, ensure_ascii=False), reasoning, response),
        )


def recent_interactions(user_id: str, limit: int = 10,
                        db_path: str = DB_PATH) -> list[dict]:
    with _conn(db_path) as con:
        rows = con.execute(
            "SELECT * FROM interactions WHERE user_id=? ORDER BY ts DESC LIMIT ?",
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
