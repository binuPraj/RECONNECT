"""SQLite log of parser/answer LLM calls, sessions, and questions."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

from retrival.config import DATA_DIR, DB_PATH

TZ = ZoneInfo("Asia/Kathmandu")


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_debug_db() -> None:
    with _connect() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS llm_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                patient_id INTEGER,
                session_id TEXT,
                question TEXT,
                kind TEXT NOT NULL,
                prompt TEXT NOT NULL,
                response TEXT,
                extra_json TEXT NOT NULL DEFAULT '{}'
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                patient_id INTEGER NOT NULL,
                session_id TEXT NOT NULL,
                question TEXT NOT NULL,
                parsed_json TEXT,
                resolved_json TEXT,
                intent TEXT,
                answer TEXT,
                focus_memory_ids TEXT,
                is_followup INTEGER NOT NULL DEFAULT 0,
                previous_session_closed INTEGER NOT NULL DEFAULT 0
            )"""
        )


        # Ensure records_json column exists
        columns = {col[1] for col in conn.execute("PRAGMA table_info(queries)")}
        if "records_json" not in columns:
            conn.execute("ALTER TABLE queries ADD COLUMN records_json TEXT")


def now_iso() -> str:
    return datetime.now(TZ).isoformat()


def log_llm(*, kind: str, prompt: str, response: str | None, patient_id: int | None,
            session_id: str | None, question: str | None, extra: dict | None = None) -> None:
    init_debug_db()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO llm_calls (created_at, patient_id, session_id, question, kind, prompt, response, extra_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (now_iso(), patient_id, session_id, question, kind, prompt, response, json.dumps(extra or {})),
        )


def log_query(row: dict) -> None:
    init_debug_db()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO queries (
                created_at, patient_id, session_id, question, parsed_json, resolved_json,
                intent, answer, focus_memory_ids, is_followup, previous_session_closed, records_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                now_iso(),
                row["patient_id"],
                row["session_id"],
                row["question"],
                json.dumps(row.get("parsed")),
                json.dumps(row.get("resolved")),
                row.get("intent"),
                row.get("answer"),
                json.dumps(row.get("focus_memory_ids") or []),
                int(bool(row.get("is_followup"))),
                int(bool(row.get("previous_session_closed"))),
                json.dumps(row.get("records")),
            ),
        )


def list_llm_calls(limit: int = 50) -> list[dict]:
    init_debug_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM llm_calls ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]


def list_queries(limit: int = 50) -> list[dict]:
    init_debug_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM queries ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]


def clear_all_debug_data() -> None:
    init_debug_db()
    with _connect() as conn:
        conn.execute("DELETE FROM queries")
        conn.execute("DELETE FROM llm_calls")

