"""Active query sessions. Gap is measured from the last question."""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from retrival.config import DATA_DIR, DB_PATH, SESSION_GAP_SECONDS

__all__ = ["SESSION_GAP_SECONDS", "get_or_start_session", "public_session", "touch_session"]
from retrival.debug_log import init_debug_db, now_iso

TZ = ZoneInfo("Asia/Kathmandu")


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_session_db() -> None:
    init_debug_db()
    with _connect() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                patient_id INTEGER NOT NULL,
                started_at TEXT NOT NULL,
                last_question_at TEXT NOT NULL,
                turns_json TEXT NOT NULL DEFAULT '[]',
                last_query_json TEXT,
                focus_memory_ids_json TEXT NOT NULL DEFAULT '[]',
                closed_at TEXT
            )"""
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_patient ON sessions(patient_id, closed_at)"
        )


def _parse(row: sqlite3.Row) -> dict:
    value = dict(row)
    value["turns"] = json.loads(value.pop("turns_json") or "[]")
    last = value.pop("last_query_json")
    value["last_query"] = json.loads(last) if last else None
    value["focus_memory_ids"] = json.loads(value.pop("focus_memory_ids_json") or "[]")
    return value


def _empty_session(patient_id: int) -> dict:
    stamp = now_iso()
    return {
        "session_id": str(uuid.uuid4()),
        "patient_id": patient_id,
        "started_at": stamp,
        "last_question_at": stamp,
        "turns": [],
        "last_query": None,
        "focus_memory_ids": [],
        "closed_at": None,
    }


def _save(session: dict) -> None:
    init_session_db()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO sessions (
                session_id, patient_id, started_at, last_question_at, turns_json,
                last_query_json, focus_memory_ids_json, closed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                last_question_at=excluded.last_question_at,
                turns_json=excluded.turns_json,
                last_query_json=excluded.last_query_json,
                focus_memory_ids_json=excluded.focus_memory_ids_json,
                closed_at=excluded.closed_at
            """,
            (
                session["session_id"],
                session["patient_id"],
                session["started_at"],
                session["last_question_at"],
                json.dumps(session.get("turns") or []),
                json.dumps(session.get("last_query")) if session.get("last_query") is not None else None,
                json.dumps(session.get("focus_memory_ids") or []),
                session.get("closed_at"),
            ),
        )


def _active(patient_id: int) -> dict | None:
    init_session_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE patient_id=? AND closed_at IS NULL ORDER BY last_question_at DESC LIMIT 1",
            (patient_id,),
        ).fetchone()
    return _parse(row) if row else None


def _expired(session: dict, now: datetime) -> bool:
    last = datetime.fromisoformat(session["last_question_at"])
    if last.tzinfo is None:
        last = last.replace(tzinfo=TZ)
    return now - last >= timedelta(seconds=SESSION_GAP_SECONDS)


def close_session(session: dict) -> None:
    session["closed_at"] = now_iso()
    _save(session)


def get_or_start_session(patient_id: int, now: datetime | None = None) -> tuple[dict, bool]:
    """Return (session, closed_previous). A new session starts empty."""
    now = now or datetime.now(TZ)
    current = _active(patient_id)
    closed_previous = False
    if current and _expired(current, now):
        close_session(current)
        closed_previous = True
        current = None
    if current is None:
        current = _empty_session(patient_id)
        _save(current)
    return current, closed_previous


def clear_conversation(session: dict) -> dict:
    session["turns"] = []
    session["last_query"] = None
    session["focus_memory_ids"] = []
    return session


def touch_session(session: dict, *, question: str, answer: str, parsed: dict, focus_memory_ids: list[str]) -> dict:
    stamp = now_iso()
    session["last_question_at"] = stamp
    session["last_query"] = parsed
    session["focus_memory_ids"] = focus_memory_ids[:8]
    turns = list(session.get("turns") or [])
    turns.append({"question": question, "answer": answer, "intent": parsed.get("intent"), "at": stamp})
    session["turns"] = turns[-3:]
    _save(session)
    return session


def public_session(session: dict) -> dict:
    return {
        "session_id": session["session_id"],
        "patient_id": session["patient_id"],
        "started_at": session["started_at"],
        "last_question_at": session["last_question_at"],
        "turns": session.get("turns") or [],
        "last_query": session.get("last_query"),
        "focus_memory_ids": session.get("focus_memory_ids") or [],
        "closed_at": session.get("closed_at"),
        "is_active": session.get("closed_at") is None,
        "gap_seconds": SESSION_GAP_SECONDS,
    }


def list_all_sessions(patient_id: int | None = None, limit: int = 50) -> list[dict]:
    init_session_db()
    with _connect() as conn:
        if patient_id is not None:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE patient_id=? ORDER BY started_at DESC LIMIT ?",
                (patient_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    sessions = [_parse(row) for row in rows]
    for s in sessions:
        s["is_active"] = s.get("closed_at") is None
    return sessions


def clear_all_sessions() -> None:
    init_session_db()
    with _connect() as conn:
        conn.execute("DELETE FROM sessions")

