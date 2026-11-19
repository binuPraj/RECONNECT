"""Replaceable SQLite storage for durable memories."""
from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from typing import Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEMORY_DB_PATH = PROJECT_ROOT / "uploads" / "data" / "memory" / "memories.db"

class MemoryStore(Protocol):
    def create(self, memory: dict) -> None: ...
    def get(self, memory_id: str) -> dict | None: ...
    def update(self, memory: dict) -> None: ...
    def get_recent(self, participant: str, limit: int = 3, patient_id: int | None = None) -> list[dict]: ...
    def list_all(self) -> list[dict]: ...

class SQLiteMemoryStore:
    """Simple local implementation of the memory-store interface."""
    def __init__(self, path: str | Path = DEFAULT_MEMORY_DB_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS memories (
              memory_id TEXT PRIMARY KEY, category TEXT NOT NULL, summary TEXT NOT NULL,
              patient_id INTEGER,
              participants TEXT NOT NULL, about TEXT NOT NULL DEFAULT '[]', reported_by TEXT NOT NULL DEFAULT '[]', source_events TEXT NOT NULL,
              earliest_event_time TEXT NOT NULL, latest_event_time TEXT NOT NULL,
              importance TEXT NOT NULL, entities TEXT NOT NULL, topic TEXT,
              emotion TEXT NOT NULL DEFAULT 'neutral', created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL, event_starts TEXT NOT NULL DEFAULT '[]',
              update_history TEXT NOT NULL DEFAULT '[]',
              updated_events TEXT NOT NULL DEFAULT '[]',
              needs_review INTEGER NOT NULL DEFAULT 0)""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_updated ON memories(updated_at DESC)")
            columns = {row[1] for row in conn.execute("PRAGMA table_info(memories)")}
            if "about" not in columns:
                conn.execute("ALTER TABLE memories ADD COLUMN about TEXT NOT NULL DEFAULT '[]'")
            if "reported_by" not in columns:
                conn.execute("ALTER TABLE memories ADD COLUMN reported_by TEXT NOT NULL DEFAULT '[]'")
            if "event_starts" not in columns:
                conn.execute("ALTER TABLE memories ADD COLUMN event_starts TEXT NOT NULL DEFAULT '[]'")
            if "needs_review" not in columns:
                conn.execute("ALTER TABLE memories ADD COLUMN needs_review INTEGER NOT NULL DEFAULT 0")
            if "patient_id" not in columns:
                conn.execute("ALTER TABLE memories ADD COLUMN patient_id INTEGER")
            if "update_history" not in columns:
                conn.execute("ALTER TABLE memories ADD COLUMN update_history TEXT NOT NULL DEFAULT '[]'")
            if "updated_events" not in columns:
                conn.execute("ALTER TABLE memories ADD COLUMN updated_events TEXT NOT NULL DEFAULT '[]'")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_patient_updated ON memories(patient_id, updated_at DESC)")
            conn.execute("""CREATE TABLE IF NOT EXISTS time_refs (
                time_ref_id TEXT PRIMARY KEY, memory_id TEXT NOT NULL, phrase TEXT NOT NULL,
                tense TEXT NOT NULL, anchor TEXT NOT NULL, start TEXT, end TEXT, part TEXT,
                granularity TEXT, status TEXT NOT NULL, created_at TEXT NOT NULL,
                superseded_at TEXT, superseded_by TEXT, last_confirmed_at TEXT, what_was_mentioned TEXT NOT NULL DEFAULT '',
                topic TEXT NOT NULL DEFAULT '', concerned_with TEXT NOT NULL DEFAULT '',
                commitment_by TEXT NOT NULL DEFAULT '',
                FOREIGN KEY(memory_id) REFERENCES memories(memory_id))""")
            time_ref_columns = {row[1] for row in conn.execute("PRAGMA table_info(time_refs)")}
            if "what_was_mentioned" not in time_ref_columns:
                conn.execute("ALTER TABLE time_refs ADD COLUMN what_was_mentioned TEXT NOT NULL DEFAULT ''")
            if "superseded_by" not in time_ref_columns:
                conn.execute("ALTER TABLE time_refs ADD COLUMN superseded_by TEXT")
            for column in ("topic", "concerned_with", "commitment_by"):
                if column not in time_ref_columns:
                    conn.execute(f"ALTER TABLE time_refs ADD COLUMN {column} TEXT NOT NULL DEFAULT ''")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_time_refs_window ON time_refs(status, start, end)")
    def _connect(self):
        conn = sqlite3.connect(self.path); conn.row_factory = sqlite3.Row; return conn
    @staticmethod
    def _row(row):
        value = dict(row)
        for field in ("participants", "about", "reported_by", "source_events", "entities", "event_starts", "update_history", "updated_events"): value[field] = json.loads(value[field])
        value["needs_review"] = bool(value["needs_review"])
        value["time_refs"] = []
        # Migrate the earlier flat about/reported_by representation in-memory.
        if value["about"] and isinstance(value["about"][0], str):
            reporters = value.get("reported_by", [])
            value["about"] = [{"person": person, "reported_by": reporters[min(index, len(reporters) - 1)] if reporters else "unknown", "what_was_said": "Legacy record; original claim unavailable."} for index, person in enumerate(value["about"])]
        return value
    @staticmethod
    def _values(memory):
        return (memory["memory_id"], memory.get("patient_id"), memory["category"], memory["summary"], json.dumps(memory["participants"]), json.dumps(memory.get("about", [])), "[]", json.dumps(memory["source_events"]), memory["earliest_event_time"], memory["latest_event_time"], memory["importance"], json.dumps(memory["entities"]), memory.get("topic"), memory.get("emotion", "neutral"), memory["created_at"], memory["updated_at"], json.dumps(memory.get("event_starts", [])), json.dumps(memory.get("update_history", [])), json.dumps(memory.get("updated_events", [])), int(bool(memory.get("needs_review", False))))
    def create(self, memory):
        with self._connect() as conn:
            conn.execute("""INSERT INTO memories (
                memory_id, patient_id, category, summary, participants, about, reported_by,
                source_events, earliest_event_time, latest_event_time, importance,
                entities, topic, emotion, created_at, updated_at, event_starts, update_history, updated_events, needs_review
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", self._values(memory))
            self._save_time_refs(conn, memory)
    def get(self, memory_id):
        with self._connect() as conn: row = conn.execute("SELECT * FROM memories WHERE memory_id=?", (memory_id,)).fetchone()
        value = self._row(row) if row else None
        if value is not None: value["time_refs"] = self.time_refs(memory_id)
        return value
    def update(self, memory):
        with self._connect() as conn:
            result = conn.execute("""UPDATE memories SET patient_id=?, category=?, summary=?, participants=?, about=?, reported_by=?, source_events=?, earliest_event_time=?, latest_event_time=?, importance=?, entities=?, topic=?, emotion=?, created_at=?, updated_at=?, event_starts=?, update_history=?, updated_events=?, needs_review=? WHERE memory_id=?""", self._values(memory)[1:] + (memory["memory_id"],))
            if result.rowcount != 1: raise KeyError(f"Memory does not exist: {memory['memory_id']}")
            self._save_time_refs(conn, memory)

    def reconcile_concern_entities(self) -> int:
        """Remove duplicate person/relation entities already represented by about.

        The durable `about` record is the source for a CONCERNS graph edge.
        Keeping the same name in `entities` would create a redundant MENTIONS
        projection and a second graph node.
        """
        changed = 0
        with self._connect() as conn:
            rows = conn.execute("SELECT memory_id, about, entities FROM memories").fetchall()
            for row in rows:
                try:
                    about, entities = json.loads(row["about"]), json.loads(row["entities"])
                except (TypeError, json.JSONDecodeError):
                    continue
                if not isinstance(about, list) or not isinstance(entities, list):
                    continue
                entity_types = {
                    self._entity_key(entry.get("text")): str(entry.get("type", "")).upper()
                    for entry in entities if isinstance(entry, dict) and entry.get("text")
                }
                normalized_about = []
                for entry in about:
                    if not isinstance(entry, dict):
                        normalized_about.append(entry)
                        continue
                    value = dict(entry)
                    key = self._entity_key(value.get("person"))
                    value["type"] = str(value.get("type") or entity_types.get(key) or "PERSON").upper()
                    normalized_about.append(value)
                concern_keys = {self._entity_key(entry.get("person")) for entry in normalized_about if isinstance(entry, dict)}
                normalized_entities = [
                    entry for entry in entities
                    if not isinstance(entry, dict) or self._entity_key(entry.get("text")) not in concern_keys
                ]
                if normalized_about != about or normalized_entities != entities:
                    conn.execute("UPDATE memories SET about=?, entities=? WHERE memory_id=?", (
                        json.dumps(normalized_about), json.dumps(normalized_entities), row["memory_id"],
                    ))
                    changed += 1
        return changed

    @staticmethod
    def _entity_key(value) -> str:
        return " ".join(str(value or "").split()).casefold()

    @staticmethod
    def _save_time_refs(conn, memory):
        for ref in memory.get("time_refs", []):
            conn.execute("""INSERT INTO time_refs(time_ref_id,memory_id,phrase,tense,anchor,start,end,part,granularity,status,created_at,superseded_at,superseded_by,last_confirmed_at,what_was_mentioned,topic,concerned_with,commitment_by)
                VALUES(:time_ref_id,:memory_id,:phrase,:tense,:anchor,:start,:end,:part,:granularity,:status,:created_at,:superseded_at,:superseded_by,:last_confirmed_at,:what_was_mentioned,:topic,:concerned_with,:commitment_by)
                ON CONFLICT(time_ref_id) DO UPDATE SET memory_id=excluded.memory_id, phrase=excluded.phrase, tense=excluded.tense, anchor=excluded.anchor, start=excluded.start, end=excluded.end, part=excluded.part, granularity=excluded.granularity, status=excluded.status, superseded_at=excluded.superseded_at, superseded_by=excluded.superseded_by, last_confirmed_at=excluded.last_confirmed_at, what_was_mentioned=excluded.what_was_mentioned, topic=excluded.topic, concerned_with=excluded.concerned_with, commitment_by=excluded.commitment_by""", {"memory_id": memory["memory_id"], "start": None, "end": None, "part": None, "granularity": None, "superseded_at": None, "superseded_by": None, "last_confirmed_at": None, "what_was_mentioned": memory.get("summary", ""), "topic": "", "concerned_with": "", "commitment_by": "", **ref})
    def time_refs(self, memory_id, active_only=False):
        query = "SELECT * FROM time_refs WHERE memory_id=?" + (" AND status='active'" if active_only else "")
        with self._connect() as conn: return [dict(row) for row in conn.execute(query, (memory_id,))]
    def get_recent(self, participant, limit=3, patient_id: int | None = None):
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories" + (" WHERE patient_id=?" if patient_id is not None else "") + " ORDER BY updated_at DESC",
                (patient_id,) if patient_id is not None else (),
            ).fetchall()
        values = [self._row(r) for r in rows if participant in json.loads(r["participants"])][:limit]
        for value in values: value["time_refs"] = self.time_refs(value["memory_id"])
        return values

    def get_upcoming(self, now: str, limit: int = 10, patient_id: int | None = None) -> list[dict]:
        """Return memories with an active, future TimeRef, independent of speaker."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT DISTINCT m.* FROM memories m
                JOIN time_refs t ON t.memory_id=m.memory_id
                WHERE t.status='active' AND t.end > ?
                  AND (? IS NULL OR m.patient_id = ?)
                ORDER BY t.end ASC LIMIT ?
            """, (now, patient_id, patient_id, limit)).fetchall()
        values = [self._row(row) for row in rows]
        for value in values:
            value["time_refs"] = self.time_refs(value["memory_id"])
        return values

    def get_many(self, memory_ids: list[str]) -> list[dict]:
        if not memory_ids:
            return []
        placeholders = ",".join("?" for _ in memory_ids)
        with self._connect() as conn:
            rows = conn.execute(f"SELECT * FROM memories WHERE memory_id IN ({placeholders})", memory_ids).fetchall()
        by_id = {row["memory_id"]: self._row(row) for row in rows}
        for value in by_id.values():
            value["time_refs"] = self.time_refs(value["memory_id"])
        return [by_id[memory_id] for memory_id in memory_ids if memory_id in by_id]

    def list_all(self):
        with self._connect() as conn: rows = conn.execute("SELECT * FROM memories ORDER BY updated_at DESC").fetchall()
        values = [self._row(row) for row in rows]
        for value in values: value["time_refs"] = self.time_refs(value["memory_id"])
        return values
