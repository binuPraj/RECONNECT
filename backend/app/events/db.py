import sqlite3
import logging
import json
from datetime import datetime
from pathlib import Path

LOGGER = logging.getLogger(__name__)

# Resolve from this source file, not the shell's current working directory.
# The previous relative path could silently point at a different/empty SQLite
# database when the server was launched outside the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_DIR = PROJECT_ROOT / "uploads" / "data" / "events"
DB_PATH = DB_DIR / "events.db"

def init_db() -> None:
    """Initialize the SQLite database and create tables if they do not exist."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                patient_id INTEGER,
                status TEXT NOT NULL DEFAULT 'open',
                participants TEXT NOT NULL,
                start TEXT NOT NULL,
                end TEXT NOT NULL,
                location TEXT,
                recordings TEXT NOT NULL,
                segments TEXT NOT NULL,
                closed_at TEXT,
                llm_status TEXT NOT NULL DEFAULT 'not_eligible',
                consumed INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_status ON events(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_closed_at ON events(closed_at)")
        columns = {row[1] for row in conn.execute("PRAGMA table_info(events)")}
        if "llm_status" not in columns:
            conn.execute("ALTER TABLE events ADD COLUMN llm_status TEXT NOT NULL DEFAULT 'not_eligible'")
            conn.execute("UPDATE events SET llm_status = CASE WHEN status = 'closed' THEN 'pending' ELSE 'not_eligible' END")
        if "consumed" not in columns:
            conn.execute("ALTER TABLE events ADD COLUMN consumed INTEGER NOT NULL DEFAULT 0")
        if "patient_id" not in columns:
            conn.execute("ALTER TABLE events ADD COLUMN patient_id INTEGER")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_llm_status ON events(llm_status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_patient_status ON events(patient_id, status)")
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS grouping_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        
        # Initialize unconsumed_event_count if it doesn't exist
        conn.execute("""
            INSERT OR IGNORE INTO grouping_meta (key, value)
            VALUES ('unconsumed_event_count', '0')
        """)
        conn.commit()
    LOGGER.info("[db] Initialized event grouping database at %s", DB_PATH)

def get_connection() -> sqlite3.Connection:
    """Return a new SQLite connection with dict-like row factories."""
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def _event_from_row(row: sqlite3.Row) -> dict:
    """Convert SQLite JSON text columns into JSON values for callers."""
    event = dict(row)
    for column in ("participants", "recordings", "segments"):
        event[column] = json.loads(event[column])
    return event


def list_events(status: str | None = None, patient_id: int | None = None) -> list[dict]:
    """Return events, newest first, optionally limited to one status."""
    query = "SELECT * FROM events"
    clauses: list[str] = []
    params: list[object] = []
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    if patient_id is not None:
        clauses.append("patient_id = ?")
        params.append(patient_id)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY start DESC"

    with get_connection() as conn:
        return [_event_from_row(row) for row in conn.execute(query, params).fetchall()]


def get_event(event_id: str) -> dict | None:
    """Return one event by ID, or None when it does not exist."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM events WHERE event_id = ?", (event_id,)
        ).fetchone()
    return _event_from_row(row) if row is not None else None


def purge_consumed_closed_events(cutoff: datetime | str) -> int:
    """Delete only durable, successfully consumed events older than cutoff."""
    cutoff_value = cutoff.isoformat() if isinstance(cutoff, datetime) else cutoff
    with get_connection() as conn:
        result = conn.execute(
            "DELETE FROM events WHERE status='closed' AND consumed=1 AND closed_at < ?",
            (cutoff_value,),
        )
    return result.rowcount


def replace_speaker_identity(old_identity: str, new_name: str) -> int:
    """Rename a confirmed speaker in historical event JSON, preserving provenance."""
    changed_events = 0
    with get_connection() as conn:
        rows = conn.execute("SELECT event_id, participants, segments FROM events").fetchall()
        for row in rows:
            participants = json.loads(row["participants"])
            segments = json.loads(row["segments"])
            changed = False
            renamed_participants = []
            for participant in participants:
                value = new_name if participant == old_identity else participant
                if value not in renamed_participants:
                    renamed_participants.append(value)
                changed = changed or value != participant
            for segment in segments:
                if segment.get("speaker") == old_identity:
                    segment.setdefault("original_speaker_identity", old_identity)
                    segment["speaker"] = new_name
                    changed = True
                if segment.get("speaker_identity") == old_identity:
                    segment.setdefault("original_speaker_identity", old_identity)
                    segment["speaker_identity"] = new_name
                    changed = True
            if changed:
                conn.execute(
                    "UPDATE events SET participants=?, segments=? WHERE event_id=?",
                    (json.dumps(renamed_participants), json.dumps(segments), row["event_id"]),
                )
                changed_events += 1
    return changed_events
