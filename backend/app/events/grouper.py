import json
import uuid
import logging
from datetime import datetime, timezone
from dataclasses import dataclass
from app.events.db import get_connection, init_db

LOGGER = logging.getLogger(__name__)

# Ensure DB and tables exist on module load
init_db()

GAP_THRESHOLD_SECONDS = 90.0

@dataclass
class GrouperSegment:
    parent_recording_id: str
    segment_id: str
    speaker: str
    start: str
    end: str
    text: str

def parse_iso(iso: str) -> datetime:
    """Parse an ISO 8601 string safely into a datetime object."""
    return datetime.fromisoformat(iso)

def now_iso() -> str:
    """Return current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()

def generate_event_id() -> str:
    return str(uuid.uuid4())

def get_unconsumed_event_count() -> int:
    """Read the unconsumed_event_count for the separate LLM scheduler."""
    with get_connection() as conn:
        row = conn.execute("SELECT value FROM grouping_meta WHERE key = 'unconsumed_event_count'").fetchone()
        return int(row["value"]) if row else 0

def increment_unconsumed_event_count(conn) -> None:
    """Increment unconsumed_event_count safely within an existing transaction."""
    conn.execute("""
        UPDATE grouping_meta 
        SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)
        WHERE key = 'unconsumed_event_count'
    """)

def create_new_event(conn, segment: GrouperSegment) -> None:
    event_id = generate_event_id()
    segments_json = json.dumps([{
        "speaker": segment.speaker,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text
    }])
    participants_json = json.dumps([segment.speaker])
    recordings_json = json.dumps([segment.parent_recording_id])

    conn.execute("""
        INSERT INTO events (
            event_id, status, participants, start, end, location, recordings, segments, closed_at, llm_status
        ) VALUES (?, 'open', ?, ?, ?, NULL, ?, ?, NULL, 'not_eligible')
    """, (
        event_id, participants_json, segment.start, segment.end,
        recordings_json, segments_json
    ))
    LOGGER.info("[grouper] Created new event %s starting with segment %s", event_id, segment.segment_id)

def extend_event(conn, open_event: dict, segment: GrouperSegment) -> None:
    participants = json.loads(open_event["participants"])
    if segment.speaker not in participants:
        participants.append(segment.speaker)

    segments = json.loads(open_event["segments"])
    segments.append({
        "speaker": segment.speaker,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text
    })
    # Offline recording jobs can finish out of order. Keep the event's time
    # range correct instead of assuming that this is the newest segment.
    segments.sort(key=lambda item: parse_iso(item["start"]))

    recordings = json.loads(open_event["recordings"])
    if segment.parent_recording_id not in recordings:
        recordings.append(segment.parent_recording_id)

    conn.execute("""
        UPDATE events 
        SET participants = ?,
            start = ?,
            end = ?,
            segments = ?,
            recordings = ?
        WHERE event_id = ?
    """, (
        json.dumps(participants),
        min(parse_iso(open_event["start"]), parse_iso(segment.start)).isoformat(),
        max(parse_iso(open_event["end"]), parse_iso(segment.end)).isoformat(),
        json.dumps(segments),
        json.dumps(recordings),
        open_event["event_id"]
    ))
    LOGGER.info("[grouper] Extended open event %s with segment %s", open_event["event_id"], segment.segment_id)

def close_event(conn, open_event: dict) -> None:
    conn.execute("""
        UPDATE events 
        SET status = 'closed',
            closed_at = ?,
            llm_status = 'pending'
        WHERE event_id = ?
    """, (now_iso(), open_event["event_id"]))
    increment_unconsumed_event_count(conn)
    LOGGER.info("[grouper] Closed event %s (gap threshold exceeded)", open_event["event_id"])

def process_segment(segment: GrouperSegment) -> None:
    """
    Main entry point. Call this per segment immediately after finalization.
    """
    with get_connection() as conn:
        # We start an exclusive transaction to prevent concurrent processes from modifying the open event.
        conn.execute("BEGIN EXCLUSIVE")
        try:
            row = conn.execute("SELECT * FROM events WHERE status = 'open'").fetchone()
            
            if row is None:
                create_new_event(conn, segment)
            else:
                open_event = dict(row)
                gap = (parse_iso(segment.start) - parse_iso(open_event["end"])).total_seconds()
                
                if gap > GAP_THRESHOLD_SECONDS:
                    close_event(conn, open_event)
                    create_new_event(conn, segment)
                else:
                    extend_event(conn, open_event, segment)
            
            conn.commit()
        except Exception as exc:
            conn.rollback()
            LOGGER.error("[grouper] Failed to process segment %s: %s", segment.segment_id, exc)
            raise
