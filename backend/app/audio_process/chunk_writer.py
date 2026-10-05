"""
chunk_writer.py
---------------
Transforms a recording's segments.json into chunk.json.

chunk.json is a clean, wall-clock-anchored representation of a recording:
  - All timestamps are absolute ISO 8601 (recorded_at + duration arithmetic).
  - File-relative fields (start_sec, end_sec, audio_path, etc.) are dropped.
  - participants is deduplicated by first-appearance order.

Called immediately after segments.json is written by the pipeline.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

LOGGER = logging.getLogger(__name__)

# Tolerance: warn if a segment's start is this many seconds BEFORE the
# previous segment's computed end (negative gap means overlap / bug).
_OVERLAP_WARN_THRESHOLD_SEC = -0.5


def _parse_dt(iso: str) -> datetime:
    """Parse an ISO 8601 string, preserving the original timezone offset."""
    # Python 3.10 fromisoformat handles offsets like +05:45 directly.
    return datetime.fromisoformat(iso)


def _add_seconds(dt: datetime, seconds: float) -> datetime:
    """Return dt + seconds as a datetime, keeping the original tzinfo."""
    return dt + timedelta(seconds=seconds)


def _format_dt(dt: datetime) -> str:
    """
    Serialise back to ISO 8601, preserving the original UTC offset exactly
    (e.g. +05:45) — never converts to UTC or strips the offset.
    """
    return dt.isoformat()


def _recording_id_from_path(recording_dir: Path) -> int:
    """
    Extract integer recording ID from folder name.
    e.g. 'recording_0004' -> 4, 'recording_0001' -> 1
    Falls back to 0 if the pattern doesn't match.
    """
    match = re.search(r"(\d+)$", recording_dir.name)
    return int(match.group(1)) if match else 0


def build_chunk(segments_path: Path) -> dict:
    """
    Read segments.json and produce the chunk dict (not yet written to disk).

    Raises:
        FileNotFoundError: if segments_path does not exist.
        ValueError: if the segments array is empty.
    """
    raw = json.loads(segments_path.read_text(encoding="utf-8"))
    segs = raw.get("segments", [])

    if not segs:
        raise ValueError(f"segments.json at {segments_path} contains no segments")

    recording_id = raw.get("recording_id")
    if not recording_id:
        recording_dir = segments_path.parent.parent   # …/recording_0004/segments/.. → recording_0004
        recording_id = _recording_id_from_path(recording_dir)

    # ── Build output segments + sanity-check timestamps ──────────────────────
    out_segments: list[dict] = []
    seen_speakers: list[str] = []       # preserves first-appearance order
    prev_end_dt: datetime | None = None

    for i, seg in enumerate(segs):
        recorded_at_str: str = seg["recorded_at"]
        duration_sec: float = float(seg["duration_sec"])
        speaker: str = seg.get("speaker_identity") or seg.get("speaker") or "unknown"

        start_dt = _parse_dt(recorded_at_str)
        end_dt = _add_seconds(start_dt, duration_sec)

        # ── Sanity check: overlapping / out-of-order timestamps ──────────────
        if prev_end_dt is not None:
            gap_sec = (start_dt - prev_end_dt).total_seconds()
            if gap_sec < _OVERLAP_WARN_THRESHOLD_SEC:
                LOGGER.warning(
                    "[chunk_writer] Timestamp anomaly in %s — seg %s starts %.2fs "
                    "before previous segment ended (gap=%.2fs). "
                    "This may indicate a recorded_at anchor bug (e.g. stream_start_sec "
                    "incorrectly applied to the first segment).",
                    segments_path,
                    seg.get("segment_id", f"index_{i}"),
                    -gap_sec,
                    gap_sec,
                )

        prev_end_dt = end_dt

        # ── Deduplicate participants by first appearance ───────────────────────
        if speaker not in seen_speakers:
            seen_speakers.append(speaker)

        out_segments.append(
            {
                "segment_id": seg["segment_id"],
                "speaker": speaker,
                "start": recorded_at_str,           # copied as-is, no arithmetic
                "end": _format_dt(end_dt),
                "text": seg.get("transcript", ""),
            }
        )

    # recording_start = first segment's recorded_at, unchanged
    recording_start: str = segs[0]["recorded_at"]

    # recording_end = last segment's computed end
    last_seg = segs[-1]
    last_start_dt = _parse_dt(last_seg["recorded_at"])
    last_end_dt = _add_seconds(last_start_dt, float(last_seg["duration_sec"]))
    recording_end: str = _format_dt(last_end_dt)

    return {
        "recording_id": recording_id,
        "recording_start": recording_start,
        "recording_end": recording_end,
        "participants": seen_speakers,
        "segments": out_segments,
    }


def write_chunk(segments_path: Path) -> Path:
    """
    Read segments.json, build chunk dict, write chunk.json next to segments.json.

    Returns the path to the written chunk.json.
    """
    chunk_path = segments_path.parent / "chunk.json"

    try:
        chunk = build_chunk(segments_path)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        LOGGER.error("[chunk_writer] Failed to build chunk for %s: %s", segments_path, exc)
        raise

    chunk_path.write_text(
        json.dumps(chunk, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    LOGGER.info("[chunk_writer] chunk.json written → %s", chunk_path)
    return chunk_path
