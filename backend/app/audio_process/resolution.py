import os
import json
import logging
import numpy as np
from pathlib import Path
from app.utils.storage import STREAMS_FOLDER

logger = logging.getLogger(__name__)

def backfill_unenrolled_transcripts(unenrolled_id: int, enrolled_name: str):
    """
    Update historical transcript segments tagged with unenrolled_{id} to the new enrolled_name.
    Touches STREAMS_FOLDER/*/transcript.json and STREAMS_FOLDER/*/recording_*/segments/segments.json.
    """
    target_ident = f"unenrolled_{unenrolled_id}"
    
    # Iterate all sessions
    for session_dir in STREAMS_FOLDER.iterdir():
        if not session_dir.is_dir():
            continue
            
        # Update session-level transcript.json
        transcript_path = session_dir / "transcript.json"
        if transcript_path.exists():
            try:
                with open(transcript_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                changed = False
                for seg in data.get("segments", []):
                    if seg.get("speaker_identity") == target_ident:
                        seg["speaker_identity"] = enrolled_name
                        seg["speaker"] = enrolled_name
                        changed = True
                        
                if changed:
                    with open(transcript_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, sort_keys=True)
                    logger.info("Backfilled %s in %s", target_ident, transcript_path)
            except Exception as e:
                logger.error("Failed to backfill %s: %s", transcript_path, e)
                
        # Update recording-level segments.json
        for recording_dir in session_dir.iterdir():
            if not recording_dir.is_dir() or not recording_dir.name.startswith("recording_"):
                continue
                
            segments_json_path = recording_dir / "segments" / "segments.json"
            if segments_json_path.exists():
                try:
                    with open(segments_json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    changed = False
                    for seg in data.get("segments", []):
                        if seg.get("speaker_identity") == target_ident:
                            seg["speaker_identity"] = enrolled_name
                            seg["speaker"] = enrolled_name
                            changed = True
                            
                    if changed:
                        with open(segments_json_path, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2, sort_keys=True)
                except Exception as e:
                    logger.error("Failed to backfill %s: %s", segments_json_path, e)


def cascade_resolve_unenrolled(enrolled_id: int, enrolled_name: str, new_embedding, session_id: str, max_cascades: int = 5):
    """
    Re-score other unresolved unenrolled_* rows from the same session against the newly added embedding.
    If > 0.62, resolve them too. Cap at max_cascades per trigger.
    """
    from database.db import resolve_unenrolled_identity, voice_blob_to_embeddings, init_database, _connect, _DB_LOCK
    
    init_database()
    
    target_emb = np.asarray(new_embedding, dtype=np.float32).reshape(-1)
    target_norm = np.linalg.norm(target_emb)
    if target_norm == 0:
        return
        
    resolved_count = 0
    
    with _DB_LOCK, _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM unenrolled_identities WHERE resolved_to IS NULL"
        ).fetchall()
        
    # We copy them to avoid DB locking issues when iterating and updating
    for row in rows:
        if resolved_count >= max_cascades:
            break
            
        un_id = row["id"]
        voice_blob = row["voice_embedding"]
        if not voice_blob:
            continue
            
        is_match = False
        for stored_emb in voice_blob_to_embeddings(voice_blob):
            stored = np.asarray(stored_emb, dtype=np.float32).reshape(-1)
            stored_norm = np.linalg.norm(stored)
            if stored.shape != target_emb.shape or stored_norm == 0:
                continue
            
            sim = float(np.dot(target_emb, stored) / (target_norm * stored_norm))
            if sim > 0.62:
                is_match = True
                break
                
        if is_match:
            logger.info("Cascade resolving unenrolled_%d to %s (score > 0.62)", un_id, enrolled_name)
            resolve_unenrolled_identity(un_id, enrolled_id)
            backfill_unenrolled_transcripts(un_id, enrolled_name)
            resolved_count += 1
