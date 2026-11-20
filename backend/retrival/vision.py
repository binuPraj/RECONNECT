"""Face match for who_is. Imports the existing vision pipeline without changing it."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from retrival.config import NO_FACE, SEVERAL_FACES, UNRECOGNISED


def match_faces(video_path: Path, patient_id: int, graph_store=None) -> dict:
    from config import CAPTURE_SECONDS, TARGET_FPS
    from database.db import get_patient_identity_by_name

    try:
        from vision.pipeline import WhoIsThisPipeline
    except ImportError:
        return {"status": "error", "message": "Face recognition is not available on this server."}

    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return {"status": "no_face", "message": NO_FACE}
    frames = []
    frame_interval = 1.0 / TARGET_FPS
    next_timestamp = 0.0
    capture_limit = 5.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if timestamp > capture_limit:
                break
            if timestamp + 0.001 >= next_timestamp:
                frames.append((timestamp, frame))
                next_timestamp += frame_interval
    finally:
        capture.release()
    if not frames:
        return {"status": "no_face", "message": NO_FACE}

    result = WhoIsThisPipeline().process_frames(frames, capture_seconds=capture_limit)
    people = result.get("people") or []
    if len(people) > 1:
        return {"status": "several_faces", "message": SEVERAL_FACES, "people": people}
    if not people:
        return {"status": "no_face", "message": NO_FACE}

    person = people[0]
    status = person.get("status")
    if status == "known" and person.get("identity"):
        identity = get_patient_identity_by_name(person["identity"], patient_id)
        if identity:
            payload = {
                "status": "known",
                "identity_id": identity["id"],
                "name": identity["name"],
                "relation": identity["relation"],
                "person_id": f"patient:{patient_id}:enrolled:{identity['id']}",
            }
            if graph_store is not None:
                try:
                    payload["graph_context"] = graph_store.get_person_by_id(payload["person_id"])
                except Exception:
                    payload["graph_context"] = None
            return payload
        return {"status": "unrecognised", "message": UNRECOGNISED}
    if status in {"new_unknown", "existing_unknown"}:
        return {"status": "unrecognised", "message": UNRECOGNISED}
    return {"status": "no_face", "message": NO_FACE}


def save_upload(upload) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="retrival_who_"))
    path = temp_dir / "capture.mp4"
    path.write_bytes(upload)
    return path


def cleanup_upload(path: Path) -> None:
    shutil.rmtree(path.parent, ignore_errors=True)
