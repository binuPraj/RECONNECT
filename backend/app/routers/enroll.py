"""HTTP enrollment endpoint — reuses the same building blocks as
enroll_user.py / enroll_audio.py, so behavior stays identical to the
core flow (duplicate-face short-circuit, 3-sample voice concatenation, etc.)
"""

import asyncio
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response

from database.db import (
    append_voice_embedding,
    caregiver_owns_patient,
    create_identity,
    delete_identity,
    find_matching_face,
    get_unenrolled_identity,
    get_patient_name,
    get_patient_enrolled_people,
    get_patient_identity_image,
    get_pending_unknown_image,
    resolve_pending_unknown,
    resolve_unenrolled_identity,
    blob_to_embedding,
    voice_blob_to_embeddings,
)
from app.routers.auth import current_actor, require_caregiver
# Do not import ``enrollment.enroll`` here just for this path.  That legacy
# module imports OpenCV at module load time, which loads its bundled FFmpeg
# libraries into the long-running audio server.  On macOS those clash with
# Homebrew FFmpeg and can terminate the whole process during audio processing.
ENROLLMENT_IMAGE_PATH = str(
    Path(__file__).resolve().parents[2] / "uploads" / "enrollment_images"
)
from enroll_audio import create_voice_embedding, ENROLLMENT_AUDIO_PATH
from app.audio_process.preprocessing import preprocess_audio
from app.identity_resolution import reconcile_confirmed_identity

router = APIRouter(prefix="/enroll", tags=["enrollment"])
LOGGER = logging.getLogger(__name__)

_face_engine = None


def _get_face_engine():
    global _face_engine
    if _face_engine is None:
        from vision.face_engine import FaceEngine
        _face_engine = FaceEngine()
    return _face_engine


def _decode_image(raw_bytes: bytes):
    import cv2

    array = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=422, detail="Could not decode one of the uploaded photos.")
    return image


async def _sync_caregiver_relationship(
    request: Request, *, patient_id: int, relative_name: str, relation: str, enrolled_identity_id: int
) -> bool:
    """Mirror a completed enrollment into Neo4j when graph memory is active."""
    graph_store = getattr(request.app.state, "graph_store", None)
    patient_name = get_patient_name(patient_id)
    if graph_store is None or not patient_name:
        return False
    try:
        await asyncio.to_thread(
            graph_store.upsert_caregiver_relationship,
            relative_name=relative_name.strip(),
            relation=relation.strip(),
            patient_name=patient_name,
            enrolled_identity_id=enrolled_identity_id,
            patient_id=patient_id,
        )
        return True
    except Exception:
        LOGGER.exception(
            "[enrollment] Neo4j relationship sync failed for %s and patient %s",
            relative_name.strip(), patient_id,
        )
        return False


async def _resolve_confirmed_unknown(
    request: Request, *, unenrolled_id: int | None, enrolled_id: int, enrolled_name: str
) -> dict | None:
    """Resolve the unknown identity row, then reconcile all derived history."""
    if unenrolled_id is None:
        return None
    resolve_unenrolled_identity(unenrolled_id, enrolled_id, source="caregiver")
    try:
        return await asyncio.to_thread(
            reconcile_confirmed_identity,
            unenrolled_id=unenrolled_id,
            enrolled_name=enrolled_name,
            graph_store=getattr(request.app.state, "graph_store", None),
            vector_store=getattr(request.app.state, "vector_store", None),
        )
    except Exception:
        # The SQLite resolution is durable. A later application startup can
        # backfill projections; do not roll back caregiver confirmation.
        LOGGER.exception("[enrollment] Historical identity reconciliation failed for unenrolled_%s", unenrolled_id)
        return {"pending_retry": True}


@router.post("/person")
async def enroll_person(
    request: Request,
    patient_id: int = Form(...),
    name: str = Form(...),
    relation: str = Form(...),
    photos: list[UploadFile] = File(default=[]),
    voice_samples: list[UploadFile] = File(default=[]),
    pending_unknown_id: Optional[int] = Form(default=None),
    unenrolled_id: Optional[int] = Form(default=None),
    caregiver=Depends(require_caregiver),
):
    if not name.strip() or not relation.strip():
        raise HTTPException(status_code=422, detail="Name and relation are required.")
    if not caregiver_owns_patient(caregiver["actor_id"], patient_id):
        raise HTTPException(status_code=403, detail="That patient is not linked to this caregiver.")

    best_face = None
    best_area = -1
    best_image = None

    # A caregiver resolving an already bound unknown profile may reuse its
    # captured face/voice profile; only name and relation need re-entering.
    captured_unknown = get_unenrolled_identity(unenrolled_id) if unenrolled_id is not None else None
    if not photos and captured_unknown is not None:
        if captured_unknown.get("patient_id") not in (None, patient_id):
            raise HTTPException(status_code=403, detail="That unknown profile belongs to another patient.")
        if captured_unknown.get("face_embedding") is None:
            raise HTTPException(
                status_code=422,
                detail="This unknown profile has no captured face yet. Add a photo before enrolling.",
            )
        embedding = blob_to_embedding(captured_unknown["face_embedding"])
        best_image = captured_unknown.get("face_image")
    elif not photos and pending_unknown_id is not None:
        cv_img = get_pending_unknown_image(pending_unknown_id, patient_id, as_cv2=True)
        if cv_img is None:
            raise HTTPException(status_code=404, detail="Pending unknown image not found.")
        face_engine = _get_face_engine()
        faces = face_engine.detect_faces(cv_img)
        usable = [f for f in faces if f.get("embedding") is not None]
        for face in usable:
            x1, y1, x2, y2 = face["bbox"]
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_face = face
                best_image = cv_img
        if best_face is None:
            raise HTTPException(
                status_code=422,
                detail="No usable face found in the pending unknown image. Add a photo before enrolling.",
            )
        embedding = best_face["embedding"]
    else:
        face_engine = _get_face_engine()
        # Find the best usable face across submitted photos.
        for photo in photos:
            raw = await photo.read()
            image = _decode_image(raw)
            faces = face_engine.detect_faces(image)
            usable = [f for f in faces if f.get("embedding") is not None]
            for face in usable:
                x1, y1, x2, y2 = face["bbox"]
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    best_face = face
                    best_image = image
        if best_face is None:
            raise HTTPException(
                status_code=422,
                detail="Add a photo, or select an unknown voice with a captured face.",
            )
        embedding = best_face["embedding"]

    # 2. Duplicate check
    existing, similarity = find_matching_face(embedding, patient_id=patient_id)
    if existing is not None:
        resolved_unknown = await _resolve_confirmed_unknown(
            request,
            unenrolled_id=unenrolled_id or pending_unknown_id,
            enrolled_id=existing["id"],
            enrolled_name=existing["name"],
        )
        graph_synced = await _sync_caregiver_relationship(
            request, patient_id=patient_id, relative_name=existing["name"], relation=relation,
            enrolled_identity_id=existing["id"],
        )
        return {
            "status": "duplicate",
            "identity_id": existing["id"],
            "name": existing["name"],
            "similarity": similarity,
            "graph_relationship_synced": graph_synced,
            "identity_resolution": resolved_unknown,
        }

    # 3. New identity — create SQLite row + save the chosen photo to disk
    identity_id = create_identity(
        name=name.strip(),
        relation=relation.strip(),
        face_embedding=embedding,
        face_image=best_image,
        patient_id=patient_id,
    )

    image_dir = Path(ENROLLMENT_IMAGE_PATH) / str(identity_id)
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / (datetime.now().strftime("%Y%m%d_%H%M%S.jpg"))
    if isinstance(best_image, bytes):
        image_path.write_bytes(best_image)
    elif best_image is not None:
        import cv2

        cv2.imwrite(str(image_path), best_image)

    # 4. Voice processing (if voice samples were provided)
    audio_dir = Path(ENROLLMENT_AUDIO_PATH) / str(identity_id)
    audio_dir.mkdir(parents=True, exist_ok=True)
    if voice_samples:
        try:
            for index, sample in enumerate(voice_samples, start=1):
                raw_path = audio_dir / f"enrollment_voice_{index:02d}.wav"
                cleaned_path = audio_dir / f"enrollment_voice_{index:02d}_cleaned.wav"

                raw_bytes = await sample.read()
                if not raw_bytes:
                    continue
                raw_path.write_bytes(raw_bytes)
                preprocess_audio(raw_path, cleaned_path)
                voice_embedding = create_voice_embedding(cleaned_path)
                append_voice_embedding(identity_id, voice_embedding)
        except Exception as exc:
            delete_identity(identity_id)
            shutil.rmtree(image_dir, ignore_errors=True)
            shutil.rmtree(audio_dir, ignore_errors=True)
            raise HTTPException(status_code=422, detail=f"Could not process enrollment audio: {exc}") from exc
    elif captured_unknown is not None and captured_unknown.get("voice_embedding"):
        # Carry over the captured unknown voice embeddings so they are preserved
        stored_voice = captured_unknown["voice_embedding"]
        embeddings = voice_blob_to_embeddings(stored_voice)
        for emb in embeddings:
            append_voice_embedding(identity_id, emb)

    if pending_unknown_id is not None:
        try:
            resolve_pending_unknown(pending_unknown_id, patient_id, identity_id)
        except Exception:
            pass

    resolved_unknown = await _resolve_confirmed_unknown(
        request,
        unenrolled_id=unenrolled_id or pending_unknown_id,
        enrolled_id=identity_id,
        enrolled_name=name.strip(),
    )

    # Neo4j is an optional projection, so an unavailable graph must never undo
    # a completed face/voice enrollment in the primary SQLite database.
    graph_synced = await _sync_caregiver_relationship(
        request, patient_id=patient_id, relative_name=name, relation=relation,
        enrolled_identity_id=identity_id,
    )

    return {
        "status": "enrolled",
        "identity_id": identity_id,
        "patient_id": patient_id,
        "name": name,
        "relation": relation,
        "photos_submitted": len(photos),
        "voice_samples_processed": len(voice_samples),
        "graph_relationship_synced": graph_synced,
        "identity_resolution": resolved_unknown,
        "photo_url": f"/who-is-this/people/{identity_id}/photo?patient_id={patient_id}",
    }


@router.get("/patients/{patient_id}/people")
def enrolled_people(patient_id: int, actor=Depends(current_actor)):
    allowed = (actor["actor_type"] == "patient" and actor["actor_id"] == patient_id) or (
        actor["actor_type"] == "caregiver" and caregiver_owns_patient(actor["actor_id"], patient_id)
    )
    if not allowed:
        raise HTTPException(status_code=403, detail="That patient is not linked to this actor.")
    people = get_patient_enrolled_people(patient_id)
    for person in people:
        person["photo_url"] = f"/who-is-this/people/{person['id']}/photo?patient_id={patient_id}"
    return {"people": people}


@router.get("/people/{identity_id}/photo")
def enrolled_photo_direct(identity_id: int, actor=Depends(current_actor)):
    image = get_patient_identity_image(identity_id)
    if not image:
        raise HTTPException(status_code=404, detail="Enrollment photo not found.")
    return Response(content=image, media_type="image/jpeg")
