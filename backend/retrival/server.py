"""Standalone retrieval API and debug browser. Does not mount into the audio server."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from retrival.config import RETRIVAL_PORT, SESSION_GAP_SECONDS
from retrival.debug_log import init_debug_db, list_llm_calls, list_queries
from retrival.llm import RetrievalLLM
from retrival.pipeline import run_query
from retrival.sessions import public_session
from retrival.transcribe import transcribe_question_audio
from retrival.vision import cleanup_upload, match_faces, save_upload

LOGGER = logging.getLogger("retrival")
STATIC_DIR = Path(__file__).resolve().parent / "static"


def _load_env() -> None:
    for path in (BACKEND_ROOT / ".env", BACKEND_ROOT.parent / ".env"):
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))



def build_context() -> dict:
    _load_env()
    from app.memory.store import SQLiteMemoryStore

    memory_store = SQLiteMemoryStore()
    graph_store = None
    try:
        from app.memory.graph_store import GraphStore, VectorStore
        graph_store = GraphStore.from_environment()
    except Exception:
        LOGGER.exception("Neo4j unavailable for retrieval")
        graph_store = None
    # Loading the sentence-transformer model can take a while (and may try to
    # reach Hugging Face).  It is only needed by the open-ended ``other``
    # route, so do not hold up the phone's typed query screen at server start.
    # The first other-intent request loads it; all structured retrieval works
    # immediately from SQLite/Neo4j.
    vector_store = None

    def load_vector_store():
        nonlocal vector_store
        if vector_store is not None:
            return vector_store
        try:
            from app.memory.graph_store import VectorStore
            vector_store = VectorStore()
        except Exception:
            LOGGER.exception("FAISS unavailable for retrieval; using SQLite ranking for this query")
            vector_store = None
        return vector_store

    return {
        "memory_store": memory_store,
        "graph_store": graph_store,
        "vector_store": vector_store,
        "load_vector_store": load_vector_store,
        "llm": RetrievalLLM(),
    }


app = FastAPI(title="RECONNECT retrieval")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.state.context = None


def initialize_for_host(context: dict | None = None) -> None:
    """Prepare retrieval when standalone or hosted by ``app.main``.

    The normal backend hosts this FastAPI app at ``/retrieval``.  Passing its
    existing stores avoids opening a second SQLite/Neo4j/LLM setup.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    init_debug_db()
    app.state.context = context or build_context()
    LOGGER.info("Retrieval ready; session gap %ss", SESSION_GAP_SECONDS)


@app.on_event("startup")
def startup() -> None:
    initialize_for_host()


def _context() -> dict:
    if app.state.context is None:
        app.state.context = build_context()
    return app.state.context


@app.get("/", response_class=HTMLResponse)
def debug_home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    ctx = _context()
    return {
        "ok": True,
        "session_gap_seconds": SESSION_GAP_SECONDS,
        "graph": ctx.get("graph_store") is not None,
        "vector": ctx.get("vector_store") is not None,
        "vector_mode": "ready" if ctx.get("vector_store") is not None else "lazy",
        "llm": ctx["llm"].client is not None,
    }


@app.post("/query")
async def query(payload: dict):
    patient_id = int(payload.get("patient_id") or 0)
    question = str(payload.get("question") or "").strip()
    if not patient_id or not question:
        return JSONResponse({"detail": "patient_id and question are required"}, status_code=422)
    face_result = payload.get("face_result")
    result = run_query(patient_id=patient_id, question=question, context=_context(), face_result=face_result)
    return result


@app.post("/transcribe")
async def transcribe(patient_id: int = Form(...), audio: UploadFile = File(...)):
    data = await audio.read()
    if not data:
        return JSONResponse({"detail": "audio was empty"}, status_code=422)
    text = transcribe_question_audio(data, audio.filename or "question.wav")
    return {"patient_id": patient_id, "transcript": text}


@app.post("/ask")
async def ask(
    patient_id: int = Form(...),
    question: str = Form(""),
    audio: UploadFile | None = File(None),
):
    transcript = question.strip()
    if audio is not None:
        data = await audio.read()
        if data:
            transcript = transcribe_question_audio(data, audio.filename or "question.wav") or transcript
    if not transcript:
        return JSONResponse({"detail": "Could not transcribe a question."}, status_code=422)
    return run_query(patient_id=patient_id, question=transcript, context=_context())


@app.post("/who-is")
async def who_is(
    patient_id: int = Form(...),
    question: str = Form("who is this"),
    video: UploadFile = File(...),
):
    data = await video.read()
    if not data:
        return JSONResponse({"detail": "video was empty"}, status_code=422)
    path = save_upload(data)
    try:
        face_result = match_faces(path, patient_id, _context().get("graph_store"))
    finally:
        cleanup_upload(path)
    return run_query(
        patient_id=patient_id,
        question=question or "who is this",
        context=_context(),
        face_result=face_result,
    )


@app.get("/debug")
def debug(limit: int = 50, patient_id: int | None = None):
    from retrival.sessions import _active, list_all_sessions

    current = _active(patient_id) if patient_id else None
    return {
        "health": health(),
        "llm_calls": list_llm_calls(limit),
        "queries": list_queries(limit),
        "sessions": list_all_sessions(patient_id=patient_id, limit=limit),
        "session_gap_seconds": SESSION_GAP_SECONDS,
        "active_hint": public_session(current) if current else None,
    }


@app.get("/sessions")
def sessions_list(patient_id: int | None = None, limit: int = 50):
    from retrival.sessions import list_all_sessions
    return {"sessions": list_all_sessions(patient_id=patient_id, limit=limit)}


@app.post("/sessions/{session_id}/close")
def close_session_endpoint(session_id: str):
    from retrival.sessions import _connect, now_iso
    with _connect() as conn:
        conn.execute("UPDATE sessions SET closed_at = ? WHERE session_id = ?", (now_iso(), session_id))
    return {"ok": True, "session_id": session_id}


@app.get("/session/{patient_id}")
def session_view(patient_id: int):
    from retrival.sessions import _active
    current = _active(patient_id)
    return {"session": public_session(current) if current else None, "gap_seconds": SESSION_GAP_SECONDS}


@app.post("/clear-all")
def clear_all_data():
    from retrival.debug_log import clear_all_debug_data
    from retrival.sessions import clear_all_sessions
    clear_all_debug_data()
    clear_all_sessions()
    return {"ok": True, "message": "All queries, LLM calls, and sessions have been cleared."}



if __name__ == "__main__":
    import uvicorn
    _load_env()
    uvicorn.run("retrival.server:app", host="0.0.0.0", port=RETRIVAL_PORT, reload=False)
