"""RECONNECT live audio WebSocket API."""

import json
import logging
import os
import sys

# Prevent multiprocessing & OpenMP crashes on macOS Apple Silicon
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import time
import asyncio
import html
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.audio_stream.format_adapter import StreamAudioFormat, adapt_pcm16_stream_chunk
from app.audio_stream.segment_processor import StreamingSegmentProcessor
from app.audio_stream.streaming_recorder import StreamingSpeechRecorder
from app.events.db import DB_PATH, get_connection, get_event, init_db, list_events
from app.memory.consolidation import MemoryConsolidator, make_batch_prompt, render_transcript_text
from app.memory.graph_store import GraphStore, GraphVectorMemoryStore, VectorStore, get_direct, get_open_ended
from app.memory.openrouter import OpenRouterError, OpenRouterMemoryClient
from app.memory.store import SQLiteMemoryStore
from app.utils.storage import allocate_stream_session_id
from database.db import get_all_identities, get_all_patients, get_resolved_unenrolled_identities, init_database
from app.routers import auth as auth_router
from app.routers import enroll as enroll_router
from app.routers import memories as memories_router
from app.routers import who_is_this as wit_router


logging.basicConfig(
    level=getattr(logging, os.getenv("RECONNECT_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

for _noisy in ("faster_whisper", "speechbrain", "pyannote", "urllib3", "httpx", "huggingface_hub", "onnxruntime"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

MEMORY_SCHEDULER_INTERVAL_SECONDS = 60
TZ = ZoneInfo("Asia/Kathmandu")


def _display_time(value: object) -> str:
    """Present legacy UTC and current Kathmandu ISO timestamps consistently."""
    if not value:
        return "—"
    try:
        return datetime.fromisoformat(str(value)).astimezone(TZ).isoformat()
    except ValueError:
        return str(value)


def _memory_time_refs_html(memory: dict) -> str:
    refs = []
    for ref in memory.get("time_refs", []):
        refs.append({
            "phrase": ref.get("phrase"),
            "what_was_mentioned": ref.get("what_was_mentioned"),
            "topic": ref.get("topic"),
            "concerned_with": ref.get("concerned_with"),
            "commitment_by": ref.get("commitment_by"),
            "tense": ref.get("tense"),
            "anchor": _display_time(ref.get("anchor")),
            "start": _display_time(ref.get("start")),
            "end": _display_time(ref.get("end")),
            "status": ref.get("status"),
            "created_at": _display_time(ref.get("created_at")),
            "last_confirmed_at": _display_time(ref.get("last_confirmed_at")),
            "superseded_at": _display_time(ref.get("superseded_at")),
        })
    return html.escape(json.dumps(refs, indent=2))


def _memory_history_html(memory: dict) -> str:
    history = [
        {**entry, "at": _display_time(entry.get("at"))}
        for entry in memory.get("update_history", [])
    ]
    return html.escape(json.dumps(history, indent=2))


def _memory_updated_events_html(memory: dict) -> str:
    updates = [
        {**entry, "event_time": _display_time(entry.get("event_time"))}
        for entry in memory.get("updated_events", [])
    ]
    return html.escape(json.dumps(updates, indent=2))


async def _run_memory_batch(app: FastAPI, force: bool = False):
    """Serialize all automatic and manual batches in this server process."""
    async with app.state.memory_batch_lock:
        if force:
            return await asyncio.to_thread(app.state.memory_consolidator.run, app.state.memory_client)
        return await asyncio.to_thread(app.state.memory_consolidator.run_if_due, app.state.memory_client)


async def _memory_scheduler(app: FastAPI) -> None:
    """Check once a minute; the consolidator enforces the 5-minute/15-event rule."""
    while True:
        try:
            results = await _run_memory_batch(app)
            if results is not None:
                LOGGER.info("[memory] Consolidated %s durable memories", len(results))
        except Exception:
            LOGGER.exception("[memory] Consolidation batch failed; events remain pending for retry")
        await asyncio.sleep(MEMORY_SCHEDULER_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_database()
    init_db()
    app.state.memory_consolidator = None
    app.state.memory_client = None
    app.state.memory_batch_lock = asyncio.Lock()
    app.state.graph_store = None
    app.state.vector_store = None
    try:
        client = OpenRouterMemoryClient()
        memory_store = SQLiteMemoryStore()
        reconciled_concerns = memory_store.reconcile_concern_entities()
        
        # Check if Neo4j and FAISS graph projection should be enabled
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_user = os.getenv("NEO4J_USER")
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        graph_enabled = os.getenv("RECONNECT_GRAPH_MEMORY", "0") == "1" or bool(neo4j_uri and neo4j_user and neo4j_password)
        
        if graph_enabled:
            graph_store = None
            try:
                graph_store = GraphStore.from_environment()
                if graph_store is None:
                    raise RuntimeError("Graph memory requires NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD")
                graph_store.ensure_schema()
                migrated_people = graph_store.migrate_person_entities()
                removed_mentions = graph_store.remove_person_mentions()
                restored_entity_mentions = graph_store.restore_entity_mentions()
                for patient in get_all_patients():
                    graph_store.upsert_patient(
                        patient_id=patient["id"], patient_name=patient["full_name"],
                    )
                for identity in get_all_identities():
                    graph_store.upsert_biometric_person(
                        name=identity["name"],
                        identity_table="enrolled_identities",
                        identity_id=identity["id"],
                        patient_id=identity["patient_id"],
                    )
                    if identity["patient_id"] is not None:
                        graph_store.reconcile_enrolled_identity_entities(
                            patient_id=identity["patient_id"],
                            enrolled_identity_id=identity["id"],
                            enrolled_name=identity["name"],
                            relation=identity["relation"],
                        )
                # If Neo4j was unavailable during enrollment, use the durable
                # SQLite resolution record to remove its stale unknown node on
                # the next successful graph startup.
                for resolution in get_resolved_unenrolled_identities():
                    graph_store.retire_unenrolled_person(
                        f"unenrolled_{resolution['unenrolled_id']}",
                        resolution["enrolled_name"],
                    )
                projected_store = GraphVectorMemoryStore(memory_store, graph_store, VectorStore())
                backfilled = projected_store.backfill()
                split_non_person_entities = graph_store.split_shared_non_person_entities()
                graph_store.backfill_display_names()
                pruned_entities = graph_store.prune_orphan_entities()
                memory_store = projected_store
                app.state.graph_store, app.state.vector_store = graph_store, projected_store.vector
                LOGGER.info("[memory] Neo4j + FAISS memory projection enabled; migrated %s legacy person entities; removed %s person mentions; restored %s entity mentions; reconciled %s concern records; backfilled %s SQLite memories; split %s shared non-person entities; pruned %s orphan entities", migrated_people, removed_mentions, restored_entity_mentions, reconciled_concerns, backfilled, split_non_person_entities, pruned_entities)
            except Exception:
                # Graph/vector retrieval is optional: retain normal SQLite consolidation.
                LOGGER.exception("[memory] Neo4j + FAISS projection disabled; SQLite remains active")
                if graph_store is not None:
                    graph_store.close()

        consolidator = MemoryConsolidator(memory_store)
        app.state.memory_store = memory_store
        app.state.memory_consolidator = consolidator
        app.state.memory_client = client
        app.state.memory_task = asyncio.create_task(_memory_scheduler(app))
        LOGGER.info("[memory] OpenRouter consolidation scheduler started")
    except (OpenRouterError, RuntimeError) as error:
        app.state.memory_task = None
        LOGGER.warning("[memory] Scheduler disabled: %s", error)
    try:
        yield
    finally:
        task = app.state.memory_task
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        graph_store = getattr(app.state, "graph_store", None)
        if graph_store is not None:
            graph_store.close()


app = FastAPI(title="RECONNECT Audio Streaming API", version="1.2.0", lifespan=lifespan)

# Allow the Flutter app (running on the same LAN / Android device) to reach the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Flutter-facing REST routers
app.include_router(auth_router.router)
app.include_router(enroll_router.router)
app.include_router(memories_router.router)
app.include_router(wit_router.router)
app.include_router(wit_router.unknown_voice_router)


@app.get("/")
async def root():
    return {"service": "RECONNECT Audio Streaming API", "status": "running"}


@app.get("/events")
async def read_events(status: str | None = Query(default=None, pattern="^(open|closed)$")):
    """Read all events from the configured SQLite database."""
    return {"database": str(DB_PATH), "events": list_events(status=status)}


@app.get("/events/{event_id}")
async def read_event(event_id: str):
    """Read a single event by its event_id."""
    event = get_event(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return event


@app.patch("/events/{event_id}/llm-status")
async def set_event_llm_status(event_id: str, payload: dict = Body(...)):
    """Set an event's LLM queue state: pending, processed, skipped, or not_eligible."""
    llm_status = payload.get("llm_status")
    allowed = {"pending", "processed", "skipped", "not_eligible"}
    if llm_status not in allowed:
        raise HTTPException(status_code=400, detail=f"llm_status must be one of {sorted(allowed)}")
    event = get_event(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    if event["status"] != "closed" and llm_status != "not_eligible":
        raise HTTPException(status_code=400, detail="Only closed events can be queued, skipped, or marked processed")
    with get_connection() as conn:
        conn.execute("UPDATE events SET llm_status=? WHERE event_id=?", (llm_status, event_id))
    return {"event_id": event_id, "llm_status": llm_status}


@app.post("/events/{event_id}/close")
async def close_event_manually(event_id: str):
    """Manually close an open event and queue it as pending for LLM memory synthesis."""
    event = get_event(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    now_str = datetime.now(TZ).isoformat()
    with get_connection() as conn:
        conn.execute(
            "UPDATE events SET status='closed', closed_at=?, llm_status='pending' WHERE event_id=?",
            (now_str, event_id),
        )
    return {"event_id": event_id, "status": "closed", "llm_status": "pending"}


@app.get("/events/{event_id}/prompt-transcript")
async def read_prompt_transcript(event_id: str):
    """Show the flattened timestamped transcript supplied to memory prompting."""
    event = get_event(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return {"event_id": event_id, "transcript_sent_to_llm": render_transcript_text(event)}


@app.get("/memory-batches/preview")
async def preview_memory_request(event_ids: str = Query(..., description="Comma-separated event IDs")):
    """Build the request that would be sent now for selected closed events; it does not call the LLM."""
    selected = [get_event(event_id.strip()) for event_id in event_ids.split(",") if event_id.strip()]
    if not selected or any(event is None for event in selected):
        raise HTTPException(status_code=404, detail="One or more event IDs were not found")
    if any(event["status"] != "closed" for event in selected):
        raise HTTPException(status_code=400, detail="Only closed events can be previewed")
    return {"event_ids": [event["event_id"] for event in selected], "llm_request_preview": make_batch_prompt(selected, SQLiteMemoryStore())}


@app.get("/memory-batches", response_class=HTMLResponse)
async def memory_batch_audit():
    """Show complete saved LLM requests and responses for future consolidation batches."""
    try:
        with get_connection() as conn:
            batches = [dict(row) for row in conn.execute("SELECT * FROM memory_consolidation_audit ORDER BY created_at DESC")]
    except sqlite3.OperationalError:
        batches = []
    records = "".join(
        f"<details class='record'><summary><b>{html.escape(batch['status'])}</b> · {html.escape(batch['created_at'])} · {html.escape(batch['event_ids'])}</summary>"
        f"<button class='delete-audit' onclick=\"deleteAudit({html.escape(json.dumps(batch['batch_id']), quote=True)})\">Delete this saved view</button>"
        f"<h3>Request sent to OpenRouter</h3><pre>{html.escape(batch['request_prompt'])}</pre><h3>Raw LLM response</h3>"
        f"<pre>{html.escape(batch['response_text'] or '')}</pre>{('<p>Error: ' + html.escape(batch['error']) + '</p>') if batch['error'] else ''}</details>"
        for batch in batches
    ) or "<p>No audited batches yet. Requests from before this audit feature was added cannot be recovered exactly.</p>"
    return f"""<!doctype html><html><head><title>LLM batch audit</title><style>body{{font:15px system-ui,sans-serif;margin:32px;background:#f6f8fb;color:#172033}}.record{{background:white;padding:16px;border-radius:10px;box-shadow:0 1px 3px #0002;margin:12px 0}}summary{{cursor:pointer}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f8;padding:12px;border-radius:6px}}a{{color:#2455bb}}button{{padding:7px 10px;border:0;border-radius:6px;cursor:pointer}}.delete-audit,.delete-all{{margin:14px 6px 2px 0;background:#b42318;color:white}}</style></head><body><h1>LLM request audit</h1><p>Local-only diagnostic history. Deleting an entry here removes only this saved request/response view; it never changes events or memories. Automatic batching is controlled from the <a href='/dashboard'>Dashboard</a>.</p><button class='delete-all' onclick=\"deleteAllAudit()\">Delete all saved views</button>{records}<script>async function deleteAudit(id){{if(!confirm('Delete this saved request/response view? Events and memories will not be changed.'))return;let response=await fetch('/memory-batches/'+encodeURIComponent(id),{{method:'DELETE'}});if(!response.ok){{alert('Could not delete saved view.');return}}location.reload()}}async function deleteAllAudit(){{if(!confirm('Delete every saved LLM request/response view? Events and memories will not be changed.'))return;let response=await fetch('/memory-batches/all',{{method:'DELETE'}});if(!response.ok){{alert('Could not delete saved views.');return}}location.reload()}}</script></body></html>"""


@app.put("/memory-batches/automatic-pause")
async def set_automatic_memory_batching(payload: dict = Body(...)):
    """Pause or resume automatic LLM batching without changing events or memories."""
    paused = payload.get("paused")
    if not isinstance(paused, bool):
        raise HTTPException(status_code=400, detail="paused must be a boolean")
    consolidator = app.state.memory_consolidator
    if consolidator is None:
        raise HTTPException(status_code=503, detail="Memory consolidation is not configured")
    async with app.state.memory_batch_lock:
        await asyncio.to_thread(consolidator.set_automatic_batching_paused, paused)
    return {"automatic_batching_paused": paused, "events_or_memories_changed": False}


@app.delete("/memory-batches/all")
async def delete_all_memory_batch_audits():
    """Delete all LLM request/response audit records, never events or memories."""
    try:
        with get_connection() as conn:
            deleted = conn.execute("DELETE FROM memory_consolidation_audit").rowcount
    except sqlite3.OperationalError:
        deleted = 0
    return {"deleted_saved_views": deleted, "events_or_memories_changed": False}


@app.delete("/memory-batches/{batch_id}")
async def delete_memory_batch_audit(batch_id: str):
    """Delete only an LLM request/response audit record, never events or memories."""
    try:
        with get_connection() as conn:
            deleted = conn.execute(
                "DELETE FROM memory_consolidation_audit WHERE batch_id=?", (batch_id,)
            ).rowcount
    except sqlite3.OperationalError:
        deleted = 0
    if not deleted:
        raise HTTPException(status_code=404, detail="Saved LLM request/response view not found")
    return {"deleted_batch_id": batch_id, "events_or_memories_changed": False}


@app.get("/memories")
async def read_memories():
    """Read all durable memories as JSON, newest update first."""
    return {"database": str(SQLiteMemoryStore().path), "memories": SQLiteMemoryStore().list_all()}


@app.get("/memories/direct/{person_name}")
async def direct_memory_lookup(person_name: str):
    """Return graph-backed, explicitly reported third-person memories."""
    if app.state.graph_store is None:
        raise HTTPException(status_code=503, detail="Graph memory is not enabled")
    return {"memories": await asyncio.to_thread(get_direct, app.state.graph_store, person_name)}


@app.get("/memories/search")
async def semantic_memory_lookup(query: str = Query(min_length=1), k: int = Query(default=5, ge=1, le=20)):
    """Use FAISS to find IDs, then Neo4j to return their structured records."""
    if app.state.graph_store is None or app.state.vector_store is None:
        raise HTTPException(status_code=503, detail="Graph memory is not enabled")
    memories = await asyncio.to_thread(get_open_ended, query, app.state.graph_store, app.state.vector_store, k)
    return {"memories": memories}


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Small local, no-login browser view of event and memory databases."""
    events = list_events()
    memories = SQLiteMemoryStore().list_all()
    consolidator = app.state.memory_consolidator
    automatic_batching_paused = consolidator is not None and await asyncio.to_thread(consolidator.automatic_batching_paused)
    automatic_batching_control = (
        "<button class='resume-batching' onclick=\"setAutomaticBatching(false)\">Resume automatic LLM batching</button>"
        if automatic_batching_paused else
        "<button class='pause-batching' onclick=\"setAutomaticBatching(true)\">Pause automatic LLM batching</button>"
    )
    automatic_batching_message = (
        "Automatic LLM batching is paused. Closed pending events will continue accumulating but will not be sent automatically."
        if automatic_batching_paused else
        "Automatic LLM batching is active. Pending closed events are sent together after five minutes."
    )
    event_row_list = []
    for event in events:
        close_btn = f'<button class="close-btn" onclick="closeEvent(\'{html.escape(event["event_id"])}\')">Close event & queue for LLM</button>' if event["status"] == "open" else ""
        selected_pending = "selected" if event.get("llm_status") == "pending" else ""
        selected_processed = "selected" if event.get("llm_status") == "processed" else ""
        selected_skipped = "selected" if event.get("llm_status") == "skipped" else ""
        selected_not_eligible = "selected" if event.get("llm_status") == "not_eligible" else ""
        
        event_row_list.append(
            f"<details class='record'><summary><b>{html.escape(event['status'])}</b> · LLM: <b>{html.escape(event.get('llm_status', 'not_eligible'))}</b> · {html.escape(', '.join(event['participants']))} · <code>{html.escape(event['event_id'])}</code></summary>"
            f"<dl><dt>Start</dt><dd>{html.escape(event['start'])}</dd><dt>End</dt><dd>{html.escape(event['end'])}</dd>"
            f"<dt>Closed at</dt><dd>{html.escape(_display_time(event.get('closed_at')))}</dd><dt>Participants</dt><dd>{html.escape(json.dumps(event['participants']))}</dd>"
            f"<dt>LLM processing</dt><dd><select onchange=\"setLlmStatus('{html.escape(event['event_id'])}',this.value)\"><option value='pending' {selected_pending}>pending — queue for LLM</option><option value='processed' {selected_processed}>processed — do not resend</option><option value='skipped' {selected_skipped}>skipped — ignore for now</option><option value='not_eligible' {selected_not_eligible}>not eligible</option></select>"
            f"{close_btn}</dd>"
            f"<dt>Transcript sent to LLM (flattened)</dt><dd><pre>{html.escape(render_transcript_text(event))}</pre></dd>"
            f"<dt>Raw stored segments</dt><dd><pre>{html.escape(json.dumps(event['segments'], indent=2))}</pre></dd></dl>"
            f"<a href='/events/{html.escape(event['event_id'])}/prompt-transcript'>Open transcript JSON</a> · <a href='/memory-batches/preview?event_ids={html.escape(event['event_id'])}'>Preview full LLM request</a></details>"
        )
    event_rows = "".join(event_row_list) or "<p>No events yet.</p>"
    memory_rows = "".join(
        f"<details class='record'><summary><b>{html.escape(memory['category'])}</b> · {html.escape(memory['summary'])}</summary><dl>"
        f"<dt>Memory ID</dt><dd><code>{html.escape(memory['memory_id'])}</code></dd><dt>Category</dt><dd>{html.escape(memory['category'])}</dd>"
        f"<dt>Summary</dt><dd>{html.escape(memory['summary'])}</dd><dt>Participants</dt><dd>{html.escape(json.dumps(memory['participants']))}</dd>"
        f"<dt>Absent third parties / provenance</dt><dd><pre>{html.escape(json.dumps(memory.get('about', []), indent=2))}</pre></dd>"
        f"<dt>Source events</dt><dd>{html.escape(json.dumps(memory['source_events']))}</dd><dt>Earliest event time</dt><dd>{html.escape(_display_time(memory['earliest_event_time']))}</dd>"
        f"<dt>Latest event time</dt><dd>{html.escape(_display_time(memory['latest_event_time']))}</dd><dt>Importance</dt><dd>{html.escape(memory['importance'])}</dd>"
        f"<dt>Entities</dt><dd><pre>{html.escape(json.dumps(memory['entities'], indent=2))}</pre></dd><dt>Topic</dt><dd>{html.escape(str(memory.get('topic')))}</dd>"
        f"<dt>Time references</dt><dd><pre>{_memory_time_refs_html(memory)}</pre></dd><dt>Updated events</dt><dd><pre>{_memory_updated_events_html(memory)}</pre></dd>"
        f"<dt>Emotion</dt><dd>{html.escape(memory.get('emotion', 'neutral'))}</dd><dt>Created at</dt><dd>{html.escape(_display_time(memory['created_at']))}</dd>"
        f"<dt>Updated at</dt><dd>{html.escape(_display_time(memory['updated_at']))}</dd></dl></details>"
        for memory in memories
    ) or "<p>No durable memories yet.</p>"
    return f'''<!doctype html><html><head><title>RECONNECT Dashboard</title><style>
body{{font:15px system-ui,sans-serif;margin:32px;background:#f6f8fb;color:#172033}}h1,h2{{margin-bottom:8px}}.stats{{display:flex;gap:12px;margin-bottom:24px}}.card,.record{{background:white;padding:16px;border-radius:10px;box-shadow:0 1px 3px #0002;margin:12px 0}}.record summary{{cursor:pointer}}dl{{display:grid;grid-template-columns:190px 1fr;gap:8px 16px;margin-top:16px}}dt{{font-weight:650}}dd{{margin:0;overflow-wrap:anywhere}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f8;padding:12px;border-radius:6px;margin:0}}code{{font-size:12px}}a{{color:#2455bb}}button{{background:#2455bb;color:white;border:0;border-radius:6px;padding:10px 14px;cursor:pointer}}.pause-batching{{background:#9a6700}}.resume-batching{{background:#1a7f37}}.close-btn{{background:#0969da;margin-left:10px;padding:6px 10px;font-size:13px}}</style></head><body>
<h1>RECONNECT memory dashboard</h1><p>Local-only view — no login required. <a href='/events'>Events JSON</a> · <a href='/memories'>Memories JSON</a> · <a href='/memory-batches'>Saved LLM requests/responses</a> · <a href='/docs'>API docs</a></p>
<div class='stats'><div class='card'><strong>{len(events)}</strong><br>total events</div><div class='card'><strong>{sum(e['status'] == 'closed' for e in events)}</strong><br>closed events</div><div class='card'><strong>{sum(e.get('llm_status') == 'pending' for e in events)}</strong><br>pending LLM events</div><div class='card'><strong>{len(memories)}</strong><br>durable memories</div></div><p>{automatic_batching_message}</p>{automatic_batching_control} <button onclick='sendBatch()'>Send pending closed events to LLM now</button><span id='batch-result'></span>
<h2>Durable memories</h2><p>Open a memory to see every stored field.</p>{memory_rows}
<h2>Events and transcripts sent to the LLM</h2><p>Open an event to see its flattened timestamped prompt transcript and original diarization segments. Set a closed event to <i>pending</i> to include it in the next automatic batch or in the Send button's batch.</p>{event_rows}<script>async function setLlmStatus(id,status){{let r=await fetch('/events/'+id+'/llm-status',{{method:'PATCH',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{llm_status:status}})}});if(!r.ok){{alert((await r.json()).detail);location.reload();return}}location.reload()}}async function closeEvent(id){{let r=await fetch('/events/'+id+'/close',{{method:'POST'}});if(!r.ok){{alert((await r.json()).detail);location.reload();return}}location.reload()}}async function setAutomaticBatching(paused){{let response=await fetch('/memory-batches/automatic-pause',{{method:'PUT',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{paused}})}});if(!response.ok){{alert('Could not update automatic batching.');return}}location.reload()}}async function sendBatch(){{let button=event.target;button.disabled=true;try{{let r=await fetch('/memories/consolidate',{{method:'POST'}});let body=await r.text();let data;try{{data=JSON.parse(body)}}catch{{data={{detail:body||'Server error while sending to the LLM.'}}}}if(!r.ok){{throw new Error(data.detail||'Server error while sending to the LLM.')}}document.getElementById('batch-result').textContent=' Processed '+data.processed_memories+' memories.';setTimeout(()=>location.reload(),1200)}}catch(error){{document.getElementById('batch-result').textContent=' '+error.message;button.disabled=false}}}}</script></body></html>'''


@app.post("/memories/consolidate")
async def consolidate_memories_now():
    """Run a currently pending closed-event batch immediately, for testing/admin use."""
    consolidator = app.state.memory_consolidator
    if consolidator is None:
        raise HTTPException(status_code=503, detail="Memory consolidation is not configured")
    results = await _run_memory_batch(app, force=True)
    return {"processed_memories": len(results), "results": results}


async def _receive_stream_format(websocket: WebSocket) -> tuple[StreamAudioFormat, int | None]:
    """Read and validate the required first JSON WebSocket message."""

    message = await websocket.receive()
    raw_text = message.get("text")
    if raw_text is None:
        raise ValueError("First WebSocket message must be JSON format metadata")
    try:
        metadata = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise ValueError("Format metadata must be valid JSON") from error
    if not isinstance(metadata, dict):
        raise ValueError("Format metadata must be a JSON object")
    patient_id = metadata.get("patient_id")
    if patient_id is not None:
        try:
            patient_id = int(patient_id)
        except (TypeError, ValueError) as error:
            raise ValueError("patient_id must be an integer when provided") from error
        if patient_id <= 0:
            raise ValueError("patient_id must be greater than zero")
    return StreamAudioFormat.from_message(metadata), patient_id


@app.websocket("/ws/audio")
async def audio_stream(websocket: WebSocket):
    """Receive binary PCM audio and process finalized speech recordings."""

    await websocket.accept()
    session_id = allocate_stream_session_id()
    loop = asyncio.get_running_loop()
    identity_events: asyncio.Queue[dict[str, object]] = asyncio.Queue(maxsize=50)

    def on_identity_result(result: dict[str, object]) -> None:
        def enqueue_result() -> None:
            try:
                identity_events.put_nowait(result)
            except asyncio.QueueFull:
                pass

        loop.call_soon_threadsafe(enqueue_result)

    chunk_count = source_bytes = canonical_bytes = 0
    last_progress_at = time.monotonic()
    flush_requested = False

    try:
        try:
            source_format, patient_id = await _receive_stream_format(websocket)
        except ValueError as error:
            LOGGER.warning(
                "[STREAM] format=rejected session=%s error=%s",
                session_id,
                error,
            )
            await websocket.send_json({"received": False, "error": str(error)})
            await websocket.close(code=1003)
            return

        processor = StreamingSegmentProcessor(
            session_id,
            patient_id=patient_id,
            identity_callback=on_identity_result,
        )
        recorder = StreamingSpeechRecorder(
            on_finalized=processor.submit,
            on_confirmed_idle_audio=processor.add_confirmed_idle_audio,
        )

        LOGGER.info(
            "[STREAM] format=accepted session=%s source=%sHz/%sch/%s normalization=%s",
            session_id,
            source_format.sample_rate,
            source_format.channels,
            source_format.encoding,
            "passthrough" if source_format.is_canonical else "enabled",
        )
        await websocket.send_json({"received": True, "session_id": session_id})

        while True:
            message = await websocket.receive()
            if message.get("text") is not None:
                try:
                    payload = json.loads(message["text"])
                    if payload.get("action") == "flush_and_stop":
                        LOGGER.info("[STREAM] flush_and_stop received session=%s; draining pipeline...", session_id)
                        flush_requested = True
                        break
                except Exception:
                    pass
                await websocket.send_json(
                    {"received": False, "error": "Audio format cannot change after stream start"}
                )
                await websocket.close(code=1003)
                return
            data = message.get("bytes")
            if data is None:
                raise WebSocketDisconnect(code=1000)
            try:
                canonical_data = adapt_pcm16_stream_chunk(data, source_format)
            except ValueError as error:
                LOGGER.warning(
                    "[STREAM] normalization=rejected session=%s error=%s",
                    session_id,
                    error,
                )
                await websocket.send_json({"received": False, "error": str(error)})
                await websocket.close(code=1003)
                return

            recorder.process(canonical_data)
            chunk_count += 1
            source_bytes += len(data)
            canonical_bytes += len(canonical_data)
            LOGGER.debug(
                "[STREAM] normalization=processed session=%s source_bytes=%s canonical_bytes=%s",
                session_id,
                len(data),
                len(canonical_data),
            )
            now = time.monotonic()
            if now - last_progress_at >= 15.0:
                status = recorder.status_snapshot()
                LOGGER.debug(
                    "[STREAM] progress session=%s chunks=%s source_bytes=%s "
                    "canonical_bytes=%s temp_buffer=%s/%.3fs vad=%s frames=%s "
                    "probability=%s recording=%s main_duration=%.3fs",
                    session_id,
                    chunk_count,
                    source_bytes,
                    canonical_bytes,
                    status["temp_buffer_state"],
                    status["temp_buffer_duration_sec"],
                    status["vad_state"],
                    status["vad_frames"],
                    status["last_probability"],
                    status["active_segment_id"],
                    status["main_duration_sec"],
                )
                last_progress_at = now

            try:
                identity_result = identity_events.get_nowait()
            except asyncio.QueueEmpty:
                identity_result = None

            acknowledgement = {
                "received": True,
                "chunk_number": chunk_count,
                "chunk_size": len(data),
            }
            if identity_result is not None:
                acknowledgement["identity_result"] = identity_result

            await websocket.send_json(acknowledgement)

    except WebSocketDisconnect:
        LOGGER.info("[STREAM] disconnected session=%s", session_id)
    finally:
        recorder.finish()
        await processor.drain()
        while not identity_events.empty():
            try:
                event = identity_events.get_nowait()
                await websocket.send_json({"received": True, "identity_result": event})
            except Exception:
                break
        # Do not close immediately after sending unknown-voice capture events.
        # The phone must first drain its camera queue and acknowledge it. This
        # keeps the same WebSocket session alive throughout the capture flow.
        if flush_requested:
            try:
                await websocket.send_json({"received": True, "event": "capture_queue_ready"})
                LOGGER.info("[STREAM] capture queue ready session=%s; waiting for phone acknowledgement", session_id)
                while True:
                    message = await asyncio.wait_for(websocket.receive(), timeout=180)
                    if message.get("text") is None:
                        continue
                    payload = json.loads(message["text"])
                    if payload.get("action") == "capture_queue_complete":
                        LOGGER.info("[STREAM] capture queue complete session=%s", session_id)
                        break
            except asyncio.TimeoutError:
                LOGGER.warning("[STREAM] capture queue acknowledgement timed out session=%s", session_id)
            except (WebSocketDisconnect, json.JSONDecodeError):
                LOGGER.info("[STREAM] capture queue disconnected session=%s", session_id)
        try:
            await websocket.send_json({"received": True, "event": "stream_closed"})
            await websocket.close(code=1000)
        except Exception:
            pass
        LOGGER.info(
            "[STREAM] closed session=%s chunks=%s source_bytes=%s canonical_bytes=%s",
            session_id,
            chunk_count,
            source_bytes,
            canonical_bytes,
        )
