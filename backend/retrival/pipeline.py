"""End-to-end retrieval: session → parser → resolve → fetch → answer."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from retrival.answer import empty_records, write_answer
from retrival.config import (
    FOLLOWUP_EMPTY, NEED_CAMERA, NO_RECENT_EVENTS, REFUSAL, UNRECOGNISED, VAGUE_QUESTION,
)
from retrival.debug_log import log_query
from retrival.fetch import (
    compact_memory, faiss_other, memories_by_ids, retrieve_happening,
    retrieve_schedule_recall, retrieve_who_is, source_events_for,
)
from retrival.parser import inherit_slots, is_vague, parse_question
from retrival.resolver import resolve_query
from retrival.sessions import (
    clear_conversation, get_or_start_session, public_session, touch_session,
)

TZ = ZoneInfo("Asia/Kathmandu")


def _get_patient_name(patient_id: int) -> str:
    try:
        import sqlite3
        from retrival.config import BACKEND_ROOT
        db_path = BACKEND_ROOT / "uploads" / "database" / "reconnect.db"
        if db_path.exists():
            with sqlite3.connect(str(db_path)) as conn:
                row = conn.execute("SELECT full_name FROM patients WHERE id = ?", (patient_id,)).fetchone()
                if row and row[0]:
                    return row[0].strip()
    except Exception:
        pass
    return "the wearer"


def _store(context) -> object:
    return context["memory_store"]


def run_query(
    *,
    patient_id: int,
    question: str,
    context: dict,
    face_result: dict | None = None,
    asked_at: datetime | None = None,
) -> dict:
    asked_at = asked_at or datetime.now(TZ)
    question = (question or "").strip()
    llm = context["llm"]
    session, closed_previous = get_or_start_session(patient_id, asked_at)
    parsed, parser_raw = parse_question(question, llm, session=session, patient_id=patient_id)
    is_followup = bool(parsed.get("is_followup"))

    if is_followup and not session.get("last_query"):
        answer = FOLLOWUP_EMPTY
        return _finish(session, patient_id, question, parsed, None, answer, [], closed_previous, parser_raw, "followup_empty")

    if is_followup:
        parsed = inherit_slots(parsed, session.get("last_query"))
    else:
        clear_conversation(session)

    if is_vague(parsed):
        answer = VAGUE_QUESTION
        return _finish(session, patient_id, question, parsed, None, answer, [], closed_previous, parser_raw, "vague")

    resolved = resolve_query(parsed, patient_id, graph_store=context.get("graph_store"), now=asked_at)
    person = resolved.get("person") or {}
    if person.get("status") == "ambiguous":
        answer = person["message"]
        return _finish(session, patient_id, question, parsed, resolved, answer, [], closed_previous, parser_raw, "ambiguous_person")
    if resolved.get("occasion_missing_date"):
        parsed = dict(parsed)
        parsed["intent"] = "other"

    intent = parsed["intent"]
    records = None
    kind = intent
    action = None

    if intent == "who_is":
        person = resolved.get("person") or {}
        person_name = person.get("name") if person.get("status") == "resolved" else (person.get("name") or parsed.get("person_ref"))
        
        # If a specific person or relation was verbally named, answer from memory without camera!
        if face_result is None and person_name and person_name.casefold() not in {"this", "that", "this person", "that person"}:
            bundle = retrieve_who_is(
                _store(context), patient_id, person_name, asked_at, context.get("graph_store"),
                relation=person.get("relation"),
            )
            bundle["person"] = {
                "name": person.get("name") or person_name,
                "relation": person.get("relation"),
                "person_id": person.get("person_id"),
            }
            records = bundle
            kind = "who_is"
            if empty_records("who_is", records, resolved=resolved):
                recall_mems = retrieve_schedule_recall(
                    _store(context), patient_id, {**parsed, "intent": "recall"}, resolved,
                    graph_store=context.get("graph_store"),
                )
                if recall_mems:
                    records = recall_mems
                    kind = "recall"
                else:
                    answer = REFUSAL
                    return _finish(session, patient_id, question, parsed, resolved, answer, [], closed_previous, parser_raw, "empty")
        elif face_result is None:
            action = "need_camera"
            answer = NEED_CAMERA
            return _finish(session, patient_id, question, parsed, resolved, answer, [], closed_previous, parser_raw, "need_camera", action=action)
        elif face_result.get("status") != "known":
            answer = face_result.get("message") or UNRECOGNISED
            return _finish(session, patient_id, question, parsed, resolved, answer, [], closed_previous, parser_raw, face_result.get("status") or "unrecognised")
        else:
            resolved["person"] = {
                "name": face_result.get("name"),
                "relation": face_result.get("relation"),
                "person_id": face_result.get("person_id"),
                "status": "resolved",
            }
            bundle = retrieve_who_is(
                _store(context), patient_id, face_result["name"], asked_at, context.get("graph_store"),
                relation=face_result.get("relation"),
            )
            bundle["person"] = resolved["person"]
            records = bundle
            if empty_records("who_is", records, resolved=resolved):
                answer = REFUSAL
                return _finish(session, patient_id, question, parsed, resolved, answer, [], closed_previous, parser_raw, "empty")
    elif intent == "what_happening":
        earlier = bool(parsed.get("time_ref") and "earlier" in parsed["time_ref"].casefold())
        records = retrieve_happening(patient_id, asked_at, earlier=earlier)
        if empty_records("what_happening", records, resolved=resolved):
            answer = NO_RECENT_EVENTS
            return _finish(session, patient_id, question, parsed, resolved, answer, [], closed_previous, parser_raw, "empty")
    elif intent in {"schedule", "recall"}:
        records = retrieve_schedule_recall(
            _store(context), patient_id, parsed, resolved,
            graph_store=context.get("graph_store"),
        )
        kind = intent
        if empty_records(intent, records, resolved=resolved):
            answer = REFUSAL
            return _finish(session, patient_id, question, parsed, resolved, answer, [], closed_previous, parser_raw, "empty")
    else:
        # Keep API startup quick: FAISS/SentenceTransformer is loaded only if
        # an open-ended question actually needs semantic retrieval.
        vector_store = context.get("vector_store")
        if vector_store is None and callable(context.get("load_vector_store")):
            vector_store = context["load_vector_store"]()
            context["vector_store"] = vector_store
        records = faiss_other(
            question, _store(context), patient_id, vector_store, context.get("graph_store"),
        )
        kind = "other"
        if empty_records("other", records, resolved=resolved):
            answer = REFUSAL
            return _finish(session, patient_id, question, parsed, resolved, answer, [], closed_previous, parser_raw, "empty")

    extra_focus = []
    extra_events = []
    if is_followup and session.get("focus_memory_ids"):
        extra_focus = memories_by_ids(_store(context), session["focus_memory_ids"])
        extra_events = source_events_for(extra_focus)
        if isinstance(records, list) and records and records[0].get("memory_id"):
            known = {item.get("memory_id") for item in records if isinstance(item, dict)}
            merged = list(records) + [item for item in extra_focus if item.get("memory_id") not in known]
            records = {"memories": merged, "source_events": extra_events + source_events_for(merged)}
        elif isinstance(records, list):
            records = {"events": records, "focus_memories": extra_focus, "focus_events": extra_events}
        elif isinstance(records, dict):
            records = dict(records)
            records["focus_memories"] = extra_focus
            records["focus_events"] = extra_events

    from retrival.answer import records_payload
    patient_name = _get_patient_name(patient_id)
    answer = write_answer(
        question,
        records,
        llm,
        kind=kind,
        patient_id=patient_id,
        session_id=session["session_id"],
        resolved=resolved,
        turns=session.get("turns") or [],
        patient_name=patient_name,
    )
    focus_ids = _focus_ids(records)
    payload_to_log = records_payload(kind, records, resolved=resolved)
    return _finish(session, patient_id, question, parsed, resolved, answer, focus_ids, closed_previous, parser_raw, "answered", records=payload_to_log)


def _focus_ids(records) -> list[str]:
    if isinstance(records, list):
        return [item["memory_id"] for item in records if isinstance(item, dict) and item.get("memory_id")][:8]
    if isinstance(records, dict):
        ids = []
        for key in ("memories", "relationship_memories", "latest_memories", "commitments", "focus_memories"):
            for item in records.get(key) or []:
                if item.get("memory_id"):
                    ids.append(item["memory_id"])
        return ids[:8]
    return []


def _public_records(records):
    if records is None:
        return None
    if isinstance(records, list):
        return [compact_memory(item) if item.get("memory_id") else item for item in records]
    if isinstance(records, dict):
        out = dict(records)
        for key in ("memories", "relationship_memories", "latest_memories", "commitments", "focus_memories"):
            if key in out:
                out[key] = [compact_memory(item) for item in out[key]]
        return out
    return records


def _finish(session, patient_id, question, parsed, resolved, answer, focus_ids, closed_previous, parser_raw, outcome, records=None, action=None) -> dict:
    session = touch_session(session, question=question, answer=answer, parsed=parsed, focus_memory_ids=focus_ids)
    log_query({
        "patient_id": patient_id,
        "session_id": session["session_id"],
        "question": question,
        "parsed": parsed,
        "resolved": resolved,
        "intent": parsed.get("intent"),
        "answer": answer,
        "focus_memory_ids": focus_ids,
        "is_followup": parsed.get("is_followup"),
        "previous_session_closed": closed_previous,
        "records": _public_records(records),
    })
    return {
        "answer": answer,
        "intent": parsed.get("intent"),
        "parsed": parsed,
        "resolved": resolved,
        "session": public_session(session),
        "records": _public_records(records),
        "outcome": outcome,
        "action": action,
        "previous_session_closed": closed_previous,
        "parser_raw": parser_raw,
    }
