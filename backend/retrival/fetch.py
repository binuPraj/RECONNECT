"""Read-only retrieval over SQLite, events, Neo4j, and FAISS. No storage writes."""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from retrival.config import EVENT_WINDOW_MINUTES, VECTOR_THRESHOLD

TZ = ZoneInfo("Asia/Kathmandu")


def _key(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).casefold()


def _parse_dt(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=TZ)
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=TZ)


def _memory_mentions_person(memory: dict, name: str | None, relation: str | None = None) -> bool:
    needles = set()
    if name:
        needles.add(_key(name))
    if relation:
        needles.add(_key(relation))
    if not needles:
        return False
    for needle in needles:
        if needle in {_key(item) for item in memory.get("participants") or []}:
            return True
        for entry in memory.get("about") or []:
            if isinstance(entry, dict) and _key(str(entry.get("person") or "")) == needle:
                return True
        for entity in memory.get("entities") or []:
            if isinstance(entity, dict) and _key(str(entity.get("text") or "")) == needle:
                return True
        if needle in _key(memory.get("summary") or ""):
            return True
    return False


def _memory_mentions_place(memory: dict, place: str | None) -> bool:
    if not place:
        return False
    needle = _key(place)
    for entity in memory.get("entities") or []:
        if not isinstance(entity, dict):
            continue
        if str(entity.get("type") or "").upper() != "PLACE":
            continue
        if needle in _key(str(entity.get("text") or "")):
            return True
    return False


def _active_refs(memory: dict) -> list[dict]:
    return [ref for ref in memory.get("time_refs") or [] if ref.get("status") == "active"]


def _ref_overlaps(ref: dict, start: datetime, end: datetime) -> bool:
    ref_start, ref_end = _parse_dt(ref.get("start")), _parse_dt(ref.get("end"))
    if not ref_start or not ref_end:
        return False
    return ref_start < end and ref_end > start


def _recall_time_hit(memory: dict, start: datetime, end: datetime) -> bool:
    if any(_ref_overlaps(ref, start, end) for ref in _active_refs(memory)):
        return True
    for stamp in memory.get("event_starts") or []:
        moment = _parse_dt(stamp)
        if moment and start <= moment < end:
            return True
    for item in memory.get("update_history") or []:
        moment = _parse_dt(item.get("at") if isinstance(item, dict) else None)
        if moment and start <= moment < end:
            return True
    return False


def patient_memories(store, patient_id: int) -> list[dict]:
    return [memory for memory in store.list_all() if memory.get("patient_id") == patient_id]


def compact_memory(memory: dict) -> dict:
    return {
        "memory_id": memory.get("memory_id"),
        "category": memory.get("category"),
        "summary": memory.get("summary"),
        "topic": memory.get("topic"),
        "participants": memory.get("participants"),
        "about": memory.get("about"),
        "entities": memory.get("entities"),
        "time_refs": [
            {key: ref.get(key) for key in ("phrase", "start", "end", "status", "topic", "what_was_mentioned", "tense")}
            for ref in memory.get("time_refs") or []
        ],
        "updated_at": memory.get("updated_at"),
        "source_events": memory.get("source_events"),
    }


def rank_by_topic(memories: list[dict], topic: str, limit: int = 5) -> list[dict]:
    if len(memories) <= 5 or not topic:
        return memories[:8 if not topic else limit]
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        return memories[:limit]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    summaries = [memory.get("summary") or "" for memory in memories]
    vectors = model.encode([topic, *summaries], normalize_embeddings=True)
    query, docs = vectors[0], np.asarray(vectors[1:])
    scores = docs @ query
    order = list(np.argsort(-scores))
    return [memories[index] for index in order[:limit]]


def faiss_other(query: str, store, patient_id: int, vector_store=None, graph_store=None, k: int = 5) -> list[dict]:
    candidates: list[dict] = []
    ids: list[str] = []
    if vector_store is not None:
        ids = vector_store.search(query, k=max(k * 4, 20))
        found = store.get_many(ids) if hasattr(store, "get_many") else []
        candidates = [memory for memory in found if memory.get("patient_id") == patient_id]
        if not candidates and graph_store is not None:
            try:
                graph_rows = graph_store.get_by_ids(ids)
                graph_ids = [row["memory_id"] for row in graph_rows if row.get("memory_id")]
                found = store.get_many(graph_ids) if hasattr(store, "get_many") else []
                candidates = [memory for memory in found if memory.get("patient_id") == patient_id]
            except Exception:
                candidates = []
    if not candidates:
        candidates = patient_memories(store, patient_id)
    if not candidates:
        return []
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model = SentenceTransformer("all-MiniLM-L6-v2")
        summaries = [memory.get("summary") or "" for memory in candidates]
        vectors = model.encode([query, *summaries], normalize_embeddings=True)
        query_vec, docs = vectors[0], np.asarray(vectors[1:])
        scores = docs @ query_vec
        ranked = []
        for index in np.argsort(-scores):
            if float(scores[index]) < VECTOR_THRESHOLD:
                continue
            ranked.append(candidates[int(index)])
            if len(ranked) >= k:
                break
        return ranked
    except ImportError:
        return candidates[:k]


def filter_schedule_recall(memories: list[dict], resolved: dict, intent: str, category_hints: list[str] | None = None) -> list[dict]:
    person = resolved.get("person") or {}
    person_name = person.get("name") if person.get("status") == "resolved" else (person.get("name") or person.get("phrase"))
    if person.get("status") == "unresolved":
        person_name = person.get("name") or person.get("phrase")
    relation = person.get("relation")
    place = resolved.get("location")
    window = resolved.get("time_window")
    selected = list(memories)
    if intent == "schedule":
        selected = [memory for memory in selected if memory.get("category") == "commitment"]
        if window:
            start, end = _parse_dt(window["start"]), _parse_dt(window["end"])
            selected = [memory for memory in selected if any(_ref_overlaps(ref, start, end) for ref in _active_refs(memory))]
        else:
            now = datetime.now(TZ)
            selected = [
                memory for memory in selected
                if any(_parse_dt(ref.get("end")) and _parse_dt(ref.get("end")) > now for ref in _active_refs(memory))
            ]
    else:
        if window:
            start, end = _parse_dt(window["start"]), _parse_dt(window["end"])
            selected = [memory for memory in selected if _recall_time_hit(memory, start, end)]
    if person_name or relation:
        matched_person = [memory for memory in selected if _memory_mentions_person(memory, person_name, relation)]
        if matched_person:
            selected = matched_person
    if place:
        selected = [memory for memory in selected if _memory_mentions_place(memory, place)]

    # Boost memories matching category_hint if provided
    if category_hints:
        hints_set = {c.casefold() for c in category_hints}
        cat_matches = [m for m in selected if (m.get("category") or "").casefold() in hints_set]
        other_mems = [m for m in selected if (m.get("category") or "").casefold() not in hints_set]
        selected = cat_matches + other_mems

    return selected


def retrieve_schedule_recall(store, patient_id: int, parsed: dict, resolved: dict, graph_store=None) -> list[dict]:
    intent = parsed["intent"]
    memories = patient_memories(store, patient_id)
    person = resolved.get("person") or {}
    person_id = person.get("person_id") if person.get("status") == "resolved" else None
    person_name = person.get("name") if person.get("status") == "resolved" else (person.get("name") or person.get("phrase"))
    location = resolved.get("location")
    category_hints = parsed.get("category_hint") or []

    graph_ids = graph_filter_ids(graph_store, patient_id, person_id, location, person_name=person_name)
    if graph_ids is not None and (person_id or location or person_name) and graph_ids:
        narrowed = [memory for memory in memories if memory.get("memory_id") in graph_ids]
        if narrowed:
            memories = narrowed
    memories = filter_schedule_recall(memories, resolved, intent, category_hints=category_hints)
    topic = parsed.get("topic_ref")
    if topic and len(memories) > 5:
        return rank_by_topic(memories, topic, 5)
    if intent == "recall" and not resolved.get("time_window"):
        memories = sorted(memories, key=lambda item: item.get("updated_at") or "", reverse=True)
        return memories[:5]
    memories = sorted(memories, key=lambda item: (item.get("updated_at") or ""), reverse=intent != "schedule")
    if intent == "schedule":
        def sort_key(memory):
            ends = [_parse_dt(ref.get("start")) or datetime.max.replace(tzinfo=TZ) for ref in _active_refs(memory)]
            return min(ends) if ends else datetime.max.replace(tzinfo=TZ)
        memories = sorted(memories, key=sort_key)
    return memories[:8]


def retrieve_who_is(store, patient_id: int, person_name: str, now: datetime | None = None, graph_store=None, relation: str | None = None) -> dict:
    now = now or datetime.now(TZ)
    all_memories = patient_memories(store, patient_id)
    involving = [memory for memory in all_memories if _memory_mentions_person(memory, person_name, relation)]
    if graph_store is not None and not involving:
        try:
            graph_ids = [row["memory_id"] for row in graph_store.get_direct(person_name, 8)]
            involving = [memory for memory in all_memories if memory.get("memory_id") in set(graph_ids)]
        except Exception:
            involving = involving
    identity = [memory for memory in involving if memory.get("category") in {"identity", "relationship"}]
    identity = sorted(identity, key=lambda item: item.get("updated_at") or "", reverse=True)[:3]
    latest = sorted(involving, key=lambda item: item.get("updated_at") or "", reverse=True)[:3]
    upcoming = store.get_upcoming(now.isoformat(), 10, patient_id) if hasattr(store, "get_upcoming") else []
    upcoming = [memory for memory in upcoming if _memory_mentions_person(memory, person_name, relation)][:3]
    return {
        "relationship_memories": identity,
        "latest_memories": latest,
        "commitments": upcoming,
    }


def retrieve_happening(patient_id: int, asked_at: datetime | None = None, earlier: bool = False) -> list[dict]:
    from app.events.db import list_events
    from app.memory.consolidation import render_transcript_text

    asked_at = asked_at or datetime.now(TZ)
    minutes = EVENT_WINDOW_MINUTES
    start = asked_at - timedelta(minutes=minutes * (2 if earlier else 1))
    end = asked_at - timedelta(minutes=minutes) if earlier else asked_at
    events = list_events(patient_id=patient_id)
    found = []
    for event in events:
        status = event.get("status")
        event_start = _parse_dt(event.get("start"))
        event_end = _parse_dt(event.get("end")) or event_start
        if status == "open":
            found.append(event)
            continue
        if status == "closed" and event_end and start <= event_end <= end:
            found.append(event)

    # Fallback 1: If nothing in the last 5 minutes, grab the most recent closed events from today (up to 24h)
    if not found:
        day_start = asked_at - timedelta(hours=24)
        recent_closed = []
        for event in events:
            if event.get("status") == "closed":
                event_end = _parse_dt(event.get("end")) or _parse_dt(event.get("start"))
                if event_end and event_end >= day_start:
                    recent_closed.append((event_end, event))
        if recent_closed:
            recent_closed.sort(key=lambda item: item[0], reverse=True)
            for _, event in recent_closed[:2]:
                found.append(event)

    packed = []
    for event in found:
        packed.append({
            "event_id": event.get("event_id"),
            "status": event.get("status"),
            "start": event.get("start"),
            "end": event.get("end"),
            "participants": event.get("participants"),
            "transcript": render_transcript_text(event),
        })

    # Fallback 2: If still no events from today, check the latest memories
    if not packed:
        from app.memory.store import SQLiteMemoryStore
        store = SQLiteMemoryStore()
        all_mems = patient_memories(store, patient_id)
        if all_mems:
            latest_mems = sorted(all_mems, key=lambda m: m.get("updated_at") or "", reverse=True)[:2]
            for m in latest_mems:
                packed.append({
                    "memory_id": m.get("memory_id"),
                    "summary": m.get("summary"),
                    "category": m.get("category"),
                    "updated_at": m.get("updated_at"),
                    "participants": m.get("participants"),
                    "is_fallback_memory": True,
                })

    return packed


def memories_by_ids(store, memory_ids: list[str]) -> list[dict]:
    if not memory_ids:
        return []
    if hasattr(store, "get_many"):
        return store.get_many(memory_ids)
    return [store.get(memory_id) for memory_id in memory_ids if store.get(memory_id)]


def source_events_for(memories: list[dict]) -> list[dict]:
    from app.events.db import get_event
    from app.memory.consolidation import render_transcript_text

    seen: set[str] = set()
    events = []
    for memory in memories:
        for event_id in memory.get("source_events") or []:
            if event_id in seen:
                continue
            seen.add(event_id)
            event = get_event(event_id)
            if event is None:
                continue
            events.append({
                "event_id": event_id,
                "status": event.get("status"),
                "transcript": render_transcript_text(event),
            })
    return events


def graph_filter_ids(graph_store, patient_id: int, person_id: str | None, location: str | None, person_name: str | None = None) -> set[str] | None:
    """Optional Neo4j person/place narrowing. None means graph was not used."""
    if graph_store is None or (not person_id and not location and not person_name):
        return None
    clauses = ["m.patient_id = $patient_id"]
    params: dict = {"patient_id": patient_id}
    extra = ""
    if person_id or person_name:
        p_conds = []
        if person_id:
            p_conds.append("p.person_id = $person_id")
            params["person_id"] = person_id
        if person_name:
            p_conds.append("p.name_key = $name_key OR toLower(trim(coalesce(p.name, ''))) = $name_key")
            params["name_key"] = _key(person_name)
        extra += f" MATCH (p:Person) WHERE ({' OR '.join(p_conds)}) MATCH (p)-[:PRESENT_AT|CONCERNS]->(m)"
    if location:
        extra += """ MATCH (m)-[:MENTIONS|HAS_ENTITY]->(e:Entity)
                     WHERE e.type = 'PLACE' AND toLower(coalesce(e.text, e.text_key, '')) CONTAINS $place"""
        params["place"] = _key(location)
    try:
        with graph_store.driver.session() as session:
            rows = session.run(
                f"MATCH (m:Memory) WHERE {' AND '.join(clauses)} {extra} RETURN DISTINCT m.memory_id AS memory_id",
                **params,
            )
            return {row["memory_id"] for row in rows}
    except Exception:
        return None
