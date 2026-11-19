"""Reconcile historical data after a caregiver confirms an unknown speaker."""
from __future__ import annotations

import re
from typing import Any

from app.audio_process.resolution import backfill_unenrolled_transcripts
from app.events.db import replace_speaker_identity
from app.memory.store import SQLiteMemoryStore


def _key(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def _replace(value: str, unknown_label: str, enrolled_name: str) -> str:
    return enrolled_name if _key(value) == _key(unknown_label) else value


def _reconcile_memory(memory: dict[str, Any], unknown_label: str, enrolled_name: str) -> bool:
    changed = False
    participants: list[str] = []
    for participant in memory.get("participants", []):
        replacement = _replace(str(participant), unknown_label, enrolled_name)
        changed = changed or replacement != participant
        if _key(replacement) not in {_key(item) for item in participants}:
            participants.append(replacement)
    participant_keys = {_key(item) for item in participants}

    about: list[dict] = []
    for entry in memory.get("about", []):
        entry = dict(entry)
        person = _replace(str(entry.get("person", "")), unknown_label, enrolled_name)
        reporter = _replace(str(entry.get("reported_by", "")), unknown_label, enrolled_name)
        changed = changed or person != entry.get("person") or reporter != entry.get("reported_by")
        # A confirmed participant cannot remain an absent third-party concern.
        if _key(person) in participant_keys:
            changed = True
            continue
        entry["person"], entry["reported_by"] = person, reporter
        if entry not in about:
            about.append(entry)

    entities = []
    for entity in memory.get("entities", []):
        entity = dict(entity)
        if str(entity.get("type", "")).upper() == "PERSON":
            text = _replace(str(entity.get("text", "")), unknown_label, enrolled_name)
            changed = changed or text != entity.get("text")
            entity["text"] = text
        if entity not in entities:
            entities.append(entity)

    summary = memory.get("summary", "")
    rewritten_summary = re.sub(re.escape(unknown_label), enrolled_name, summary, flags=re.IGNORECASE)
    changed = changed or rewritten_summary != summary
    if changed:
        memory["participants"] = participants
        memory["about"] = about
        memory["entities"] = entities
        memory["summary"] = rewritten_summary
    return changed


def reconcile_confirmed_identity(
    *, unenrolled_id: int, enrolled_name: str, memory_store=None, graph_store=None, vector_store=None
) -> dict[str, int | bool]:
    """Apply the deterministic, no-LLM portion of caregiver identity confirmation."""
    unknown_label = f"unenrolled_{int(unenrolled_id)}"
    stream_changes = backfill_unenrolled_transcripts(unenrolled_id, enrolled_name)
    event_changes = replace_speaker_identity(unknown_label, enrolled_name)

    store = memory_store or SQLiteMemoryStore()
    changed_memories: list[dict] = []
    for memory in store.list_all():
        if _reconcile_memory(memory, unknown_label, enrolled_name):
            store.update(memory)
            changed_memories.append(memory)

    graph_synced = False
    if graph_store is not None:
        for memory in changed_memories:
            graph_store.rebuild_memory_people(memory)
            if vector_store is not None:
                vector_store.upsert(memory["memory_id"], memory["summary"])
        graph_store.retire_unenrolled_person(unknown_label, enrolled_name)
        graph_synced = True

    return {
        "stream_field_changes": stream_changes,
        "event_changes": event_changes,
        "memory_changes": len(changed_memories),
        "graph_synced": graph_synced,
    }
