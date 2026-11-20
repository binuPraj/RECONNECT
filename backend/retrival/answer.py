"""Answer LLM: only the retrieved records, or a fixed refusal with no LLM call."""
from __future__ import annotations

import json
import re

from retrival.config import REFUSAL
from retrival.fetch import compact_memory

ANSWER_PROMPT = """You answer questions for a person with memory difficulties, using ONLY the
records below. The patient is {patient_name}; speak to them as "you".
RULES:
- Use only facts in the records. Never add outside knowledge.
- If the answer is not in the records, set insufficient_context=true.
- A record with verified=false must be called unconfirmed, or left out.
- Report secondhand facts with their attribution ("Dad mentioned that...").
- Use dates exactly as shown. Never convert or guess dates.
- A plan for a past day is a plan, not proof it happened.
- 1-3 short, calm sentences. Never say "as I said before".
- If resolved_person is provided in the records, treat that person's relationship (e.g. Daughter) as authoritative.

Question: {question}

Recent conversation turns:
{turns}

Records JSON:
{records}
"""


def format_turns(turns: list | None) -> list[dict]:
    if not turns:
        return []
    formatted = []
    for t in turns[-4:]:
        if isinstance(t, dict):
            formatted.append({
                "question": t.get("question"),
                "answer": t.get("answer"),
            })
    return formatted


def records_payload(kind: str, records: dict | list, resolved: dict | None = None) -> dict:
    payload = {}

    # 1. Authoritative resolved identity and references
    if resolved:
        person = resolved.get("person") or {}
        if person.get("name") or person.get("relation"):
            payload["resolved_person"] = {
                "name": person.get("name"),
                "relation": person.get("relation"),
                "status": person.get("status"),
            }
        if resolved.get("time_window"):
            payload["resolved_time_window"] = resolved.get("time_window")
        if resolved.get("occasion"):
            payload["resolved_occasion"] = resolved.get("occasion")
        if resolved.get("location"):
            payload["resolved_location"] = resolved.get("location")

    # 2. Category-specific payloads
    if kind == "who_is" and isinstance(records, dict):
        if records.get("person"):
            payload["person"] = records.get("person")
        payload["relationship_memories"] = [compact_memory(item) for item in records.get("relationship_memories") or []]
        payload["latest_memories"] = [compact_memory(item) for item in records.get("latest_memories") or []]
        payload["commitments"] = [compact_memory(item) for item in records.get("commitments") or []]
        if records.get("focus_memories"):
            payload["focus_memories"] = [compact_memory(item) for item in records.get("focus_memories") or []]
        if records.get("focus_events"):
            payload["focus_events"] = records.get("focus_events")
        return payload

    if isinstance(records, dict) and ("memories" in records or "events" in records):
        payload["memories"] = [compact_memory(item) for item in records.get("memories") or []]
        payload["events"] = records.get("events") or []
        if records.get("source_events"):
            payload["source_events"] = records.get("source_events")
        if records.get("focus_memories"):
            payload["focus_memories"] = [compact_memory(item) for item in records.get("focus_memories") or []]
        if records.get("focus_events"):
            payload["focus_events"] = records.get("focus_events")
        return payload

    if isinstance(records, list):
        if records and isinstance(records[0], dict) and records[0].get("memory_id"):
            payload["memories"] = [compact_memory(item) for item in records]
        else:
            payload["memories"] = records
        return payload

    payload["records"] = records
    return payload


def empty_records(kind: str, records, resolved: dict | None = None) -> bool:
    if resolved and resolved.get("person") and resolved["person"].get("relation"):
        return False
    if records is None:
        return True
    if kind == "who_is" and isinstance(records, dict):
        if records.get("person"):
            return False
        return not (
            records.get("relationship_memories")
            or records.get("latest_memories")
            or records.get("commitments")
        )
    return not records


def _clean_response(text: str | None) -> str:
    if not text:
        return REFUSAL
    cleaned = text.strip()

    # Check for insufficient_context flag
    if "insufficient_context=true" in cleaned.lower() or "insufficient_context: true" in cleaned.lower():
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if len(lines) <= 2 and any("insufficient_context" in l.lower() for l in lines):
            return REFUSAL

    # Check if response is JSON with answer or insufficient_context
    if cleaned.startswith("{") and cleaned.endswith("}"):
        try:
            data = json.loads(cleaned)
            if data.get("insufficient_context") is True:
                return REFUSAL
            if data.get("answer"):
                return str(data["answer"]).strip()
        except Exception:
            pass

    # Remove any trailing "insufficient_context=false" artifacts from text
    cleaned = re.sub(r"(?i)\binsufficient_context\s*=\s*(true|false)\b", "", cleaned).strip()
    return cleaned or REFUSAL


def write_answer(
    question: str,
    records,
    llm,
    *,
    kind: str,
    patient_id: int,
    session_id: str,
    resolved: dict | None = None,
    turns: list | None = None,
    patient_name: str | None = None,
) -> str:
    payload = records_payload(kind, records, resolved=resolved)
    formatted_turns = format_turns(turns)
    name = patient_name or "the wearer"

    prompt = ANSWER_PROMPT.format(
        patient_name=name,
        question=question,
        turns=json.dumps(formatted_turns, indent=2) if formatted_turns else "None",
        records=json.dumps(payload, indent=2, default=str),
    )

    response = llm.complete(
        prompt,
        kind="answer",
        patient_id=patient_id,
        session_id=session_id,
        question=question,
        extra={"retrieve_kind": kind},
    )

    return _clean_response(response)
