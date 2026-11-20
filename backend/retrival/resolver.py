"""Turn parser word-slots into person IDs, date ranges, and places."""
from __future__ import annotations

import re
from datetime import datetime
from zoneinfo import ZoneInfo

from retrival.config import AMBIGUOUS_PERSON

TZ = ZoneInfo("Asia/Kathmandu")

RELATION_WORDS = {
    "daughter", "son", "sister", "brother", "mother", "father", "wife", "husband",
    "granddaughter", "grandson", "grandmother", "grandfather", "friend", "neighbour",
    "neighbor", "caregiver", "doctor", "nurse",
}


def _key(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).casefold()


def _strip_person_phrase(phrase: str) -> str:
    text = _key(phrase)
    text = re.sub(r"^(my|your|the|our)\s+", "", text)
    text = re.sub(r"'s$", "", text)
    return text.strip()


def resolve_time_ref(time_ref: str | None, *, now: datetime | None = None, tense: str | None = None) -> dict | None:
    if not time_ref:
        return None
    from app.memory.time_resolution import resolve_time
    anchor = now or datetime.now(TZ)
    window = resolve_time(time_ref, anchor, tense or "any")
    return window


def lookup_occasion(occasion_ref: str | None, patient_id: int) -> dict | None:
    """Occasions are not stored yet; undated occasions stay unresolved."""
    if not occasion_ref:
        return None
    return None


def resolve_person_ref(person_ref: str | None, patient_id: int, graph_store=None) -> dict:
    if not person_ref:
        return {"status": "none"}
    phrase = _strip_person_phrase(person_ref)
    if not phrase:
        return {"status": "none"}

    from database.db import get_patient_enrolled_people, get_patient_identity_by_name

    enrolled = get_patient_enrolled_people(patient_id)
    exact = get_patient_identity_by_name(person_ref.strip(), patient_id) or get_patient_identity_by_name(phrase.title(), patient_id)
    if exact is None:
        for row in enrolled:
            if _key(row["name"]) == phrase:
                exact = row
                break
    if exact is not None:
        row = dict(exact)
        return {
            "status": "resolved",
            "person_id": f"patient:{patient_id}:enrolled:{row['id']}",
            "name": row["name"],
            "relation": row.get("relation"),
            "enrolled_id": row["id"],
        }

    if phrase in RELATION_WORDS:
        matches = [dict(row) for row in enrolled if _key(row.get("relation") or "") == phrase]
        if graph_store is not None and not matches:
            matches = _graph_relations(graph_store, patient_id, phrase)
        if len(matches) == 1:
            row = matches[0]
            enrolled_id = row.get("id") or row.get("enrolled_id")
            return {
                "status": "resolved",
                "person_id": row.get("person_id") or f"patient:{patient_id}:enrolled:{enrolled_id}",
                "name": row["name"],
                "relation": row.get("relation") or phrase,
                "enrolled_id": enrolled_id,
            }
        if len(matches) > 1:
            names = ", ".join(item["name"] for item in matches)
            return {
                "status": "ambiguous",
                "relation": phrase,
                "matches": matches,
                "message": AMBIGUOUS_PERSON.format(relation=phrase) + f" I know {names}.",
            }
        return {"status": "unresolved", "phrase": person_ref}

    if graph_store is not None:
        hit = _graph_name(graph_store, phrase)
        if hit:
            return hit
    return {"status": "unresolved", "phrase": person_ref, "name": phrase}


def _graph_relations(graph_store, patient_id: int, relation: str) -> list[dict]:
    patient_person_id = f"patient:{patient_id}:patient"
    with graph_store.driver.session() as session:
        rows = session.run(
            """
            MATCH (p:Person)-[r:RELATED_TO]->(patient:Person {person_id:$patient_person_id})
            WHERE toLower(trim(coalesce(r.type, r.relation, ''))) = $relation
            RETURN p.name AS name, p.person_id AS person_id, p.identity_id AS enrolled_id, coalesce(r.type, r.relation) AS relation
            """,
            patient_person_id=patient_person_id,
            relation=_key(relation),
        )
        return [dict(row) for row in rows]


def _graph_name(graph_store, phrase: str) -> dict | None:
    with graph_store.driver.session() as session:
        row = session.run(
            """
            MATCH (p:Person {name_key:$name_key})
            RETURN p.name AS name, p.person_id AS person_id, p.identity_id AS enrolled_id
            LIMIT 1
            """,
            name_key=_key(phrase),
        ).single()
    if not row:
        return None
    return {
        "status": "resolved",
        "person_id": row["person_id"],
        "name": row["name"],
        "enrolled_id": row["enrolled_id"],
    }


def resolve_query(parsed: dict, patient_id: int, *, graph_store=None, now: datetime | None = None) -> dict:
    now = now or datetime.now(TZ)
    person = resolve_person_ref(parsed.get("person_ref"), patient_id, graph_store)
    time_window = resolve_time_ref(parsed.get("time_ref"), now=now, tense=parsed.get("tense"))
    occasion = lookup_occasion(parsed.get("occasion_ref"), patient_id)
    if parsed.get("occasion_ref") and occasion and occasion.get("start"):
        time_window = {"start": occasion["start"], "end": occasion["end"], "part": None, "granularity": "day"}
    return {
        "person": person,
        "time_window": time_window,
        "occasion": occasion,
        "location": parsed.get("location_ref"),
        "topic": parsed.get("topic_ref"),
        "tense": parsed.get("tense"),
        "occasion_missing_date": bool(parsed.get("occasion_ref") and not time_window),
    }
