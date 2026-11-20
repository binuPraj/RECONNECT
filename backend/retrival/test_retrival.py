"""Offline checks for session, parser fallback, person resolve, and empty refusals."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from retrival.parser import heuristic_parse, inherit_slots, is_vague, normalize_parsed
from retrival.config import SESSION_GAP_SECONDS
from retrival.sessions import _expired

TZ = ZoneInfo("Asia/Kathmandu")


class FakeLLM:
    def __init__(self, payload: str | None) -> None:
        self.payload = payload
        self.calls = []

    def complete(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        return self.payload


class FakeStore:
    def __init__(self, memories=None) -> None:
        self._memories = memories or []

    def list_all(self):
        return list(self._memories)

    def get_upcoming(self, now, limit=10, patient_id=None):
        return []

    def get_many(self, ids):
        by_id = {item["memory_id"]: item for item in self._memories}
        return [by_id[item] for item in ids if item in by_id]

    def get(self, memory_id):
        return next((item for item in self._memories if item["memory_id"] == memory_id), None)


def test_heuristic_intents():
    assert heuristic_parse("who is this")["intent"] == "who_is"
    assert heuristic_parse("what's happening")["intent"] == "what_happening"
    assert heuristic_parse("what am I supposed to do next week")["intent"] == "schedule"
    assert heuristic_parse("what happened last Monday with my sister")["intent"] == "recall"
    assert heuristic_parse("what happened last Monday with my sister")["person_ref"]


def test_invalid_json_is_other():
    parsed = normalize_parsed({"intent": "banana", "drill_down": True})
    assert parsed["intent"] == "other"
    assert "drill_down" not in parsed


def test_inherit_blank_slots_only():
    last = {"intent": "recall", "person_ref": "my sister", "time_ref": "last monday", "topic_ref": None}
    current = {"intent": "recall", "person_ref": None, "time_ref": "Tuesday", "topic_ref": None, "occasion_ref": None, "location_ref": None, "tense": None, "is_followup": True}
    merged = inherit_slots(current, last)
    assert merged["person_ref"] == "my sister"
    assert merged["time_ref"] == "Tuesday"


def test_vague_other():
    assert is_vague({"intent": "other", "person_ref": None, "time_ref": None, "occasion_ref": None, "location_ref": None, "topic_ref": None})
    assert not is_vague({"intent": "who_is", "person_ref": None, "time_ref": None, "occasion_ref": None, "location_ref": None, "topic_ref": None})


def test_session_gap_from_last_question():
    now = datetime.now(TZ)
    session = {"last_question_at": (now - timedelta(seconds=SESSION_GAP_SECONDS + 1)).isoformat()}
    assert _expired(session, now)
    session = {"last_question_at": (now - timedelta(seconds=max(SESSION_GAP_SECONDS - 10, 1))).isoformat()}
    assert not _expired(session, now)


def test_followup_without_last_query_refuses():
    from retrival.pipeline import run_query
    from retrival.config import FOLLOWUP_EMPTY
    from retrival.sessions import clear_conversation, get_or_start_session

    llm = FakeLLM('{"intent":"recall","person_ref":null,"time_ref":null,"occasion_ref":null,"location_ref":null,"topic_ref":null,"tense":null,"is_followup":true}')
    context = {"llm": llm, "memory_store": FakeStore([]), "graph_store": None, "vector_store": None}
    session, _ = get_or_start_session(99)
    clear_conversation(session)
    from retrival.sessions import _save
    _save(session)
    result = run_query(patient_id=99, question="what about that", context=context)
    kinds = [item["kind"] for item in llm.calls]
    assert "answer" not in kinds
    assert result["answer"] == FOLLOWUP_EMPTY


def test_vague_asks_to_be_specific():
    from retrival.pipeline import run_query
    from retrival.config import VAGUE_QUESTION

    llm = FakeLLM('{"intent":"other","person_ref":null,"time_ref":null,"occasion_ref":null,"location_ref":null,"topic_ref":null,"tense":null,"is_followup":false}')
    context = {"llm": llm, "memory_store": FakeStore([]), "graph_store": None, "vector_store": None}
    result = run_query(patient_id=1, question="hmm", context=context)
    assert result["answer"] == VAGUE_QUESTION
    assert "answer" not in [item["kind"] for item in llm.calls]


def test_empty_schedule_does_not_call_answer_llm(monkeypatch=None):
    from retrival.pipeline import run_query
    from retrival.config import REFUSAL, VAGUE_QUESTION

    llm = FakeLLM('{"intent":"schedule","person_ref":null,"time_ref":"next week","occasion_ref":null,"location_ref":null,"topic_ref":null,"tense":"future","is_followup":false}')
    context = {"llm": llm, "memory_store": FakeStore([]), "graph_store": None, "vector_store": None}
    result = run_query(patient_id=1, question="what am I supposed to do next week", context=context)
    kinds = [item["kind"] for item in llm.calls]
    assert "parser" in kinds
    assert "answer" not in kinds
    assert result["answer"] in {REFUSAL, VAGUE_QUESTION} or result["outcome"] == "empty"


def test_schedule_keeps_overlapping_commitments():
    from retrival.fetch import filter_schedule_recall

    now = datetime(2026, 11, 20, 10, 0, tzinfo=TZ)
    memory = {
        "memory_id": "m1",
        "category": "commitment",
        "summary": "Meet daughter",
        "participants": ["Rina"],
        "about": [{"person": "Rina"}],
        "entities": [],
        "updated_at": now.isoformat(),
        "time_refs": [{"status": "active", "start": "2026-11-23T00:00:00+05:45", "end": "2026-11-30T00:00:00+05:45"}],
        "event_starts": [],
        "update_history": [],
    }
    resolved = {
        "person": {"status": "resolved", "name": "Rina"},
        "time_window": {"start": "2026-11-23T00:00:00+05:45", "end": "2026-11-30T00:00:00+05:45"},
        "location": None,
    }
    found = filter_schedule_recall([memory], resolved, "schedule")
    assert found == [memory]


def test_who_is_does_not_leak_other_people():
    from retrival.fetch import retrieve_who_is

    store = FakeStore([
        {
            "memory_id": "other",
            "patient_id": 1,
            "category": "significant_event",
            "summary": "Sam visited",
            "participants": ["Sam"],
            "about": [],
            "entities": [],
            "updated_at": "2026-11-20T10:00:00+05:45",
            "time_refs": [],
        },
        {
            "memory_id": "rina",
            "patient_id": 1,
            "category": "relationship",
            "summary": "Rina is the daughter",
            "participants": ["Rina"],
            "about": [{"person": "Rina"}],
            "entities": [],
            "updated_at": "2026-11-19T10:00:00+05:45",
            "time_refs": [],
        },
    ])
    bundle = retrieve_who_is(store, 1, "Rina")
    assert [item["memory_id"] for item in bundle["relationship_memories"]] == ["rina"]
    assert [item["memory_id"] for item in bundle["latest_memories"]] == ["rina"]
    assert bundle["commitments"] == []


def test_schedule_pipeline_answers_from_records():
    from retrival.pipeline import run_query

    llm = FakeLLM('{"intent":"schedule","person_ref":null,"time_ref":"next week","occasion_ref":null,"location_ref":null,"topic_ref":null,"tense":"future","is_followup":false}')
    store = FakeStore([{
        "memory_id": "c1",
        "patient_id": 1,
        "category": "commitment",
        "summary": "Meet daughter next week",
        "participants": ["Rina"],
        "about": [{"person": "Rina"}],
        "entities": [],
        "updated_at": "2026-11-20T10:00:00+05:45",
        "time_refs": [{"status": "active", "start": "2026-11-23T00:00:00+05:45", "end": "2026-11-30T00:00:00+05:45"}],
        "event_starts": [],
        "update_history": [],
    }])
    result = run_query(
        patient_id=1,
        question="what am I supposed to do next week",
        context={"llm": llm, "memory_store": store, "graph_store": None, "vector_store": None},
        asked_at=datetime(2026, 11, 20, 10, 0, tzinfo=TZ),
    )
    kinds = [item["kind"] for item in llm.calls]
    assert "parser" in kinds
    assert "answer" in kinds
    assert result["intent"] == "schedule"
    assert result["outcome"] == "answered"


def test_recall_without_time_keeps_five():
    from retrival.fetch import retrieve_schedule_recall

    memories = []
    for index in range(8):
        memories.append({
            "memory_id": f"m{index}",
            "patient_id": 1,
            "category": "significant_event",
            "summary": f"Visit {index}",
            "participants": ["Rina"],
            "about": [{"person": "Rina"}],
            "entities": [],
            "updated_at": f"2026-11-{10+index:02d}T10:00:00+05:45",
            "time_refs": [],
            "event_starts": [],
            "update_history": [],
        })
    store = FakeStore(memories)
    parsed = {"intent": "recall", "topic_ref": None}
    resolved = {"person": {"status": "resolved", "name": "Rina"}, "time_window": None, "location": None}
    found = retrieve_schedule_recall(store, 1, parsed, resolved, graph_store=None)
    assert len(found) == 5
    assert found[0]["memory_id"] == "m7"


def test_open_ended_query_loads_vector_only_when_needed():
    """Structured questions must not initialise the slow vector model."""
    from retrival.pipeline import run_query

    loaded = []

    def loader():
        loaded.append(True)
        return None

    llm = FakeLLM('{"intent":"schedule","person_ref":null,"time_ref":"next week","occasion_ref":null,"location_ref":null,"topic_ref":null,"tense":"future","is_followup":false}')
    context = {
        "llm": llm, "memory_store": FakeStore([]), "graph_store": None,
        "vector_store": None, "load_vector_store": loader,
    }
    run_query(patient_id=1, question="what am I supposed to do next week", context=context)
    assert loaded == []


if __name__ == "__main__":
    test_heuristic_intents()
    test_invalid_json_is_other()
    test_inherit_blank_slots_only()
    test_vague_other()
    test_session_gap_from_last_question()
    test_followup_without_last_query_refuses()
    test_vague_asks_to_be_specific()
    test_empty_schedule_does_not_call_answer_llm()
    test_schedule_keeps_overlapping_commitments()
    test_who_is_does_not_leak_other_people()
    test_schedule_pipeline_answers_from_records()
    test_recall_without_time_keeps_five()
    test_open_ended_query_loads_vector_only_when_needed()
    print("retrival unit checks passed")
