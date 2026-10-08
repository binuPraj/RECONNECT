from datetime import datetime, timedelta, timezone
from app.memory.consolidation import _validate_results, apply_batch_response, build_batch_prompt, flatten_transcript, make_batch_prompt, parse_llm_json_response, render_transcript_text, should_run_batch
from app.memory.store import SQLiteMemoryStore
from app.memory.graph_store import GraphVectorMemoryStore

def make_event(event_id, start, end):
    return {"event_id": event_id, "status": "closed", "participants": ["kris"], "start": start, "end": end, "segments": [{"speaker": "kris", "start": start, "end": end, "text": "Tea please."}]}

def result(action, source_events, **extra):
    return {"action": action, "category": "preference", "summary": "Kris prefers tea.", "participants": ["kris"], "about": [], "source_events": source_events, "importance": "low", "entities": [], "emotion": "calm", **extra}

def test_flattening_and_rendering_accept_event_db_iso_strings():
    event = make_event("one", "2026-09-06T18:16:46+05:45", "2026-09-06T18:17:17+05:45")
    event["segments"] = [
        {"speaker": "kris", "start": "2026-09-06T18:16:46+05:45", "end": "2026-09-06T18:16:47+05:45", "text": "Hello?"},
        {"speaker": "kris", "start": "2026-09-06T18:16:48+05:45", "end": "2026-09-06T18:16:51+05:45", "text": "Welcome."},
        {"speaker": "unknown_4", "start": "2026-09-06T18:17:16+05:45", "end": "2026-09-06T18:17:17+05:45", "text": "Project."},
    ]
    assert flatten_transcript(event)[0]["end"] == event["segments"][1]["end"]
    assert "[2026-09-06 18:16:46+0545 - 2026-09-06 18:16:51+0545] kris: Hello? Welcome." in render_transcript_text(event)

def test_write_derives_ranges_and_preserves_created_at(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    first = make_event("first", "2026-01-02T12:00:00+00:00", "2026-01-02T12:05:00+00:00")
    second = make_event("second", "2026-01-01T12:00:00+00:00", "2026-01-03T12:05:00+00:00")
    stamp = datetime(2026, 2, 1, tzinfo=timezone.utc)
    apply_batch_response([result("new", ["first"])], {"first": first}, store, stamp)
    memory = store.get_recent("kris")[0]
    apply_batch_response([result("merge", ["second"], target_memory_id=memory["memory_id"], summary="Kris prefers morning tea.", emotion="happy")], {"first": first, "second": second}, store, stamp + timedelta(days=1))
    updated = store.get(memory["memory_id"])
    assert updated["created_at"] == memory["created_at"]
    assert updated["updated_at"] != memory["updated_at"]
    assert updated["earliest_event_time"] == second["start"]
    assert updated["latest_event_time"] == second["end"]
    assert updated["emotion"] == "happy"

def test_prompt_and_batch_trigger(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    event = make_event("one", "2026-01-01T12:00:00+00:00", "2026-01-01T12:05:00+00:00")
    existing, events = build_batch_prompt([event], store)
    assert existing == "(none)" and "[one] event_start=" in events
    now = datetime.now(timezone.utc)
    assert should_run_batch(now - timedelta(seconds=301), 1, now)
    assert not should_run_batch(now, 15, now)
    assert not should_run_batch(now, 0, now)

def test_llm_cannot_use_a_memory_id_as_a_source_event():
    invalid = result("merge", ["memory-uuid"], target_memory_id="memory-uuid")
    try:
        _validate_results([invalid], {"event-uuid"})
    except ValueError as error:
        assert str(error) == "LLM response references event outside this batch"
    else:
        raise AssertionError("Memory IDs must not be accepted as source event IDs")

def test_prompt_defaults_to_separate_memories_and_requires_a_genuine_merge_link(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    prompt = make_batch_prompt([make_event("event-uuid", "2026-01-01T12:00:00+00:00", "2026-01-01T12:01:00+00:00")], store)
    prompt_without_line_breaks = " ".join(prompt.split())
    assert "Create a NEW memory unless" in prompt
    assert "Sharing a participant" in prompt
    assert "same specific fact, relationship, plan, or ongoing matter" in prompt_without_line_breaks

def test_parse_llm_json_response_accepts_markdown_fenced_json():
    assert parse_llm_json_response("```json\n[{\"action\": \"new\"}]\n```") == [{"action": "new"}]

def test_absent_third_party_provenance_is_preserved(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    event = {**make_event("sarah-visit", "2026-09-07T09:00:00+00:00", "2026-09-07T09:01:00+00:00"), "participants": ["Dad"]}
    third_party = result("new", ["sarah-visit"], participants=["Dad"], about=[{"person": "Sarah", "reported_by": "Dad", "what_was_said": "Coming 2026-09-08 instead of 2026-09-07."}])
    apply_batch_response([third_party], {"sarah-visit": event}, store)
    assert store.list_all()[0]["about"] == third_party["about"]
    assert _validate_results([third_party], {"sarah-visit"})[0]["about"] == third_party["about"]

def test_graph_projection_failure_does_not_block_sqlite_memory_write(tmp_path):
    class FailingGraph:
        def write_memory(self, memory): raise RuntimeError("Neo4j unavailable")
    class UnusedVector:
        def upsert(self, memory_id, summary): raise AssertionError("vector should not run")

    store = GraphVectorMemoryStore(SQLiteMemoryStore(tmp_path / "memory.db"), FailingGraph(), UnusedVector())
    event = make_event("event", "2026-09-07T09:00:00+00:00", "2026-09-07T09:01:00+00:00")
    apply_batch_response([result("new", ["event"])], {"event": event}, store)
    assert len(store.primary.list_all()) == 1

def test_merge_reprojects_the_updated_memory_to_graph_and_vector(tmp_path):
    class RecordingGraph:
        def __init__(self): self.writes = []
        def write_memory(self, memory): self.writes.append(memory.copy())
    class RecordingVector:
        def __init__(self): self.upserts = []
        def upsert(self, memory_id, summary): self.upserts.append((memory_id, summary))

    graph, vector = RecordingGraph(), RecordingVector()
    store = GraphVectorMemoryStore(SQLiteMemoryStore(tmp_path / "memory.db"), graph, vector)
    first = make_event("first", "2026-01-01T12:00:00+00:00", "2026-01-01T12:01:00+00:00")
    second = make_event("second", "2026-01-02T12:00:00+00:00", "2026-01-02T12:01:00+00:00")
    apply_batch_response([result("new", ["first"])], {"first": first}, store)
    memory_id = store.primary.list_all()[0]["memory_id"]
    apply_batch_response([result("merge", ["second"], target_memory_id=memory_id, summary="Kris prefers green tea.")], {"first": first, "second": second}, store)

    assert [summary for _, summary in vector.upserts] == ["Kris prefers tea.", "Kris prefers green tea."]
    assert graph.writes[-1]["memory_id"] == memory_id
    assert graph.writes[-1]["summary"] == "Kris prefers green tea."

def test_merge_rejects_a_category_mismatch(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    first = make_event("first", "2026-01-01T12:00:00+00:00", "2026-01-01T12:01:00+00:00")
    apply_batch_response([result("new", ["first"], category="identity")], {"first": first}, store)
    memory_id = store.list_all()[0]["memory_id"]
    second = make_event("second", "2026-01-02T12:00:00+00:00", "2026-01-02T12:01:00+00:00")
    try:
        apply_batch_response([result("merge", ["second"], category="relationship", target_memory_id=memory_id)], {"first": first, "second": second}, store)
    except ValueError as error:
        assert "Cannot merge relationship memory into identity memory" in str(error)
    else:
        raise AssertionError("A relationship must never merge into an identity memory")
