from datetime import datetime, timedelta, timezone
from app.memory.consolidation import _normalize_aliases, _validate_results, apply_batch_response, build_batch_prompt, flatten_transcript, make_batch_prompt, parse_llm_json_response, render_transcript_text, should_run_batch
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
    apply_batch_response([result("merge", ["second"], target_memory_id=memory["memory_id"], summary="Kris prefers morning tea.", emotion="happy", what_was_updated="Preference updated from tea to morning tea.")], {"first": first, "second": second}, store, stamp + timedelta(days=1))
    updated = store.get(memory["memory_id"])
    assert updated["created_at"] == memory["created_at"]
    assert updated["updated_at"] != memory["updated_at"]
    assert updated["earliest_event_time"] == second["start"]
    assert updated["latest_event_time"] == second["end"]
    assert updated["emotion"] == "happy"
    assert [entry["action"] for entry in updated["update_history"]] == ["new", "merge"]
    assert updated["update_history"][-1]["what_was_updated"] == "Preference updated from tea to morning tea."
    assert updated["updated_events"] == [{
        "event_time": second["end"],
        "what_was_updated": "Preference updated from tea to morning tea.",
    }]


def test_time_reference_persists_the_supported_statement(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    event = make_event("time-event", "2026-09-16T12:00:00+05:45", "2026-09-16T12:01:00+05:45")
    event["segments"][0]["text"] = "Ram picked up Sita last Monday."
    item = result("new", ["time-event"], time_refs=[{
        "phrase": "last Monday", "tense": "past", "from_event": "time-event",
        "what_was_mentioned": "Ram picked up Sita last Monday.",
    }])
    apply_batch_response([item], {"time-event": event}, store)
    ref = store.list_all()[0]["time_refs"][0]
    assert ref["what_was_mentioned"] == "Ram picked up Sita last Monday."


def test_time_reference_persists_subject_and_commitment_context(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    event = make_event("plan", "2026-09-16T12:00:00+05:45", "2026-09-16T12:01:00+05:45")
    event["segments"][0]["text"] = "Sam will take Rita to the doctor next Tuesday."
    item = result("new", ["plan"], category="commitment", time_refs=[{
        "ref_key": "T1", "phrase": "next Tuesday", "tense": "future", "from_event": "plan",
        "topic": "Rita doctor appointment", "concerned_with": "Rita", "commitment_by": "Sam",
        "what_was_mentioned": "Sam will take Rita to the doctor next Tuesday.",
    }])
    apply_batch_response([item], {"plan": event}, store)
    ref = store.list_all()[0]["time_refs"][0]
    assert (ref["topic"], ref["concerned_with"], ref["commitment_by"]) == ("Rita doctor appointment", "Rita", "Sam")


def test_existing_timeref_is_linked_to_its_valid_replacement(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    first = make_event("first", "2026-09-16T12:00:00+05:45", "2026-09-16T12:01:00+05:45")
    first["segments"][0]["text"] = "Sam will take Rita next Monday."
    apply_batch_response([result("new", ["first"], category="commitment", time_refs=[{
        "ref_key": "T1", "phrase": "next Monday", "tense": "future", "from_event": "first",
        "topic": "Rita appointment", "concerned_with": "Rita", "commitment_by": "Sam",
        "what_was_mentioned": "Sam will take Rita next Monday.",
    }])], {"first": first}, store)
    memory = store.list_all()[0]
    second = make_event("second", "2026-09-17T12:00:00+05:45", "2026-09-17T12:01:00+05:45")
    second["segments"][0]["text"] = "Sam will take Rita next Tuesday instead."
    apply_batch_response([result("merge", ["second"], category="commitment", target_memory_id=memory["memory_id"], what_was_updated="Appointment rescheduled.", time_refs=[{
        "ref_key": "T1", "phrase": "next Tuesday", "tense": "future", "from_event": "second",
        "topic": "Rita appointment", "concerned_with": "Rita", "commitment_by": "Sam",
        "what_was_mentioned": "Sam will take Rita next Tuesday.",
    }], supersedes_time_refs={memory["time_refs"][0]["time_ref_id"]: "T1"})], {"second": second}, store)
    old, new = store.get(memory["memory_id"])["time_refs"]
    assert old["status"] == "superseded"
    assert old["superseded_by"] == new["time_ref_id"]


def test_explicit_reschedule_keeps_one_memory_when_llm_renames_the_subject(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    first = make_event("first", "2026-09-16T12:00:00+05:45", "2026-09-16T12:01:00+05:45")
    first["segments"][0]["text"] = "Mark will take ss to Mark's house next Thursday."
    apply_batch_response([result("new", ["first"], category="commitment", topic="visit to Mark's house", time_refs=[{
        "ref_key": "T1", "phrase": "next Thursday", "tense": "future", "from_event": "first",
        "topic": "visit to Mark's house", "concerned_with": "ss", "commitment_by": "Mark",
        "what_was_mentioned": "Mark will take ss to Mark's house next Thursday.",
    }])], {"first": first}, store)
    memory = store.list_all()[0]
    old_ref = memory["time_refs"][0]
    second = make_event("second", "2026-09-17T12:00:00+05:45", "2026-09-17T12:01:00+05:45")
    second["segments"][0]["text"] = "Mark will take ss to Mark's house next Friday instead."
    apply_batch_response([result("merge", ["second"], category="commitment", target_memory_id=memory["memory_id"], topic="visit to Mark", summary="Mark and ss rescheduled their visit to Mark's house from Thursday to Friday.", what_was_updated="Visit rescheduled from Thursday to Friday.", time_refs=[{
        "ref_key": "T1", "phrase": "next Friday", "tense": "future", "from_event": "second",
        "topic": "rescheduled visit", "concerned_with": "Mark", "commitment_by": "Mark",
        "what_was_mentioned": "Mark will take ss to Mark's house next Friday instead.",
    }], supersedes_time_refs={old_ref["time_ref_id"]: "T1"})], {"second": second}, store)
    merged = store.get(memory["memory_id"])
    assert len(store.list_all()) == 1
    assert merged["time_refs"][0]["superseded_by"] == merged["time_refs"][1]["time_ref_id"]


def test_existing_timeref_upsert_can_move_to_the_retained_memory(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    event = make_event("event", "2026-09-16T12:00:00+05:45", "2026-09-16T12:01:00+05:45")
    first = result("new", ["event"], time_refs=[{
        "phrase": "next Monday", "tense": "future", "from_event": "event",
        "what_was_mentioned": "Kris will visit next Monday.",
    }])
    second = result("new", ["event"], summary="Kris plans a second visit.", time_refs=[{
        "phrase": "next Tuesday", "tense": "future", "from_event": "event",
        "what_was_mentioned": "Kris will visit next Tuesday.",
    }])
    apply_batch_response([first, second], {"event": event}, store)
    memories = store.list_all()
    retained, duplicate = memories[0], memories[1]
    moved = duplicate["time_refs"][0]
    moved["memory_id"] = retained["memory_id"]
    retained["time_refs"].append(moved)
    store.update(retained)
    assert len(store.get(retained["memory_id"])["time_refs"]) == 2


def test_same_non_person_entity_text_stays_in_separate_memories(tmp_path):
    class Graph:
        def __init__(self): self.writes = []
        def write_memory(self, memory): self.writes.append(memory)
    class Vector:
        def upsert(self, *_): pass
    graph, vector = Graph(), Vector()
    store = GraphVectorMemoryStore(SQLiteMemoryStore(tmp_path / "memory.db"), graph, vector)
    first = make_event("first", "2026-09-16T12:00:00+05:45", "2026-09-16T12:01:00+05:45")
    second = make_event("second", "2026-09-17T12:00:00+05:45", "2026-09-17T12:01:00+05:45")
    apply_batch_response([
        result("new", ["first"], entities=[{"type": "EVENT", "text": "Mark's birthday"}]),
        result("new", ["second"], summary="Another birthday update.", entities=[{"type": "EVENT", "text": "Mark's birthday"}]),
    ], {"first": first, "second": second}, store)
    assert len(store.list_all()) == 2
    assert graph.writes[0]["entities"] == graph.writes[1]["entities"]


def test_normalization_never_keeps_patient_as_participant():
    normalized = _normalize_aliases(
        [{"action": "new", "source_events": ["E1"], "participants": ["Rita", "Sam"]}],
        {"E1": "event-id"}, {"Rita", "Sam"}, {"event-id": ["Rita", "Sam"]}, "Rita",
    )[0]
    assert normalized["participants"] == ["Sam"]


def test_different_commitment_makers_do_not_merge_only_because_time_is_related(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    first = make_event("first-plan", "2026-09-16T12:00:00+05:45", "2026-09-16T12:01:00+05:45")
    first["segments"][0]["text"] = "unenrolled_7 will take Rita to the doctor next Monday."
    first_item = result("new", ["first-plan"], category="commitment", time_refs=[{
        "ref_key": "T1", "phrase": "next Monday", "tense": "future", "from_event": "first-plan",
        "topic": "Rita doctor appointment", "concerned_with": "Rita", "commitment_by": "unenrolled_7",
        "what_was_mentioned": "unenrolled_7 will take Rita to the doctor next Monday.",
    }])
    apply_batch_response([first_item], {"first-plan": first}, store)
    existing = store.list_all()[0]
    second = make_event("second-plan", "2026-09-17T12:00:00+05:45", "2026-09-17T12:01:00+05:45")
    second["segments"][0]["text"] = "unenrolled_8 will take Rita to the doctor next month."
    second_item = result("merge", ["second-plan"], category="commitment", target_memory_id=existing["memory_id"], what_was_updated="A separate plan was stated.", time_refs=[{
        "ref_key": "T1", "phrase": "next month", "tense": "future", "from_event": "second-plan",
        "topic": "Rita doctor appointment", "concerned_with": "Rita", "commitment_by": "unenrolled_8",
        "what_was_mentioned": "unenrolled_8 will take Rita to the doctor next month.",
    }])
    apply_batch_response([second_item], {"second-plan": second}, store)
    assert len(store.list_all()) == 2


def test_explicit_reschedule_creates_a_superseded_timeref_chain(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    first = make_event("monday", "2026-11-19T12:00:00+05:45", "2026-11-19T12:01:00+05:45")
    first["segments"][0]["text"] = "Sam will pick up Rita to meet Alex next Monday."
    first_item = result("new", ["monday"], category="commitment", time_refs=[{
        "ref_key": "T1", "phrase": "next Monday", "tense": "future", "from_event": "monday",
        "topic": "meeting Alex", "concerned_with": "Rita", "commitment_by": "Sam",
        "what_was_mentioned": "Sam will pick up Rita next Monday to meet Alex.",
    }])
    apply_batch_response([first_item], {"monday": first}, store)
    existing = store.list_all()[0]
    second = make_event("tuesday", "2026-11-20T12:00:00+05:45", "2026-11-20T12:01:00+05:45")
    second["segments"][0]["text"] = "Sam rescheduled the pickup for Rita to next Tuesday to meet Alex."
    second_item = result("merge", ["tuesday"], category="commitment", target_memory_id=existing["memory_id"],
        summary="Sam will pick up Rita to meet Alex next Tuesday.",
        what_was_updated="The pickup date was rescheduled from Monday to next Tuesday.", time_refs=[{
            "ref_key": "T1", "phrase": "next Tuesday", "tense": "future", "from_event": "tuesday",
            "topic": "pickup for meeting", "concerned_with": "Rita", "commitment_by": "Sam",
            "what_was_mentioned": "Sam will pick up Rita next Tuesday to meet Alex.",
        }], supersedes_time_refs={existing["time_refs"][0]["time_ref_id"]: "T1"})
    apply_batch_response([second_item], {"tuesday": second}, store)
    memories = store.list_all()
    assert len(memories) == 1
    old, new = sorted(memories[0]["time_refs"], key=lambda ref: ref["created_at"])
    assert old["status"] == "superseded" and old["superseded_by"] == new["time_ref_id"]
    assert new["status"] == "active" and new["topic"] == "meeting Alex"

def test_prompt_and_batch_trigger(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    event = make_event("one", "2026-01-01T12:00:00+00:00", "2026-01-01T12:05:00+00:00")
    existing, events = build_batch_prompt([event], store)
    assert existing == "(none)" and "[E1] event_start=" in events
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


def test_participants_are_limited_to_speakers_or_the_known_patient():
    result = _normalize_aliases(
        [{"action": "new", "source_events": ["E1"], "participants": ["sam", "Rita", "invented"]}],
        {"E1": "event-id"},
        {"Sam", "Rita"},
    )[0]
    assert result["participants"] == ["Sam", "Rita"]
    assert result["needs_review"] is True


def test_unknown_observed_speaker_is_forced_into_the_normalized_response():
    result = _normalize_aliases(
        [{"action": "new", "source_events": ["E1"], "participants": ["Sam"]}],
        {"E1": "event-id"},
        {"Sam", "unenrolled_9"},
        {"event-id": ["Sam", "unenrolled_9"]},
    )[0]
    assert result["participants"] == ["Sam", "unenrolled_9"]

def test_prompt_defaults_to_separate_memories_and_requires_a_genuine_merge_link(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    prompt = make_batch_prompt([make_event("event-uuid", "2026-01-01T12:00:00+00:00", "2026-01-01T12:01:00+00:00")], store)
    prompt_without_line_breaks = " ".join(prompt.split())
    assert "Create a NEW memory unless" in prompt
    assert "Sharing a participant" in prompt
    assert "same specific fact, relationship, plan, or ongoing matter" in prompt_without_line_breaks


def test_active_prompt_keeps_participants_and_adds_timeref_and_patient_rules(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    prompt = make_batch_prompt([
        make_event("event-uuid", "2026-01-01T12:00:00+00:00", "2026-01-01T12:01:00+00:00")
    ], store)
    assert '"participants": ["<people present or speaking>"]' in prompt
    assert '"time_refs":' in prompt
    assert "Do not assume a speaker is the patient" in prompt
    assert "PERSON, or RELATION" in prompt
    assert "Never list the patient as a participant" in prompt
    assert "supersedes_time_refs" in prompt

def test_parse_llm_json_response_accepts_markdown_fenced_json():
    assert parse_llm_json_response("```json\n[{\"action\": \"new\"}]\n```") == [{"action": "new"}]

def test_absent_third_party_provenance_is_preserved(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    event = {**make_event("sarah-visit", "2026-09-07T09:00:00+00:00", "2026-09-07T09:01:00+00:00"), "participants": ["Dad"]}
    third_party = result("new", ["sarah-visit"], participants=["Dad"], about=[{"person": "Sarah", "reported_by": "Dad", "what_was_said": "Coming 2026-09-08 instead of 2026-09-07."}])
    apply_batch_response([third_party], {"sarah-visit": event}, store)
    assert store.list_all()[0]["about"] == third_party["about"]
    assert _validate_results([third_party], {"sarah-visit"})[0]["about"] == third_party["about"]


def test_concern_person_is_not_retained_as_a_duplicate_entity(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    event = make_event("sarah-news", "2026-09-07T09:00:00+00:00", "2026-09-07T09:01:00+00:00")
    item = result("new", ["sarah-news"], about=[{
        "person": "Sarah", "reported_by": "kris", "what_was_said": "Sarah received a promotion.",
    }], entities=[{"type": "PERSON", "text": " Sarah "}])
    apply_batch_response([item], {"sarah-news": event}, store)
    memory = store.list_all()[0]
    assert memory["about"][0]["type"] == "PERSON"
    assert memory["entities"] == []

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
    apply_batch_response([result("merge", ["second"], target_memory_id=memory_id, summary="Kris prefers green tea.", what_was_updated="Preference changed to green tea.")], {"first": first, "second": second}, store)

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
