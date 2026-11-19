from app.identity_resolution import _reconcile_memory


def test_confirmed_participant_removes_self_concern_and_renames_reporter():
    memory = {
        "participants": ["unenrolled_1"],
        "about": [
            {"person": "Priscilla", "reported_by": "unenrolled_1", "what_was_said": "Priscilla is a nice girl."},
            {"person": "Sarah", "reported_by": "unenrolled_1", "what_was_said": "Sarah will visit."},
        ],
        "entities": [
            {"type": "PERSON", "text": "unenrolled_1"},
            {"type": "PERSON", "text": "Sarah"},
        ],
        "summary": "unenrolled_1 said Priscilla is a nice girl and Sarah will visit.",
    }

    assert _reconcile_memory(memory, "unenrolled_1", "Priscilla")
    assert memory["participants"] == ["Priscilla"]
    assert memory["about"] == [
        {"person": "Sarah", "reported_by": "Priscilla", "what_was_said": "Sarah will visit."}
    ]
    assert memory["entities"][0]["text"] == "Priscilla"
    assert "unenrolled_1" not in memory["summary"]


def test_case_insensitive_replacement_deduplicates_participants():
    memory = {
        "participants": ["UNENROLLED_1", "Priscilla"],
        "about": [],
        "entities": [],
        "summary": "No anonymous name here.",
    }
    assert _reconcile_memory(memory, "unenrolled_1", "Priscilla")
    assert memory["participants"] == ["Priscilla"]
