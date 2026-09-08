import numpy as np

from app.audio_process.recognition import SessionSpeakerMemory


def test_failed_resolution_reports_gallery_and_session_diagnostics():
    memory = SessionSpeakerMemory(
        continuity_threshold=0.58,
        session_embed_threshold=0.60,
    )
    memory.remember("session-a", "binu", np.array([1.0, 0.0]), confirmed=True)

    identity, score, reason, details = memory.resolve_with_details(
        "session-a",
        np.array([0.5, np.sqrt(0.75)]),
        "binu",
        0.40,
    )

    assert identity is None
    assert score == 0.40
    assert reason is None
    assert details == {
        "gallery_best_previously_confirmed": True,
        "max_session_embedding_score": 0.5,
        "max_session_embedding_identity": "binu",
    }
