"""Retrieval-only settings. Perception and storage modules are not modified."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]

# A patient may pause before asking a follow-up, so five minutes is the normal
# live session gap.  Set RETRIVAL_SESSION_GAP_SECONDS=60 when deliberately
# testing session expiry.
SESSION_GAP_SECONDS = int(os.getenv("RETRIVAL_SESSION_GAP_SECONDS", "300"))
EVENT_WINDOW_MINUTES = int(os.getenv("RETRIVAL_EVENT_WINDOW_MINUTES", "5"))
VECTOR_THRESHOLD = float(os.getenv("RETRIVAL_VECTOR_THRESHOLD", "0.35"))
RETRIVAL_PORT = int(os.getenv("RETRIVAL_PORT", "8001"))

DATA_DIR = PROJECT_ROOT / "uploads" / "data" / "retrival"
DB_PATH = DATA_DIR / "queries.db"

REFUSAL = "I don't have a record of that."
NO_RECENT_EVENTS = "I haven't heard anything recently."
UNRECOGNISED = "I don't recognise this person yet."
AMBIGUOUS_PERSON = "Which {relation} do you mean? Please ask again with her name."
FOLLOWUP_EMPTY = "Please tell me what you mean."
VAGUE_QUESTION = "Please be more specific about your question. You can name a person, a time, or a place."
SEVERAL_FACES = "I can see more than one person. Please keep one face in view and ask again."
NO_FACE = "I could not see a face. Keep the face centered, close, and in focus."
NEED_CAMERA = "I need the camera to answer who this is."

INTENTS = ("who_is", "what_happening", "recall", "schedule", "other")
MEMORY_CATEGORIES = (
    "identity", "relationship", "commitment", "significant_event", "routine", "preference",
)
PARSER_FIELDS = (
    "intent", "person_ref", "time_ref", "occasion_ref",
    "location_ref", "topic_ref", "tense", "category_hint", "is_followup",
)
