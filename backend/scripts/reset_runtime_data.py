"""Remove RECONNECT test/runtime data while preserving login accounts.

This is intentionally a command-line utility rather than an API endpoint:
stop Uvicorn and the Flutter app before using it, then run it with ``--yes``.
It preserves the ``caregivers``, ``patients``, and ``auth_sessions`` tables
(and therefore their password hashes, caregiver-to-patient links, and active
sign-ins), but removes all enrollment, audio, event, memory, vector, and graph
data.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
load_dotenv(BACKEND_ROOT / ".env")

RUNTIME_DB = BACKEND_ROOT / "uploads" / "database" / "reconnect.db"
EVENT_DB = BACKEND_ROOT / "uploads" / "data" / "events" / "events.db"
MEMORY_DB = BACKEND_ROOT / "uploads" / "data" / "memory" / "memories.db"
VECTOR_DIR = BACKEND_ROOT / "uploads" / "data" / "memory" / "vector"


def _delete_rows(path: Path, tables: tuple[str, ...]) -> None:
    if not path.exists():
        return
    with sqlite3.connect(path) as connection:
        available = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        for table in tables:
            if table in available:
                connection.execute(f'DELETE FROM "{table}"')


def _remove_contents(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _reset_runtime_database() -> None:
    # Do not touch caregivers, patients, or auth_sessions: those tables hold
    # login credentials, profile records, ownership links, and active tokens.
    _delete_rows(
        RUNTIME_DB,
        (
            "enrolled_identities",
            "unenrolled_identities",
            "pending_unknowns",
        ),
    )


def _reset_events() -> None:
    _delete_rows(
        EVENT_DB,
        (
            "events",
            "grouping_meta",
            "consolidated_events",
            "memory_consolidation_meta",
            "memory_consolidation_audit",
        ),
    )
    # Recreate the counter expected by the event grouper after clearing it.
    if EVENT_DB.exists():
        with sqlite3.connect(EVENT_DB) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO grouping_meta(key, value) VALUES('unconsumed_event_count', '0')"
            )


def _reset_memories() -> None:
    _delete_rows(MEMORY_DB, ("time_refs", "memories"))
    _remove_contents(VECTOR_DIR)


def _reset_files() -> None:
    # Generated recordings, captured video uploads, enrollment media, and
    # locally generated biometric gallery artifacts. Source code is untouched.
    for directory in (
        BACKEND_ROOT / "uploads" / "streams",
        BACKEND_ROOT / "uploads" / "enrollment_images",
        BACKEND_ROOT / "uploads" / "enrollment_audio",
        BACKEND_ROOT / "data" / "gallery",
    ):
        _remove_contents(directory)


def _reset_neo4j() -> bool:
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    if not (uri and user and password):
        print("Neo4j: skipped (NEO4J_URI, NEO4J_USER, or NEO4J_PASSWORD is not configured).")
        return False
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            session.run("MATCH (node) DETACH DELETE node").consume()
    finally:
        driver.close()
    print("Neo4j: all graph nodes and relationships removed.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yes", action="store_true",
        help="Confirm permanent deletion of runtime data. Login accounts are preserved.",
    )
    args = parser.parse_args()
    if not args.yes:
        parser.error("Refusing to delete data without --yes. Stop the backend first.")

    _reset_runtime_database()
    _reset_events()
    _reset_memories()
    _reset_files()
    _reset_neo4j()
    print("Reset complete. Caregiver and patient login credentials were preserved.")


if __name__ == "__main__":
    main()
