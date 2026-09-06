import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


DATABASE_PATH = Path(__file__).resolve().parent / "reconnect.db"
FACE_MATCH_THRESHOLD = 0.50


def embedding_to_blob(embedding):
    return np.asarray(embedding, dtype=np.float32).tobytes()


def blob_to_embedding(blob):
    return np.frombuffer(blob, dtype=np.float32).copy()


def _connect():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_database():
    with _connect() as connection:
        existing = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name = 'enrolled_identities'"
        ).fetchone()

        legacy = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name = 'identities'"
        ).fetchone()

        if not existing and legacy:
            connection.execute(
                "ALTER TABLE identities RENAME TO enrolled_identities"
            )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS enrolled_identities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                name TEXT NOT NULL,
                relation TEXT NOT NULL,
                face_embedding BLOB NOT NULL,
                voice_embedding BLOB
            )
            """
        )


def get_all_identities():
    init_database()
    with _connect() as connection:
        return connection.execute(
            "SELECT * FROM enrolled_identities ORDER BY id"
        ).fetchall()


def find_matching_face(embedding, threshold=FACE_MATCH_THRESHOLD):
    candidate = np.asarray(embedding, dtype=np.float32)
    best_row = None
    best_similarity = -1.0

    for row in get_all_identities():
        stored = blob_to_embedding(row["face_embedding"])
        if stored.shape != candidate.shape:
            continue
        similarity = float(np.dot(candidate, stored))
        if similarity > best_similarity:
            best_similarity = similarity
            best_row = row

    if best_row is not None and best_similarity >= threshold:
        return best_row, best_similarity
    return None, best_similarity


def create_identity(name, relation, face_embedding):
    init_database()
    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO enrolled_identities
                (timestamp, name, relation, face_embedding, voice_embedding)
            VALUES (?, ?, ?, ?, NULL)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                name,
                relation,
                embedding_to_blob(face_embedding),
            ),
        )
        return cursor.lastrowid


def update_voice_embedding(identity_id, voice_embedding):
    init_database()
    with _connect() as connection:
        connection.execute(
            "UPDATE enrolled_identities SET voice_embedding = ? WHERE id = ?",
            (embedding_to_blob(voice_embedding), identity_id),
        )
