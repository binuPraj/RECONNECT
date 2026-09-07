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


def voice_blob_to_embeddings(blob, dimension=192):
    values = blob_to_embedding(blob)
    if values.size % dimension != 0:
        return []
    return [
        values[index:index + dimension].copy()
        for index in range(0, values.size, dimension)
    ]


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
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS unenrolled_identities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                face_embedding BLOB,
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


def get_identity_by_name(name):
    init_database()
    with _connect() as connection:
        return connection.execute(
            "SELECT * FROM enrolled_identities WHERE name = ? "
            "ORDER BY id LIMIT 1",
            (name,),
        ).fetchone()


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


def append_voice_embedding(identity_id, voice_embedding):
    init_database()
    with _connect() as connection:
        row = connection.execute(
            "SELECT voice_embedding FROM enrolled_identities WHERE id = ?",
            (identity_id,),
        ).fetchone()
        existing = row["voice_embedding"] if row else None
        current = blob_to_embedding(existing) if existing else np.array([], dtype=np.float32)
        combined = np.concatenate((current, np.asarray(voice_embedding, dtype=np.float32)))
        connection.execute(
            "UPDATE enrolled_identities SET voice_embedding = ? WHERE id = ?",
            (embedding_to_blob(combined), identity_id),
        )

# ---------- Unenrolled identities helpers ----------

def create_unenrolled_identity(face_embedding=None, voice_embedding=None):
    """Create a new unenrolled identity with optional face and voice embeddings.
    Returns the generated row id.
    """
    init_database()
    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO unenrolled_identities (timestamp, face_embedding, voice_embedding)
            VALUES (?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                embedding_to_blob(face_embedding) if face_embedding is not None else None,
                embedding_to_blob(voice_embedding) if voice_embedding is not None else None,
            ),
        )
        return cursor.lastrowid

def update_unenrolled_face(un_id, face_embedding):
    init_database()
    with _connect() as connection:
        connection.execute(
            "UPDATE unenrolled_identities SET face_embedding = ? WHERE id = ?",
            (embedding_to_blob(face_embedding), un_id),
        )

def update_unenrolled_voice(un_id, voice_embedding):
    init_database()
    with _connect() as connection:
        connection.execute(
            "UPDATE unenrolled_identities SET voice_embedding = ? WHERE id = ?",
            (embedding_to_blob(voice_embedding), un_id),
        )

def find_matching_unenrolled_voice(embedding, threshold=0.60):
    """Search unenrolled identities for a voice embedding similarity above threshold.
    Returns (row_dict, similarity) or (None, -1.0).
    """
    candidate = np.asarray(embedding, dtype=np.float32).reshape(-1)
    best_row = None
    best_similarity = -1.0
    with _connect() as connection:
        rows = connection.execute("SELECT * FROM unenrolled_identities").fetchall()
        for row in rows:
            voice_blob = row["voice_embedding"]
            if voice_blob is None:
                continue
            for stored_emb in voice_blob_to_embeddings(voice_blob):
                stored = np.asarray(stored_emb, dtype=np.float32).reshape(-1)
                if stored.shape != candidate.shape:
                    continue
                sim = float(np.dot(candidate, stored) / (np.linalg.norm(candidate) * np.linalg.norm(stored)))
                if sim > best_similarity:
                    best_similarity = sim
                    best_row = dict(row)
    if best_row is not None and best_similarity >= threshold:
        return best_row, best_similarity
    return None, best_similarity
