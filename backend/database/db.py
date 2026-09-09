import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_DB_LOCK = threading.Lock()
_DB_INITIALISED = False

DATABASE_PATH = Path(__file__).resolve().parent / "reconnect.db"
FACE_MATCH_THRESHOLD = 0.50


def _enable_wal_once():
    """Enable WAL journal mode exactly once at module load.

    WAL mode requires an *exclusive* lock on the database file, so it must
    be set before any other connections are opened.  Calling it from inside
    _connect() (which can be called concurrently) causes 'database is locked'.
    Errors are silently ignored – the database still works in the default
    DELETE journal mode if WAL cannot be activated.
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH, timeout=10, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.close()
    except sqlite3.OperationalError:
        pass  # Already in WAL or some other process holds the DB; non-fatal


_enable_wal_once()


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


def image_to_blob(image_input):
    """Converts an OpenCV image (numpy array), a file path, or raw bytes into a JPEG binary blob."""
    if image_input is None:
        return None
    if isinstance(image_input, (bytes, bytearray)):
        return bytes(image_input)
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if p.exists() and p.is_file():
            with open(p, "rb") as f:
                return f.read()
        return None
    if isinstance(image_input, np.ndarray):
        if image_input.size == 0:
            return None
        import cv2
        success, encoded = cv2.imencode(".jpg", image_input)
        if success:
            return encoded.tobytes()
    return None


def blob_to_image(blob_data):
    """Decodes JPEG/PNG bytes from SQLite into an OpenCV BGR image (numpy array)."""
    if not blob_data:
        return None
    import cv2
    buf = np.frombuffer(blob_data, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


@contextmanager
def _connect():
    """Open a thread-safe SQLite connection with automatic closing and busy timeout."""
    conn = sqlite3.connect(DATABASE_PATH, timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_database():
    global _DB_INITIALISED
    if _DB_INITIALISED:
        return
    with _DB_LOCK, _connect() as connection:
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
                voice_embedding BLOB,
                face_image BLOB
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS unenrolled_identities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                face_embedding BLOB,
                voice_embedding BLOB,
                face_image BLOB
            )
            """
        )

        # Migration: ensure face_image column exists on existing tables
        enrolled_cols = [r[1] for r in connection.execute("PRAGMA table_info(enrolled_identities)").fetchall()]
        if "face_image" not in enrolled_cols:
            connection.execute("ALTER TABLE enrolled_identities ADD COLUMN face_image BLOB")

        unenrolled_cols = [r[1] for r in connection.execute("PRAGMA table_info(unenrolled_identities)").fetchall()]
        if "face_image" not in unenrolled_cols:
            connection.execute("ALTER TABLE unenrolled_identities ADD COLUMN face_image BLOB")
            
        if "resolved_to" not in unenrolled_cols:
            connection.execute("ALTER TABLE unenrolled_identities ADD COLUMN resolved_to INTEGER REFERENCES enrolled_identities(id)")

    _DB_INITIALISED = True


def get_all_identities():
    init_database()
    with _DB_LOCK, _connect() as connection:
        return connection.execute(
            "SELECT * FROM enrolled_identities ORDER BY id"
        ).fetchall()


def get_identity_by_name(name):
    init_database()
    with _DB_LOCK, _connect() as connection:
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


def create_identity(name, relation, face_embedding, face_image=None):
    init_database()
    with _DB_LOCK, _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO enrolled_identities
                (timestamp, name, relation, face_embedding, voice_embedding, face_image)
            VALUES (?, ?, ?, ?, NULL, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                name,
                relation,
                embedding_to_blob(face_embedding),
                image_to_blob(face_image),
            ),
        )
        return cursor.lastrowid


def update_voice_embedding(identity_id, voice_embedding):
    init_database()
    with _DB_LOCK, _connect() as connection:
        connection.execute(
            "UPDATE enrolled_identities SET voice_embedding = ? WHERE id = ?",
            (embedding_to_blob(voice_embedding), identity_id),
        )


def append_voice_embedding(identity_id, voice_embedding, max_templates=5):
    init_database()
    with _DB_LOCK, _connect() as connection:
        row = connection.execute(
            "SELECT voice_embedding FROM enrolled_identities WHERE id = ?",
            (identity_id,),
        ).fetchone()
        existing = row["voice_embedding"] if row else None
        current = blob_to_embedding(existing) if existing else np.array([], dtype=np.float32)
        combined = np.concatenate((current, np.asarray(voice_embedding, dtype=np.float32)))
        
        # Enforce max templates cap
        dimension = 192
        num_embeddings = len(combined) // dimension
        if num_embeddings > max_templates:
            combined = combined[-(max_templates * dimension):]
            
        connection.execute(
            "UPDATE enrolled_identities SET voice_embedding = ? WHERE id = ?",
            (embedding_to_blob(combined), identity_id),
        )

def resolve_unenrolled_identity(unenrolled_id, enrolled_id):
    """Mark an unenrolled identity as resolved to an enrolled identity, and merge templates."""
    init_database()
    with _DB_LOCK, _connect() as connection:
        connection.execute(
            "UPDATE unenrolled_identities SET resolved_to = ? WHERE id = ?",
            (int(enrolled_id), int(unenrolled_id))
        )
        row = connection.execute(
            "SELECT voice_embedding FROM unenrolled_identities WHERE id = ?",
            (int(unenrolled_id),)
        ).fetchone()
        voice_embedding = row["voice_embedding"] if row else None
        
    if voice_embedding:
        for emb in voice_blob_to_embeddings(voice_embedding):
            append_voice_embedding(enrolled_id, emb, max_templates=5)

# ---------- Unenrolled identities helpers ----------

def create_unenrolled_identity(face_embedding=None, voice_embedding=None, face_image=None):
    """Create a new unenrolled identity with optional face and voice embeddings and face image.
    Returns the generated row id.
    """
    init_database()
    with _DB_LOCK, _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO unenrolled_identities (timestamp, face_embedding, voice_embedding, face_image)
            VALUES (?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                embedding_to_blob(face_embedding) if face_embedding is not None else None,
                embedding_to_blob(voice_embedding) if voice_embedding is not None else None,
                image_to_blob(face_image),
            ),
        )
        return cursor.lastrowid

def get_unenrolled_identity(un_id):
    init_database()
    with _DB_LOCK, _connect() as connection:
        row = connection.execute(
            "SELECT * FROM unenrolled_identities WHERE id = ?",
            (int(un_id),),
        ).fetchone()
        return dict(row) if row else None

def update_unenrolled_face(un_id, face_embedding, face_image=None):
    """Updates face embedding and optionally the best face image for an unenrolled identity."""
    init_database()
    img_blob = image_to_blob(face_image)
    with _DB_LOCK, _connect() as connection:
        if img_blob is not None:
            connection.execute(
                "UPDATE unenrolled_identities SET face_embedding = ?, face_image = ? WHERE id = ?",
                (embedding_to_blob(face_embedding), img_blob, int(un_id)),
            )
        else:
            connection.execute(
                "UPDATE unenrolled_identities SET face_embedding = ? WHERE id = ?",
                (embedding_to_blob(face_embedding), int(un_id)),
            )

def update_unenrolled_voice(un_id, voice_embedding):
    init_database()
    with _DB_LOCK, _connect() as connection:
        connection.execute(
            "UPDATE unenrolled_identities SET voice_embedding = ? WHERE id = ?",
            (embedding_to_blob(voice_embedding), un_id),
        )

def get_identity_image(identity_id, as_cv2=False):
    """Fetch the face image for an enrolled identity.
    Returns raw bytes by default, or an OpenCV numpy image if as_cv2=True.
    """
    init_database()
    with _DB_LOCK, _connect() as connection:
        row = connection.execute(
            "SELECT face_image FROM enrolled_identities WHERE id = ?",
            (int(identity_id),),
        ).fetchone()
        if not row or not row["face_image"]:
            return None
        blob = row["face_image"]
        return blob_to_image(blob) if as_cv2 else blob

def get_unenrolled_identity_image(un_id, as_cv2=False):
    """Fetch the face image for an unenrolled identity.
    Returns raw bytes by default, or an OpenCV numpy image if as_cv2=True.
    """
    init_database()
    with _DB_LOCK, _connect() as connection:
        row = connection.execute(
            "SELECT face_image FROM unenrolled_identities WHERE id = ?",
            (int(un_id),),
        ).fetchone()
        if not row or not row["face_image"]:
            return None
        blob = row["face_image"]
        return blob_to_image(blob) if as_cv2 else blob

def find_matching_enrolled_voice(embedding, threshold=0.50):
    """Search enrolled identities for a voice embedding similarity above threshold.
    Returns (row_dict, similarity) or (None, -1.0).
    """
    candidate = np.asarray(embedding, dtype=np.float32).reshape(-1)
    cand_norm = np.linalg.norm(candidate)
    if cand_norm == 0:
        return None, -1.0
    best_row = None
    best_similarity = -1.0
    with _DB_LOCK, _connect() as connection:
        rows = connection.execute(
            "SELECT id, name, relation, voice_embedding FROM enrolled_identities WHERE voice_embedding IS NOT NULL"
        ).fetchall()
        for row in rows:
            voice_blob = row["voice_embedding"]
            if voice_blob is None:
                continue
            for stored_emb in voice_blob_to_embeddings(voice_blob):
                stored = np.asarray(stored_emb, dtype=np.float32).reshape(-1)
                stored_norm = np.linalg.norm(stored)
                if stored.shape != candidate.shape or stored_norm == 0:
                    continue
                sim = float(np.dot(candidate, stored) / (cand_norm * stored_norm))
                if sim > best_similarity:
                    best_similarity = sim
                    best_row = dict(row)
    if best_row is not None and best_similarity >= threshold:
        return best_row, best_similarity
    return None, best_similarity


def find_matching_unenrolled_voice(embedding, threshold=0.42):
    """Search unenrolled identities for a voice embedding similarity above threshold.
    Returns (row_dict, similarity) or (None, -1.0).
    """
    candidate = np.asarray(embedding, dtype=np.float32).reshape(-1)
    cand_norm = np.linalg.norm(candidate)
    if cand_norm == 0:
        return None, -1.0
    best_row = None
    best_similarity = -1.0
    with _DB_LOCK, _connect() as connection:
        rows = connection.execute("SELECT * FROM unenrolled_identities WHERE voice_embedding IS NOT NULL").fetchall()
        for row in rows:
            voice_blob = row["voice_embedding"]
            if voice_blob is None:
                continue
            for stored_emb in voice_blob_to_embeddings(voice_blob):
                stored = np.asarray(stored_emb, dtype=np.float32).reshape(-1)
                stored_norm = np.linalg.norm(stored)
                if stored.shape != candidate.shape or stored_norm == 0:
                    continue
                sim = float(np.dot(candidate, stored) / (cand_norm * stored_norm))
                if sim > best_similarity:
                    best_similarity = sim
                    best_row = dict(row)
    if best_row is not None and best_similarity >= threshold:
        return best_row, best_similarity
    return None, best_similarity
