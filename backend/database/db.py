import hashlib
import os
import secrets
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

_DB_LOCK = threading.Lock()
_DB_INITIALISED = False

DATABASE_PATH = Path(__file__).resolve().parents[1] / "uploads" / "database" / "reconnect.db"
FACE_MATCH_THRESHOLD = 0.50


def _enable_wal_once():
    """Enable WAL journal mode exactly once at module load."""
    try:
        DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DATABASE_PATH, timeout=10, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.close()
    except sqlite3.OperationalError:
        pass


_enable_wal_once()


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


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
        # Caregivers table
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS caregivers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                phone TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                caregiver_type TEXT NOT NULL,
                family_relation TEXT,
                profession TEXT,
                created_at TEXT NOT NULL
            )
            """
        )

        # Patients table
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                caregiver_id INTEGER NOT NULL REFERENCES caregivers(id) ON DELETE CASCADE,
                full_name TEXT NOT NULL,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                dob TEXT NOT NULL,
                phone TEXT NOT NULL,
                email TEXT NOT NULL,
                address TEXT NOT NULL,
                medical_information TEXT,
                created_at TEXT NOT NULL
            )
            """
        )

        # Auth sessions
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS auth_sessions (
                token TEXT PRIMARY KEY,
                actor_type TEXT NOT NULL,
                actor_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT
            )
            """
        )

        # Enrolled identities
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
                face_image BLOB,
                patient_id INTEGER REFERENCES patients(id)
            )
            """
        )

        # Unenrolled identities
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS unenrolled_identities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                face_embedding BLOB,
                voice_embedding BLOB,
                face_image BLOB,
                resolved_to INTEGER REFERENCES enrolled_identities(id),
                patient_id INTEGER REFERENCES patients(id)
            )
            """
        )

        # Pending unknowns
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS pending_unknowns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
                image_blob BLOB NOT NULL,
                timestamp TEXT NOT NULL,
                resolved_to INTEGER REFERENCES enrolled_identities(id)
            )
            """
        )

        # Migrations: Ensure all columns exist on older tables
        enrolled_cols = [r[1] for r in connection.execute("PRAGMA table_info(enrolled_identities)").fetchall()]
        if "face_image" not in enrolled_cols:
            connection.execute("ALTER TABLE enrolled_identities ADD COLUMN face_image BLOB")
        if "patient_id" not in enrolled_cols:
            connection.execute("ALTER TABLE enrolled_identities ADD COLUMN patient_id INTEGER REFERENCES patients(id)")

        unenrolled_cols = [r[1] for r in connection.execute("PRAGMA table_info(unenrolled_identities)").fetchall()]
        if "face_image" not in unenrolled_cols:
            connection.execute("ALTER TABLE unenrolled_identities ADD COLUMN face_image BLOB")
        if "resolved_to" not in unenrolled_cols:
            connection.execute("ALTER TABLE unenrolled_identities ADD COLUMN resolved_to INTEGER REFERENCES enrolled_identities(id)")
        if "patient_id" not in unenrolled_cols:
            connection.execute("ALTER TABLE unenrolled_identities ADD COLUMN patient_id INTEGER REFERENCES patients(id)")

    _DB_INITIALISED = True


# ---------------------------------------------------------------------------
# Auth / Caregiver / Patient helpers
# ---------------------------------------------------------------------------

def create_caregiver(
    name: str,
    email: str,
    phone: str,
    password: str,
    caregiver_type: str,
    family_relation: Optional[str] = None,
    profession: Optional[str] = None,
) -> int:
    init_database()
    with _DB_LOCK, _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO caregivers (name, email, phone, password_hash, caregiver_type, family_relation, profession, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name.strip(),
                email.strip().lower(),
                phone.strip(),
                _hash_password(password),
                caregiver_type.strip(),
                family_relation.strip() if family_relation else None,
                profession.strip() if profession else None,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        return cursor.lastrowid


def create_patient(
    caregiver_phone: str,
    full_name: str,
    username: str,
    password: str,
    dob: str,
    phone: str,
    email: str,
    address: str,
    medical_information: str = "",
) -> Optional[int]:
    init_database()
    with _DB_LOCK, _connect() as connection:
        cg_row = connection.execute(
            "SELECT id FROM caregivers WHERE phone = ? LIMIT 1",
            (caregiver_phone.strip(),),
        ).fetchone()
        if not cg_row:
            return None
        caregiver_id = cg_row["id"]

        cursor = connection.execute(
            """
            INSERT INTO patients (caregiver_id, full_name, username, password_hash, dob, phone, email, address, medical_information, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                caregiver_id,
                full_name.strip(),
                username.strip(),
                _hash_password(password),
                dob.strip(),
                phone.strip(),
                email.strip().lower(),
                address.strip(),
                medical_information.strip(),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        return cursor.lastrowid


def authenticate_actor(actor_type: str, identifier: str, password: str) -> Optional[dict]:
    init_database()
    pw_hash = _hash_password(password)
    with _DB_LOCK, _connect() as connection:
        if actor_type == "caregiver":
            row = connection.execute(
                """
                SELECT id, name, email, phone, caregiver_type, family_relation, profession
                FROM caregivers
                WHERE (email = ? OR phone = ?) AND password_hash = ?
                LIMIT 1
                """,
                (identifier.strip().lower(), identifier.strip(), pw_hash),
            ).fetchone()
            if row:
                d = dict(row)
                d["actor_type"] = "caregiver"
                return d
        elif actor_type == "patient":
            row = connection.execute(
                """
                SELECT id, caregiver_id, full_name, username, dob, phone, email, address, medical_information
                FROM patients
                WHERE (username = ? OR email = ? OR phone = ?) AND password_hash = ?
                LIMIT 1
                """,
                (identifier.strip(), identifier.strip().lower(), identifier.strip(), pw_hash),
            ).fetchone()
            if row:
                d = dict(row)
                d["actor_type"] = "patient"
                d["name"] = d["full_name"]
                return d
    return None


def create_auth_session(actor_type: str, actor_id: int) -> str:
    init_database()
    token = secrets.token_hex(32)
    with _DB_LOCK, _connect() as connection:
        connection.execute(
            """
            INSERT INTO auth_sessions (token, actor_type, actor_id, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (token, actor_type, actor_id, datetime.now(timezone.utc).isoformat()),
        )
    return token


def get_session_actor(token: str) -> Optional[dict]:
    init_database()
    with _DB_LOCK, _connect() as connection:
        s_row = connection.execute(
            "SELECT token, actor_type, actor_id, created_at FROM auth_sessions WHERE token = ? LIMIT 1",
            (token.strip(),),
        ).fetchone()
        if not s_row:
            return None

        actor_type = s_row["actor_type"]
        actor_id = s_row["actor_id"]

        if actor_type == "caregiver":
            cg = connection.execute(
                "SELECT id, name, email, phone, caregiver_type, family_relation, profession FROM caregivers WHERE id = ? LIMIT 1",
                (actor_id,),
            ).fetchone()
            if not cg:
                return None
            res = dict(cg)
            res["actor_type"] = "caregiver"
            res["actor_id"] = actor_id
            return res
        elif actor_type == "patient":
            pt = connection.execute(
                "SELECT id, caregiver_id, full_name, username, dob, phone, email, address, medical_information FROM patients WHERE id = ? LIMIT 1",
                (actor_id,),
            ).fetchone()
            if not pt:
                return None
            res = dict(pt)
            res["actor_type"] = "patient"
            res["actor_id"] = actor_id
            res["name"] = res["full_name"]
            res["patient_code"] = f"PT{actor_id:04d}"
            return res
    return None


def delete_auth_session(token: str) -> None:
    init_database()
    with _DB_LOCK, _connect() as connection:
        connection.execute("DELETE FROM auth_sessions WHERE token = ?", (token.strip(),))


def get_caregiver_patients(caregiver_id: int) -> list[dict]:
    init_database()
    with _DB_LOCK, _connect() as connection:
        rows = connection.execute(
            "SELECT id, full_name, username, dob, phone, email, address, medical_information, created_at FROM patients WHERE caregiver_id = ? ORDER BY id",
            (caregiver_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def caregiver_owns_patient(caregiver_id: int, patient_id: int) -> bool:
    init_database()
    with _DB_LOCK, _connect() as connection:
        row = connection.execute(
            "SELECT id FROM patients WHERE id = ? AND caregiver_id = ? LIMIT 1",
            (patient_id, caregiver_id),
        ).fetchone()
        if row is not None:
            return True
        p_count = connection.execute("SELECT count(*) FROM patients").fetchone()[0]
        if p_count == 0:
            return True
        return False


# ---------------------------------------------------------------------------
# Enrolled & Unenrolled identities
# ---------------------------------------------------------------------------

def get_all_identities(patient_id: Optional[int] = None):
    init_database()
    with _DB_LOCK, _connect() as connection:
        if patient_id is not None:
            return connection.execute(
                "SELECT * FROM enrolled_identities WHERE patient_id = ? OR patient_id IS NULL ORDER BY id",
                (patient_id,),
            ).fetchall()
        return connection.execute(
            "SELECT * FROM enrolled_identities ORDER BY id"
        ).fetchall()


def get_patient_enrolled_people(patient_id: int) -> list[dict]:
    init_database()
    with _DB_LOCK, _connect() as connection:
        rows = connection.execute(
            "SELECT id, name, relation, COALESCE(patient_id, ?) AS patient_id, (face_image IS NOT NULL) AS has_photo FROM enrolled_identities WHERE patient_id = ? OR patient_id IS NULL ORDER BY id",
            (patient_id, patient_id),
        ).fetchall()
        return [dict(r) for r in rows]


def get_identity_by_name(name: str):
    init_database()
    with _DB_LOCK, _connect() as connection:
        return connection.execute(
            "SELECT * FROM enrolled_identities WHERE name = ? ORDER BY id LIMIT 1",
            (name,),
        ).fetchone()


def get_patient_identity_by_name(name: str, patient_id: int):
    init_database()
    with _DB_LOCK, _connect() as connection:
        return connection.execute(
            "SELECT * FROM enrolled_identities WHERE name = ? AND (patient_id = ? OR patient_id IS NULL) ORDER BY id LIMIT 1",
            (name, patient_id),
        ).fetchone()


def find_matching_face(embedding, threshold=FACE_MATCH_THRESHOLD, patient_id: Optional[int] = None):
    candidate = np.asarray(embedding, dtype=np.float32).reshape(-1)
    cand_norm = np.linalg.norm(candidate)
    if cand_norm == 0:
        return None, -1.0
    best_row = None
    best_similarity = -1.0

    for row in get_all_identities(patient_id=patient_id):
        stored = blob_to_embedding(row["face_embedding"]).reshape(-1)
        stored_norm = np.linalg.norm(stored)
        if stored.shape != candidate.shape or stored_norm == 0:
            continue
        similarity = float(np.dot(candidate, stored) / (cand_norm * stored_norm))
        if similarity > best_similarity:
            best_similarity = similarity
            best_row = row

    if best_row is not None and best_similarity >= threshold:
        return best_row, best_similarity
    return None, best_similarity


def create_identity(name: str, relation: str, face_embedding, face_image=None, patient_id: Optional[int] = None):
    init_database()
    with _DB_LOCK, _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO enrolled_identities
                (timestamp, name, relation, face_embedding, voice_embedding, face_image, patient_id)
            VALUES (?, ?, ?, ?, NULL, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                name,
                relation,
                embedding_to_blob(face_embedding),
                image_to_blob(face_image),
                patient_id,
            ),
        )
        return cursor.lastrowid


def delete_identity(identity_id: int) -> None:
    init_database()
    with _DB_LOCK, _connect() as connection:
        connection.execute("DELETE FROM enrolled_identities WHERE id = ?", (identity_id,))


def update_voice_embedding(identity_id: int, voice_embedding):
    init_database()
    with _DB_LOCK, _connect() as connection:
        connection.execute(
            "UPDATE enrolled_identities SET voice_embedding = ? WHERE id = ?",
            (embedding_to_blob(voice_embedding), identity_id),
        )


def append_voice_embedding(identity_id: int, voice_embedding, max_templates=5):
    init_database()
    with _DB_LOCK, _connect() as connection:
        row = connection.execute(
            "SELECT voice_embedding FROM enrolled_identities WHERE id = ?",
            (identity_id,),
        ).fetchone()
        existing = row["voice_embedding"] if row else None
        current = blob_to_embedding(existing) if existing else np.array([], dtype=np.float32)
        combined = np.concatenate((current, np.asarray(voice_embedding, dtype=np.float32)))

        dimension = 192
        num_embeddings = len(combined) // dimension
        if num_embeddings > max_templates:
            combined = combined[-(max_templates * dimension):]

        connection.execute(
            "UPDATE enrolled_identities SET voice_embedding = ? WHERE id = ?",
            (embedding_to_blob(combined), identity_id),
        )


def resolve_unenrolled_identity(unenrolled_id: int, enrolled_id: int):
    init_database()
    with _DB_LOCK, _connect() as connection:
        connection.execute(
            "UPDATE unenrolled_identities SET resolved_to = ? WHERE id = ?",
            (int(enrolled_id), int(unenrolled_id)),
        )
        row = connection.execute(
            "SELECT voice_embedding FROM unenrolled_identities WHERE id = ?",
            (int(unenrolled_id),),
        ).fetchone()
        voice_embedding = row["voice_embedding"] if row else None

    if voice_embedding:
        for emb in voice_blob_to_embeddings(voice_embedding):
            append_voice_embedding(enrolled_id, emb, max_templates=5)


def create_unenrolled_identity(face_embedding=None, voice_embedding=None, face_image=None, patient_id=None):
    init_database()
    with _DB_LOCK, _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO unenrolled_identities (timestamp, face_embedding, voice_embedding, face_image, patient_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                embedding_to_blob(face_embedding) if face_embedding is not None else None,
                embedding_to_blob(voice_embedding) if voice_embedding is not None else None,
                image_to_blob(face_image),
                patient_id,
            ),
        )
        return cursor.lastrowid


def get_unenrolled_identity(un_id: int):
    init_database()
    with _DB_LOCK, _connect() as connection:
        row = connection.execute(
            "SELECT * FROM unenrolled_identities WHERE id = ?",
            (int(un_id),),
        ).fetchone()
        return dict(row) if row else None


def update_unenrolled_face(un_id: int, face_embedding, face_image=None):
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


def update_unenrolled_voice(un_id: int, voice_embedding):
    init_database()
    with _DB_LOCK, _connect() as connection:
        connection.execute(
            "UPDATE unenrolled_identities SET voice_embedding = ? WHERE id = ?",
            (embedding_to_blob(voice_embedding), int(un_id)),
        )


def get_identity_image(identity_id: int, as_cv2=False):
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


def get_patient_identity_image(identity_id: int, patient_id: Optional[int] = None, as_cv2=False):
    init_database()
    with _DB_LOCK, _connect() as connection:
        if patient_id is not None:
            row = connection.execute(
                "SELECT face_image FROM enrolled_identities WHERE id = ? AND (patient_id = ? OR patient_id IS NULL)",
                (int(identity_id), patient_id),
            ).fetchone()
        else:
            row = connection.execute(
                "SELECT face_image FROM enrolled_identities WHERE id = ?",
                (int(identity_id),),
            ).fetchone()
        if not row or not row["face_image"]:
            return None
        blob = row["face_image"]
        return blob_to_image(blob) if as_cv2 else blob


def get_unenrolled_identity_image(un_id: int, as_cv2=False):
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


def find_matching_enrolled_voice(embedding, threshold=0.50, patient_id: Optional[int] = None):
    candidate = np.asarray(embedding, dtype=np.float32).reshape(-1)
    cand_norm = np.linalg.norm(candidate)
    if cand_norm == 0:
        return None, -1.0
    best_row = None
    best_similarity = -1.0
    with _DB_LOCK, _connect() as connection:
        if patient_id is not None:
            rows = connection.execute(
                "SELECT id, name, relation, voice_embedding FROM enrolled_identities WHERE voice_embedding IS NOT NULL AND (patient_id = ? OR patient_id IS NULL)",
                (patient_id,),
            ).fetchall()
        else:
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


def find_matching_unenrolled_voice(embedding, threshold=0.42, patient_id: Optional[int] = None):
    candidate = np.asarray(embedding, dtype=np.float32).reshape(-1)
    cand_norm = np.linalg.norm(candidate)
    if cand_norm == 0:
        return None, -1.0
    best_row = None
    best_similarity = -1.0
    with _DB_LOCK, _connect() as connection:
        if patient_id is not None:
            rows = connection.execute(
                "SELECT * FROM unenrolled_identities WHERE voice_embedding IS NOT NULL AND (patient_id = ? OR patient_id IS NULL)",
                (patient_id,),
            ).fetchall()
        else:
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


# ---------------------------------------------------------------------------
# Pending unknowns helpers
# ---------------------------------------------------------------------------

def create_pending_unknown(patient_id: int, image_input) -> Optional[int]:
    init_database()
    blob = image_to_blob(image_input)
    if not blob:
        return None
    with _DB_LOCK, _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO pending_unknowns (patient_id, image_blob, timestamp)
            VALUES (?, ?, ?)
            """,
            (patient_id, blob, datetime.now(timezone.utc).isoformat()),
        )
        return cursor.lastrowid


def get_pending_unknowns(patient_id: int) -> list[dict]:
    init_database()
    with _DB_LOCK, _connect() as connection:
        # Fetch from unenrolled_identities (where vision pipeline saves faces)
        rows1 = connection.execute(
            """
            SELECT id, timestamp, timestamp AS created_at
            FROM unenrolled_identities
            WHERE face_image IS NOT NULL AND resolved_to IS NULL
            ORDER BY id DESC LIMIT 50
            """
        ).fetchall()

        # Also fetch from pending_unknowns table
        rows2 = connection.execute(
            """
            SELECT id, timestamp, timestamp AS created_at
            FROM pending_unknowns
            WHERE (patient_id = ? OR patient_id IS NULL) AND resolved_to IS NULL
            ORDER BY id DESC LIMIT 50
            """,
            (patient_id,),
        ).fetchall()

        combined = [dict(r) for r in rows1]
        seen_ids = {r["id"] for r in combined}
        for r in rows2:
            if r["id"] not in seen_ids:
                combined.append(dict(r))
        return combined


def get_pending_unknown_image(pending_id: int, patient_id: Optional[int] = None, as_cv2=False):
    init_database()
    with _DB_LOCK, _connect() as connection:
        # Check unenrolled_identities first
        row = connection.execute(
            "SELECT face_image FROM unenrolled_identities WHERE id = ?",
            (pending_id,),
        ).fetchone()
        if row and row["face_image"]:
            blob = row["face_image"]
            return blob_to_image(blob) if as_cv2 else blob

        # Check pending_unknowns table
        row2 = connection.execute(
            "SELECT image_blob FROM pending_unknowns WHERE id = ?",
            (pending_id,),
        ).fetchone()
        if row2 and row2["image_blob"]:
            blob = row2["image_blob"]
            return blob_to_image(blob) if as_cv2 else blob
        return None


def resolve_pending_unknown(pending_id: int, patient_id: int, identity_id: int) -> None:
    init_database()
    with _DB_LOCK, _connect() as connection:
        connection.execute(
            "UPDATE unenrolled_identities SET resolved_to = ? WHERE id = ?",
            (identity_id, pending_id),
        )
        connection.execute(
            "UPDATE pending_unknowns SET resolved_to = ? WHERE id = ?",
            (identity_id, pending_id),
        )
