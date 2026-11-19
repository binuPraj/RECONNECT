#!/usr/bin/env python3
"""
Migration script to synchronize current SQLite data into Neo4j graph database.

Migrates:
1. Patients (nodes + identity references)
2. Enrolled identities & their relationships to patients (e.g. Spouse, Son, Friend)
3. Active unenrolled identities (if any exist)
4. Memories (Memory nodes, PRESENT_AT, CONCERNS, MENTIONS edges)
5. FAISS Vector index backfill

Usage:
    backend/.venv/bin/python backend/migrate_to_graph.py
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(BACKEND_DIR / ".env")

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import sqlite3
from database.db import DATABASE_PATH
from app.memory.store import SQLiteMemoryStore
from app.memory.graph_store import GraphStore, VectorStore


def migrate():
    print("=" * 65)
    print("🔄 RECONNECT -> Neo4j Graph & Vector Migration")
    print("=" * 65)

    neo4j_uri = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "hellyes123#")

    if not neo4j_password:
        print("❌ NEO4J_PASSWORD is not set in backend/.env!")
        return

    print(f"🔌 Connecting to Neo4j at {neo4j_uri}...")
    try:
        graph_store = GraphStore(neo4j_uri, neo4j_user, neo4j_password)
        graph_store.ensure_schema()
        print("✅ Connected to Neo4j and verified constraints.")
    except Exception as exc:
        print(f"❌ Failed to connect to Neo4j: {exc}")
        return

    # 1. Migrate Patients and Enrolled People from reconnect.db
    if not DATABASE_PATH.exists():
        print(f"⚠️ SQLite database not found at {DATABASE_PATH}")
    else:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Patients
        patients = cur.execute("SELECT id, full_name FROM patients").fetchall()
        print(f"\n👤 Migrating {len(patients)} patient(s)...")
        patient_map = {}
        for p in patients:
            patient_id, full_name = p["id"], p["full_name"]
            patient_map[patient_id] = full_name
            with graph_store.driver.session() as session:
                session.run("""
                    MERGE (p:Person {name_key: $name_key})
                    ON CREATE SET p.name = $name, p.role = 'patient'
                    SET p.name = $name, p.role = 'patient',
                        p.person_id = $person_id,
                        p.identity_table = 'patients',
                        p.identity_id = $patient_id,
                        p.patient_id = $patient_id
                """, name=full_name, name_key=graph_store.person_key(full_name),
                     person_id=f"patient:{patient_id}:patient", patient_id=patient_id)
            print(f"   ✓ Patient #{patient_id}: {full_name}")

        # Enrolled Identities & Relationships
        enrolled = cur.execute(
            "SELECT id, name, relation, patient_id FROM enrolled_identities"
        ).fetchall()
        print(f"\n👥 Migrating {len(enrolled)} enrolled identity/identities...")
        for row in enrolled:
            eid, name, relation, pid = row["id"], row["name"], row["relation"], row["patient_id"]
            pname = patient_map.get(pid, "Patient")
            graph_store.upsert_caregiver_relationship(
                relative_name=name,
                relation=relation or "relative",
                patient_name=pname,
                enrolled_identity_id=eid,
                patient_id=pid or 1,
            )
            print(f"   ✓ Enrolled #{eid}: {name} ({relation}) -> {pname}")

        # Unenrolled Identities
        unenrolled = cur.execute(
            "SELECT id, patient_id FROM unenrolled_identities WHERE resolved_to IS NULL"
        ).fetchall()
        print(f"\n❓ Migrating {len(unenrolled)} active unenrolled identity/identities...")
        for row in unenrolled:
            uid, pid = row["id"], row["patient_id"]
            uname = f"unenrolled_{uid}"
            graph_store.upsert_biometric_person(
                name=uname,
                identity_table="unenrolled_identities",
                identity_id=uid,
                patient_id=pid,
            )
            print(f"   ✓ Unenrolled #{uid}: {uname}")

        conn.close()

    # 2. Migrate Memories from memories.db
    print("\n🧠 Migrating memories...")
    sqlite_memory_store = SQLiteMemoryStore()
    memories = sqlite_memory_store.list_all()
    print(f"Found {len(memories)} memories in SQLite.")

    vector_store = None
    try:
        vector_store = VectorStore()
        print("✅ FAISS Vector Store initialized.")
    except Exception as exc:
        print(f"⚠️ Vector store skipped ({exc})")

    for mem in memories:
        graph_store.write_memory(mem)
        if vector_store:
            vector_store.upsert(mem["memory_id"], mem["summary"])
        print(f"   ✓ Memory: {mem['memory_id']} ({mem.get('topic') or mem.get('category')})")

    graph_store.close()
    print("\n" + "=" * 65)
    print("✨ Migration completed successfully!")
    print("=" * 65)


if __name__ == "__main__":
    migrate()
