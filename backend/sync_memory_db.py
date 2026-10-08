"""
Utility script to synchronize and backfill existing SQLite memories
into Neo4j (Graph) and FAISS (Vector) stores.

Usage:
    python sync_memory_db.py
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env
BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(BACKEND_DIR / ".env")

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.memory.store import SQLiteMemoryStore
from app.memory.graph_store import GraphStore, VectorStore, GraphVectorMemoryStore


def main():
    print("=" * 60)
    print("🔄 RECONNECT Memory Database Synchronizer (SQLite -> Neo4j + FAISS)")
    print("=" * 60)

    sqlite_store = SQLiteMemoryStore()
    memories = sqlite_store.list_all()
    print(f"📊 Found {len(memories)} existing memories in SQLite ({sqlite_store.path}).")

    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    if not neo4j_password:
        print("\n⚠️ NEO4J_PASSWORD is not set in backend/.env!")
        print("Please configure NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD in your backend/.env file.")
        print("Example:\n  NEO4J_URI=bolt://localhost:7687\n  NEO4J_USER=neo4j\n  NEO4J_PASSWORD=your_password\n  RECONNECT_GRAPH_MEMORY=1")
        return

    try:
        print(f"\n🔌 Connecting to Neo4j at {neo4j_uri}...")
        graph_store = GraphStore(neo4j_uri, neo4j_user, neo4j_password)
        print("✅ Neo4j connection successful.")

        print("📦 Initializing FAISS Vector Store...")
        vector_store = VectorStore()
        print(f"✅ FAISS index loaded at {vector_store.directory}.")

        print("\n🚀 Starting backfill into Neo4j and FAISS...")
        projected_store = GraphVectorMemoryStore(sqlite_store, graph_store, vector_store)
        backfilled_count = projected_store.backfill()

        print(f"\n✨ Backfill complete! Synchronized {backfilled_count}/{len(memories)} memories into Neo4j & FAISS.")

        graph_store.close()
    except Exception as e:
        print(f"\n❌ Error during synchronization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
