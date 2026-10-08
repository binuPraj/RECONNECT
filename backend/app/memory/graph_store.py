"""Optional Neo4j + FAISS projection for durable memory retrieval."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VECTOR_DIR = PROJECT_ROOT / "uploads" / "data" / "memory" / "vector"
LOGGER = logging.getLogger(__name__)


class GraphStore:
    """Neo4j graph writer and structured lookup reader."""
    def __init__(self, uri: str, user: str, password: str) -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError as error:
            raise RuntimeError("Install neo4j to enable graph memory storage") from error
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    @classmethod
    def from_environment(cls) -> "GraphStore | None":
        uri, user, password = (os.getenv(key) for key in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"))
        return cls(uri, user, password) if uri and user and password else None

    def close(self) -> None:
        self.driver.close()

    def write_memory(self, memory: dict[str, Any]) -> None:
        with self.driver.session() as session:
            session.run("""
                MERGE (m:Memory {memory_id: $memory_id})
                SET m.summary=$summary, m.category=$category, m.importance=$importance,
                    m.emotion=$emotion, m.topic=$topic, m.source_events=$source_events,
                    m.updated_at=$updated_at, m.created_at=coalesce(m.created_at, $created_at)
            """, **{key: memory.get(key) for key in ("memory_id", "summary", "category", "importance", "emotion", "topic", "source_events", "updated_at", "created_at")})
            for name in memory.get("participants", []):
                session.run("""
                    MATCH (m:Memory {memory_id:$memory_id})
                    MERGE (p:Person {name:$name})
                    MERGE (p)-[:PRESENT_AT]->(m)
                """, name=name, memory_id=memory["memory_id"])
            for entry in memory.get("about", []):
                session.run("""
                    MATCH (m:Memory {memory_id:$memory_id})
                    MERGE (p:Person {name:$person})
                    MERGE (p)-[r:CONCERNS]->(m)
                    SET r.reported_by=$reported_by, r.what_was_said=$what_was_said
                """, person=entry["person"], reported_by=entry["reported_by"], what_was_said=entry["what_was_said"], memory_id=memory["memory_id"])
            for entity in memory.get("entities", []):
                session.run("""
                    MATCH (m:Memory {memory_id:$memory_id})
                    MERGE (e:Entity {type:$type, text:$text})
                    MERGE (m)-[:MENTIONS]->(e)
                """, type=entity["type"], text=entity["text"], memory_id=memory["memory_id"])

    def get_direct(self, person_name: str, limit: int = 5) -> list[dict]:
        with self.driver.session() as session:
            return [dict(row) for row in session.run("""
                MATCH (p:Person {name:$name})-[r:CONCERNS]->(m:Memory)
                RETURN m.memory_id AS memory_id, m.summary AS summary, m.updated_at AS updated_at,
                       r.reported_by AS reported_by, r.what_was_said AS what_was_said
                ORDER BY m.updated_at DESC LIMIT $limit
            """, name=person_name, limit=limit)]

    def get_by_ids(self, memory_ids: list[str]) -> list[dict]:
        if not memory_ids:
            return []
        with self.driver.session() as session:
            rows = session.run("""
                MATCH (m:Memory) WHERE m.memory_id IN $ids
                OPTIONAL MATCH (p:Person)-[:PRESENT_AT]->(m)
                OPTIONAL MATCH (subject:Person)-[r:CONCERNS]->(m)
                RETURN m.memory_id AS memory_id, m.summary AS summary, m.category AS category,
                  collect(DISTINCT p.name) AS participants,
                  collect(DISTINCT CASE WHEN subject IS NULL THEN NULL ELSE
                    {person:subject.name, reported_by:r.reported_by, what_was_said:r.what_was_said} END) AS about
            """, ids=memory_ids)
            found = {row["memory_id"]: dict(row) for row in rows}
        return [found[memory_id] for memory_id in memory_ids if memory_id in found]


class VectorStore:
    """Persistent FAISS semantic index returning memory IDs, never memory content."""
    def __init__(self, directory: Path = DEFAULT_VECTOR_DIR, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError("Install faiss-cpu and sentence-transformers to enable vector memory search") from error
        self.faiss, self.embedder = faiss, SentenceTransformer(model_name)
        self.directory = directory; self.directory.mkdir(parents=True, exist_ok=True)
        self.index_path, self.map_path = directory / "memories.faiss", directory / "id_map.json"
        self.id_map = json.loads(self.map_path.read_text()) if self.map_path.exists() else {}
        self.next_int = max((int(value) for value in self.id_map.values()), default=-1) + 1
        self.index = faiss.read_index(str(self.index_path)) if self.index_path.exists() else faiss.IndexIDMap(faiss.IndexFlatL2(384))

    def _save(self) -> None:
        self.faiss.write_index(self.index, str(self.index_path))
        self.map_path.write_text(json.dumps(self.id_map))

    def upsert(self, memory_id: str, summary_text: str) -> None:
        import numpy as np
        vector = self.embedder.encode([summary_text])[0].astype("float32").reshape(1, -1)
        int_id = self.id_map.setdefault(memory_id, self.next_int)
        if int_id == self.next_int: self.next_int += 1
        self.index.remove_ids(np.array([int_id], dtype="int64"))
        self.index.add_with_ids(vector, np.array([int_id], dtype="int64")); self._save()

    def search(self, query_text: str, k: int = 5) -> list[str]:
        import numpy as np
        if not self.id_map: return []
        vector = self.embedder.encode([query_text])[0].astype("float32").reshape(1, -1)
        _, int_ids = self.index.search(vector, min(k, len(self.id_map)))
        reverse = {int(value): key for key, value in self.id_map.items()}
        return [reverse[int(value)] for value in int_ids[0] if int(value) in reverse]


class GraphVectorMemoryStore:
    """Best-effort Neo4j/FAISS projection layered on top of primary SQLite storage."""
    def __init__(self, primary, graph: GraphStore, vector: VectorStore) -> None:
        self.primary, self.graph, self.vector, self.path = primary, graph, vector, primary.path
    def _project(self, memory: dict) -> bool:
        try:
            write_memory_full(memory, self.graph, self.vector)
            return True
        except Exception:
            # The optional search projection must never make primary SQLite writes fail.
            LOGGER.exception("[memory] Neo4j/FAISS projection failed for memory %s", memory["memory_id"])
            return False
    def create(self, memory: dict) -> None:
        self.primary.create(memory)
        self._project(memory)
    def update(self, memory: dict) -> None:
        self.primary.update(memory)
        self._project(memory)
    def get(self, memory_id: str): return self.primary.get(memory_id)
    def get_recent(self, participant: str, limit: int = 3): return self.primary.get_recent(participant, limit)
    def list_all(self): return self.primary.list_all()
    def backfill(self) -> int:
        """Mirror pre-existing SQLite memories when projection is first enabled."""
        return sum(self._project(memory) for memory in self.primary.list_all())


def write_memory_full(memory: dict, graph_store: GraphStore, vector_store: VectorStore) -> None:
    graph_store.write_memory(memory)
    vector_store.upsert(memory["memory_id"], memory["summary"])

def get_direct(graph_store: GraphStore, person_name: str, limit: int = 5) -> list[dict]:
    return graph_store.get_direct(person_name, limit)

def get_open_ended(query_text: str, graph_store: GraphStore, vector_store: VectorStore, k: int = 5) -> list[dict]:
    return graph_store.get_by_ids(vector_store.search(query_text, k))
