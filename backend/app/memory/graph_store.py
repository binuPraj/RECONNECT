"""Optional Neo4j + FAISS projection for durable memory retrieval."""
from __future__ import annotations

import json
import logging
import os
import re
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

    @staticmethod
    def person_key(name: str) -> str:
        """Stable identity key: spacing and letter case never make a new person."""
        return re.sub(r"\s+", " ", name.strip()).casefold()

    def ensure_schema(self) -> None:
        """Enforce one canonical Person node for each normalized name."""
        with self.driver.session() as session:
            session.run(
                "CREATE CONSTRAINT person_name_key_unique IF NOT EXISTS "
                "FOR (p:Person) REQUIRE p.name_key IS UNIQUE"
            ).consume()
            session.run(
                "CREATE CONSTRAINT person_id_unique IF NOT EXISTS "
                "FOR (p:Person) REQUIRE p.person_id IS UNIQUE"
            ).consume()
            session.run("CREATE CONSTRAINT memory_id_unique IF NOT EXISTS FOR (m:Memory) REQUIRE m.memory_id IS UNIQUE").consume()
            session.run("CREATE CONSTRAINT timeref_id_unique IF NOT EXISTS FOR (t:TimeRef) REQUIRE t.time_ref_id IS UNIQUE").consume()
            session.run("CREATE INDEX timeref_window IF NOT EXISTS FOR (t:TimeRef) ON (t.status, t.start, t.end)").consume()
            session.run("CREATE INDEX memory_patient IF NOT EXISTS FOR (m:Memory) ON (m.patient_id)").consume()

    def migrate_person_entities(self) -> int:
        """Compatibility no-op.

        ``Entity(type=PERSON)`` now intentionally represents an unresolved
        person phrase such as "Sam's daughter" and must stay an Entity rather
        than becoming an observed/enrolled Person node.
        """
        return 0

    def remove_person_mentions(self) -> int:
        """Remove only legacy unlinked Person mentions.

        A confirmed enrolled person can now legitimately receive MENTIONS
        migrated from an Entity(type=PERSON) at enrollment time.
        """
        with self.driver.session() as session:
            result = session.execute_write(lambda tx: tx.run("""
                MATCH (m:Memory)-[mention:MENTIONS]->(person:Person)
                WHERE person.person_id IS NULL AND coalesce(person.role, '') <> 'patient'
                DELETE mention
                RETURN count(*) AS deleted
            """).single()["deleted"])
        return result

    def restore_entity_mentions(self) -> int:
        """Keep the established MENTIONS edge for non-person Entity nodes."""
        with self.driver.session() as session:
            result = session.execute_write(lambda tx: tx.run("""
                MATCH (m:Memory)-[old:HAS_ENTITY]->(entity:Entity)
                MERGE (m)-[:MENTIONS]->(entity)
                DELETE old
                RETURN count(*) AS migrated
            """).single()["migrated"])
        return result

    @classmethod
    def from_environment(cls) -> "GraphStore | None":
        uri, user, password = (os.getenv(key) for key in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"))
        return cls(uri, user, password) if uri and user and password else None

    def close(self) -> None:
        self.driver.close()

    @staticmethod
    def identity_reference(
        *, identity_table: str, identity_id: int, patient_id: int | None
    ) -> dict[str, object]:
        """Return graph properties that point back to the biometric SQLite row."""
        prefix = "enrolled" if identity_table == "enrolled_identities" else "unenrolled"
        person_id = (
            f"patient:{patient_id}:{prefix}:{identity_id}"
            if patient_id is not None
            else f"{prefix}:{identity_id}"
        )
        return {
            "person_id": person_id,
            "identity_table": identity_table,
            "identity_id": int(identity_id),
            "patient_id": patient_id,
        }

    def upsert_biometric_person(
        self, *, name: str, identity_table: str, identity_id: int, patient_id: int | None
    ) -> None:
        """Attach a graph Person node to its exact enrolled/unenrolled SQLite row."""
        with self.driver.session() as session:
            session.run("""
                MERGE (p:Person {name_key:$name_key})
                ON CREATE SET p.name=$name
                SET p.name=$name, p += $identity
            """, name=name.strip(), name_key=self.person_key(name),
                 identity=self.identity_reference(
                     identity_table=identity_table,
                     identity_id=identity_id,
                     patient_id=patient_id,
                 )).consume()

    def upsert_patient(self, *, patient_id: int, patient_name: str) -> None:
        """Create the canonical wearer node before any memory references it."""
        if not patient_name.strip():
            return
        with self.driver.session() as session:
            session.run("""
                MERGE (patient:Person {name_key:$name_key})
                ON CREATE SET patient.name=$name
                SET patient.name=$name, patient.role='patient',
                    patient.person_id=$person_id, patient.identity_table='patients',
                    patient.identity_id=$patient_id, patient.patient_id=$patient_id
            """, name=patient_name.strip(), name_key=self.person_key(patient_name),
                 person_id=f"patient:{patient_id}:patient", patient_id=patient_id).consume()
            # Convert legacy Entity mentions of the patient's actual name into
            # direct references to this canonical patient Person node.
            session.run("""
                MATCH (patient:Person {person_id:$person_id})
                MATCH (m:Memory {patient_id:$patient_id})-[mention:MENTIONS]->(entity:Entity)
                WHERE toLower(trim(coalesce(entity.text, '')))=$name_key
                MERGE (m)-[:MENTIONS]->(patient)
                DELETE mention
            """, person_id=f"patient:{patient_id}:patient", patient_id=patient_id,
                 name_key=self.person_key(patient_name)).consume()

    def _identity_reference_for_name(self, name: str) -> dict[str, object]:
        """Best-effort lookup for graph nodes created while projecting memories."""
        unknown = re.fullmatch(r"unenrolled_(\d+)", name.strip(), flags=re.IGNORECASE)
        if unknown:
            from database.db import get_unenrolled_identity
            row = get_unenrolled_identity(int(unknown.group(1)))
            if row:
                return self.identity_reference(
                    identity_table="unenrolled_identities",
                    identity_id=row["id"], patient_id=row.get("patient_id"),
                )
        from database.db import get_identity_by_name
        row = get_identity_by_name(name)
        if row:
            return self.identity_reference(
                identity_table="enrolled_identities",
                identity_id=row["id"], patient_id=row["patient_id"],
            )
        return {}

    def _merge_person(self, session, name: str, **extra: object) -> None:
        identity = self._identity_reference_for_name(name)
        identity.update({key: value for key, value in extra.items() if value is not None})
        session.run("""
            MERGE (p:Person {name_key:$name_key})
            ON CREATE SET p.name=$name
            SET p += $identity
        """, name=name, name_key=self.person_key(name), identity=identity).consume()

    def _patient_relative_node(self, session, patient_id: int | None, relation: str) -> str | None:
        """Return one confirmed caregiver-created relative for this patient."""
        if patient_id is None or not relation.strip():
            return None
        rows = list(session.run("""
            MATCH (relative:Person)-[r:RELATED_TO]->(patient:Person {person_id:$patient_person_id})
            WHERE toLower(trim(r.type)) = $relation_key
            RETURN elementId(relative) AS node_id
            LIMIT 2
        """, patient_person_id=f"patient:{patient_id}:patient", relation_key=self.person_key(relation)))
        return rows[0]["node_id"] if len(rows) == 1 else None

    def _mention_entity(self, session, memory_id: str, entity_type: str, entity_text: str) -> None:
        """Project entities without ever sharing non-person Entity nodes.

        A PERSON phrase can be shared until it is enrolled.  PLACE, ITEM,
        EVENT, and unresolved RELATION entities always belong to exactly one
        memory, even if their display text is identical.
        """
        text_key = self.person_key(entity_text)
        if entity_type != "PERSON":
            existing = session.run("""
                MATCH (m:Memory {memory_id:$memory_id})-[:MENTIONS]->(e:Entity {
                    type:$type, text_key:$text_key, memory_id:$memory_id
                })
                RETURN elementId(e) AS node_id LIMIT 1
            """, memory_id=memory_id, type=entity_type, text_key=text_key).single()
            if existing:
                return
            session.run("""
                MATCH (m:Memory {memory_id:$memory_id})
                CREATE (e:Entity {type:$type, name:$text, text:$text, text_key:$text_key, memory_id:$memory_id})
                CREATE (m)-[:MENTIONS]->(e)
            """, memory_id=memory_id, type=entity_type, text=entity_text,
                 text_key=text_key).consume()
            return
        existing = session.run("""
            MATCH (e:Entity {type:$type})
            WHERE toLower(trim(coalesce(e.text, ''))) = $text_key
            RETURN elementId(e) AS node_id LIMIT 1
        """, type=entity_type, text_key=text_key).single()
        if existing:
            session.run("""
                MATCH (m:Memory {memory_id:$memory_id}), (e:Entity)
                WHERE elementId(e)=$entity_node_id
                SET e.text_key=$text_key, e.name=coalesce(e.name, e.text)
                MERGE (m)-[:MENTIONS]->(e)
            """, memory_id=memory_id, entity_node_id=existing["node_id"], text_key=text_key).consume()
            return
        session.run("""
            MATCH (m:Memory {memory_id:$memory_id})
            MERGE (e:Entity {type:$type, text_key:$text_key})
            ON CREATE SET e.text=$text, e.name=$text
            MERGE (m)-[:MENTIONS]->(e)
        """, memory_id=memory_id, type=entity_type, text_key=text_key, text=entity_text).consume()

    def split_shared_non_person_entities(self) -> int:
        """Migrate old shared non-PERSON entities to one node per memory."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (m:Memory)-[old:MENTIONS]->(e:Entity)
                WHERE e.type <> 'PERSON' AND coalesce(e.memory_id, '') <> m.memory_id
                CREATE (copy:Entity {
                    type:e.type, text:e.text,
                    text_key:coalesce(e.text_key, toLower(trim(e.text))),
                    memory_id:m.memory_id
                })
                CREATE (m)-[:MENTIONS]->(copy)
                DELETE old
                RETURN count(copy) AS split
            """).single()
        return int(result["split"] if result else 0)

    def reconcile_enrolled_identity_entities(
        self, *, patient_id: int, enrolled_identity_id: int, enrolled_name: str, relation: str
    ) -> None:
        """Move existing matching PERSON/RELATION entity links after enrollment."""
        with self.driver.session() as session:
            self._reconcile_relation_enrollment(
                session, patient_id=patient_id, relation=relation,
                enrolled_identity_id=enrolled_identity_id,
            )
            self._reconcile_person_entity_enrollment(
                session, patient_id=patient_id, enrolled_identity_id=enrolled_identity_id,
                enrolled_name=enrolled_name,
            )

    def backfill_display_names(self) -> None:
        """Give legacy graph nodes human-readable captions instead of IDs."""
        with self.driver.session() as session:
            session.run("MATCH (e:Entity) SET e.name=coalesce(e.name, e.text, e.text_key)").consume()
            session.run("MATCH (t:TimeRef) SET t.name=coalesce(t.name, t.phrase, t.time_ref_id)").consume()

    def write_memory(self, memory: dict[str, Any]) -> None:
        with self.driver.session() as session:
            session.run("""
                MERGE (m:Memory {memory_id: $memory_id})
                SET m.summary=$summary, m.category=$category, m.importance=$importance,
                    m.patient_id=$patient_id,
                    m.emotion=$emotion, m.topic=$topic, m.source_events=$source_events,
                    m.update_history_json=$update_history_json,
                    m.earliest_event_time=datetime($earliest_event_time),
                    m.latest_event_time=datetime($latest_event_time),
                    m.event_starts=[value IN $event_starts | datetime(value)],
                    m.needs_review=$needs_review,
                    m.updated_at=datetime($updated_at), m.created_at=coalesce(m.created_at, datetime($created_at))
            """, **{
                **{key: memory.get(key) for key in ("memory_id", "patient_id", "summary", "category", "importance", "emotion", "topic", "source_events", "earliest_event_time", "latest_event_time", "event_starts", "needs_review", "updated_at", "created_at")},
                "update_history_json": json.dumps(memory.get("update_history", [])),
            })
            concern_keys = [self.person_key(entry.get("person", "")) for entry in memory.get("about", []) if entry.get("person")]
            if concern_keys:
                # A person represented by CONCERNS must not retain a parallel
                # MENTIONS edge to an Entity node from an earlier projection.
                session.run("""
                    MATCH (m:Memory {memory_id:$memory_id})-[mention:MENTIONS]->(e:Entity)
                    WHERE toLower(trim(coalesce(e.text, ''))) IN $concern_keys
                    DELETE mention
                """, memory_id=memory["memory_id"], concern_keys=concern_keys).consume()
            # Rebuild these derived concern edges on every projection so a
            # later relation resolution cannot leave a stale generic Person.
            session.run("""
                MATCH (m:Memory {memory_id:$memory_id})
                OPTIONAL MATCH (:Person)-[concern:CONCERNS]->(m)
                DELETE concern
            """, memory_id=memory["memory_id"]).consume()
            patient_key = ""
            try:
                from database.db import get_patient_name
                patient_key = self.person_key(get_patient_name(memory["patient_id"]) or "") if memory.get("patient_id") is not None else ""
            except Exception:
                pass
            for name in memory.get("participants", []):
                display_name = str(name).strip()
                if not display_name or self.person_key(display_name) == patient_key:
                    continue
                self._merge_person(session, display_name)
                session.run("""
                    MATCH (m:Memory {memory_id:$memory_id}), (p:Person {name_key:$name_key})
                    MERGE (p)-[:PRESENT_AT]->(m)
                """, name_key=self.person_key(display_name), memory_id=memory["memory_id"])
            for entry in memory.get("about", []):
                display_name = str(entry["person"]).strip()
                if not display_name:
                    continue
                if str(entry.get("type", "PERSON")).upper() == "RELATION":
                    node_id = self._patient_relative_node(session, memory.get("patient_id"), display_name)
                    if node_id:
                        session.run("""
                            MATCH (m:Memory {memory_id:$memory_id}), (p:Person)
                            WHERE elementId(p)=$person_node_id
                            MERGE (p)-[r:CONCERNS]->(m)
                            SET r.reported_by=$reported_by, r.what_was_said=$what_was_said
                        """, memory_id=memory["memory_id"], person_node_id=node_id,
                           reported_by=entry["reported_by"], what_was_said=entry["what_was_said"]).consume()
                    else:
                        LOGGER.warning("[memory] Could not resolve patient relation '%s' for concern in %s", display_name, memory["memory_id"])
                    continue
                self._merge_person(session, display_name)
                session.run("""
                    MATCH (m:Memory {memory_id:$memory_id}), (p:Person {name_key:$person_key})
                    MERGE (p)-[r:CONCERNS]->(m)
                    SET r.reported_by=$reported_by, r.what_was_said=$what_was_said
                """, person=display_name, person_key=self.person_key(display_name), reported_by=entry["reported_by"], what_was_said=entry["what_was_said"], memory_id=memory["memory_id"])
            for entity in memory.get("entities", []):
                entity_type = str(entity["type"]).upper()
                entity_text = str(entity["text"]).strip()
                if not entity_text:
                    continue
                if entity_type == "PERSON" and self.person_key(entity_text) == patient_key:
                    session.run("""
                        MATCH (m:Memory {memory_id:$memory_id}),
                              (patient:Person {person_id:$patient_person_id})
                        MERGE (m)-[:MENTIONS]->(patient)
                    """, memory_id=memory["memory_id"],
                       patient_person_id=f"patient:{memory['patient_id']}:patient").consume()
                    continue
                if entity_type == "PERSON" and memory.get("patient_id") is not None:
                    try:
                        from database.db import get_patient_identity_by_name
                        enrolled = get_patient_identity_by_name(entity_text, memory["patient_id"])
                    except Exception:
                        enrolled = None
                    if enrolled:
                        self._merge_person(session, entity_text)
                        session.run("""
                            MATCH (m:Memory {memory_id:$memory_id}), (p:Person {name_key:$name_key})
                            OPTIONAL MATCH (m)-[old:MENTIONS]->(e:Entity {type:'PERSON'})
                            WHERE toLower(trim(coalesce(e.text, '')))=$name_key
                            DELETE old
                            MERGE (m)-[:MENTIONS]->(p)
                        """, memory_id=memory["memory_id"], name_key=self.person_key(entity_text)).consume()
                        continue
                if entity_type == "RELATION":
                    node_id = self._patient_relative_node(session, memory.get("patient_id"), entity_text)
                    if node_id:
                        session.run("""
                            MATCH (m:Memory {memory_id:$memory_id}), (p:Person)
                            WHERE elementId(p)=$person_node_id
                            MERGE (m)-[:REFERS_TO]->(p)
                        """, memory_id=memory["memory_id"], person_node_id=node_id).consume()
                        continue
                # PERSON here deliberately means an unresolved named entity
                # (for example "Sam's daughter"), not an enrolled Person node.
                self._mention_entity(session, memory["memory_id"], entity_type, entity_text)
            for ref in memory.get("time_refs", []):
                session.run("""
                    MATCH (m:Memory {memory_id:$memory_id})
                    MERGE (t:TimeRef {time_ref_id:$time_ref_id})
                    SET t.name=$phrase, t.phrase=$phrase, t.tense=$tense, t.anchor=datetime($anchor),
                        t.start=CASE WHEN $start IS NULL THEN NULL ELSE datetime($start) END,
                        t.end=CASE WHEN $end IS NULL THEN NULL ELSE datetime($end) END,
                        t.part=$part, t.granularity=$granularity, t.status=$status,
                        t.what_was_mentioned=$what_was_mentioned,
                        t.topic=$topic, t.concerned_with=$concerned_with,
                        t.commitment_by=$commitment_by,
                        t.created_at=datetime($created_at),
                        t.superseded_at=CASE WHEN $superseded_at IS NULL THEN NULL ELSE datetime($superseded_at) END,
                        t.last_confirmed_at=CASE WHEN $last_confirmed_at IS NULL THEN NULL ELSE datetime($last_confirmed_at) END
                    MERGE (m)-[:HAS_TIME]->(t)
                """, memory_id=memory["memory_id"], **{key: ref.get(key) for key in ("time_ref_id", "phrase", "tense", "anchor", "start", "end", "part", "granularity", "status", "what_was_mentioned", "topic", "concerned_with", "commitment_by", "created_at", "superseded_at", "last_confirmed_at")}).consume()
            # All TimeRef nodes must exist before their chain edges are made.
            # The old implementation attempted this inside the loop above; an
            # old ref commonly appears first, so its replacement did not yet
            # exist and Neo4j silently matched nothing.
            for ref in memory.get("time_refs", []):
                if ref.get("superseded_by"):
                    session.run("MATCH (old:TimeRef {time_ref_id:$old}), (new:TimeRef {time_ref_id:$new}) MERGE (old)-[:SUPERSEDED_BY]->(new)", old=ref["time_ref_id"], new=ref["superseded_by"]).consume()

    def prune_orphan_entities(self) -> int:
        """Remove only Entity nodes left disconnected after concern cleanup."""
        with self.driver.session() as session:
            row = session.run("""
                MATCH (e:Entity)
                WHERE NOT (e)--()
                DELETE e
                RETURN count(e) AS deleted
            """).single()
        return int(row["deleted"] if row else 0)

    def upsert_caregiver_relationship(
        self, *, relative_name: str, relation: str, patient_name: str,
        enrolled_identity_id: int, patient_id: int,
    ) -> None:
        """Link an enrolled relative to the patient without duplicate nodes or edges."""
        relative_name = relative_name.strip()
        patient_name = patient_name.strip()
        relation = relation.strip().casefold()
        if not relative_name or not patient_name or not relation:
            raise ValueError("Relative name, patient name, and relation are required.")
        with self.driver.session() as session:
            self._merge_person(
                session, relative_name,
                **self.identity_reference(
                    identity_table="enrolled_identities",
                    identity_id=enrolled_identity_id,
                    patient_id=patient_id,
                ),
            )
            session.run("""
                MERGE (relative:Person {name_key:$relative_key})
                MERGE (patient:Person {name_key:$patient_key})
                ON CREATE SET patient.name=$patient_name, patient.role='patient'
                SET patient.name=$patient_name, patient.role='patient',
                    patient.person_id=$patient_person_id,
                    patient.identity_table='patients', patient.identity_id=$patient_id,
                    patient.patient_id=$patient_id
                MERGE (relative)-[r:RELATED_TO]->(patient)
                SET r.type=$relation, r.source='caregiver'
            """, relative_name=relative_name, relative_key=self.person_key(relative_name),
                 patient_name=patient_name, patient_key=self.person_key(patient_name),
                 patient_person_id=f"patient:{patient_id}:patient",
                 patient_id=patient_id,
                 relation=relation).consume()
            self._reconcile_relation_enrollment(
                session,
                patient_id=patient_id,
                relation=relation,
                enrolled_identity_id=enrolled_identity_id,
            )
            self._reconcile_person_entity_enrollment(
                session,
                patient_id=patient_id,
                enrolled_identity_id=enrolled_identity_id,
                enrolled_name=relative_name,
            )

    def _reconcile_person_entity_enrollment(
        self, session, *, patient_id: int, enrolled_identity_id: int, enrolled_name: str
    ) -> None:
        """Move prior same-name PERSON entity links onto an enrolled identity.

        PERSON entities are retained while no biometric identity exists. Once a
        caregiver enrolls that exact name for this patient, their MENTIONS
        links point to the enrolled Person instead, so there is no duplicate
        entity/person representation.
        """
        target_person_id = self.identity_reference(
            identity_table="enrolled_identities", identity_id=enrolled_identity_id,
            patient_id=patient_id,
        )["person_id"]
        name_key = self.person_key(enrolled_name)
        session.run("""
            MATCH (target:Person {person_id:$target_person_id})
            MATCH (m:Memory {patient_id:$patient_id})-[mention:MENTIONS]->(entity:Entity {type:'PERSON'})
            WHERE toLower(trim(coalesce(entity.text, '')))=$name_key
            MERGE (m)-[:MENTIONS]->(target)
            DELETE mention
        """, target_person_id=target_person_id, patient_id=patient_id,
             name_key=name_key).consume()
        session.run("""
            MATCH (entity:Entity {type:'PERSON'})
            WHERE toLower(trim(coalesce(entity.text, '')))=$name_key
              AND NOT (entity)--()
            DELETE entity
        """, name_key=name_key).consume()

    def _reconcile_relation_enrollment(
        self, session, *, patient_id: int, relation: str, enrolled_identity_id: int
    ) -> None:
        """Replace prior patient-relation entity links with the enrolled Person.

        This runs only when a caregiver enrolls a relation. It transfers old
        memory links scoped to that patient, then removes the obsolete MENTIONS
        edges rather than altering unrelated entities or memories.
        """
        relation_key = self.person_key(relation)
        target_person_id = self.identity_reference(
            identity_table="enrolled_identities", identity_id=enrolled_identity_id,
            patient_id=patient_id,
        )["person_id"]
        # A previous RELATION entity (for example "daughter") becomes a direct
        # reference to the exact caregiver-enrolled Person node.
        session.run("""
            MATCH (target:Person {person_id:$target_person_id})
            MATCH (m:Memory {patient_id:$patient_id})-[mention:MENTIONS]->(entity:Entity {type:'RELATION'})
            WHERE toLower(trim(coalesce(entity.text, '')))=$relation_key
            MERGE (m)-[:REFERS_TO]->(target)
            DELETE mention
        """, target_person_id=target_person_id, patient_id=patient_id,
             relation_key=relation_key).consume()
        # Earlier versions could make a generic Person called "daughter" for
        # a relation concern. Transfer that concern to the enrolled identity.
        session.run("""
            MATCH (target:Person {person_id:$target_person_id})
            MATCH (legacy:Person {name_key:$relation_key})-[old:CONCERNS]->(m:Memory {patient_id:$patient_id})
            WHERE legacy.person_id IS NULL
            MERGE (target)-[new:CONCERNS]->(m)
            SET new.reported_by=old.reported_by, new.what_was_said=old.what_was_said
            DELETE old
        """, target_person_id=target_person_id, patient_id=patient_id,
             relation_key=relation_key).consume()
        # Delete only disconnected relation entities for this relation; any
        # still-linked node is preserved for a later enrollment/review.
        session.run("""
            MATCH (entity:Entity {type:'RELATION'})
            WHERE toLower(trim(coalesce(entity.text, '')))=$relation_key
              AND NOT (entity)--()
            DELETE entity
        """, relation_key=relation_key).consume()

    def rebuild_memory_people(self, memory: dict[str, Any]) -> None:
        """Replace derived people edges so removed/stale concerns cannot survive."""
        with self.driver.session() as session:
            session.run("""
                MATCH (m:Memory {memory_id:$memory_id})
                OPTIONAL MATCH (:Person)-[present:PRESENT_AT]->(m)
                DELETE present
            """, memory_id=memory["memory_id"]).consume()
            session.run("""
                MATCH (m:Memory {memory_id:$memory_id})
                OPTIONAL MATCH (:Person)-[concern:CONCERNS]->(m)
                DELETE concern
            """, memory_id=memory["memory_id"]).consume()
            session.run("""
                MATCH (m:Memory {memory_id:$memory_id})-[mention:MENTIONS]->(:Person)
                DELETE mention
            """, memory_id=memory["memory_id"]).consume()
        self.write_memory(memory)

    def retire_unenrolled_person(self, unknown_label: str, enrolled_name: str | None = None) -> None:
        """Transfer relationships to the enrolled person if provided, and delete the temporary unknown person node."""
        with self.driver.session() as session:
            unknown_key = self.person_key(unknown_label)
            if enrolled_name:
                target_key = self.person_key(enrolled_name)
                # Transfer PRESENT_AT edges
                session.run("""
                    MATCH (u:Person {name_key: $unknown_key})
                    MATCH (target:Person {name_key: $target_key})
                    MATCH (u)-[:PRESENT_AT]->(m:Memory)
                    MERGE (target)-[:PRESENT_AT]->(m)
                """, unknown_key=unknown_key, target_key=target_key).consume()
                # Transfer CONCERNS edges
                session.run("""
                    MATCH (u:Person {name_key: $unknown_key})
                    MATCH (target:Person {name_key: $target_key})
                    MATCH (u)-[r:CONCERNS]->(m:Memory)
                    MERGE (target)-[:CONCERNS {reported_by: r.reported_by, what_was_said: r.what_was_said}]->(m)
                """, unknown_key=unknown_key, target_key=target_key).consume()
            session.run("""
                MATCH (p:Person {name_key:$name_key})
                DETACH DELETE p
            """, name_key=unknown_key).consume()

    def get_direct(self, person_name: str, limit: int = 5) -> list[dict]:
        with self.driver.session() as session:
            return [dict(row) for row in session.run("""
                MATCH (p:Person {name_key:$name_key})-[r:CONCERNS]->(m:Memory)
                RETURN m.memory_id AS memory_id, m.summary AS summary, m.updated_at AS updated_at,
                       r.reported_by AS reported_by, r.what_was_said AS what_was_said
                ORDER BY m.updated_at DESC LIMIT $limit
            """, name_key=self.person_key(person_name), limit=limit)]

    def get_person_by_id(self, person_id: str) -> dict[str, Any] | None:
        """Return graph context for a biometric identity already matched in SQLite."""
        with self.driver.session() as session:
            row = session.run("""
                MATCH (p:Person {person_id:$person_id})
                OPTIONAL MATCH (p)-[:RELATED_TO]->(patient:Person)
                OPTIONAL MATCH (p)-[:PRESENT_AT]->(present_memory:Memory)
                OPTIONAL MATCH (p)-[:CONCERNS]->(concern_memory:Memory)
                RETURN p.name AS name, p.person_id AS person_id, p.name_key AS name_key,
                       collect(DISTINCT patient.name) AS related_to_patients,
                       count(DISTINCT present_memory) AS present_memory_count,
                       count(DISTINCT concern_memory) AS concern_memory_count
            """, person_id=person_id).single()
        return dict(row) if row else None

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
        self.index = faiss.read_index(str(self.index_path)) if self.index_path.exists() else faiss.IndexIDMap(faiss.IndexFlatIP(384))
        # Older installations used L2 distance.  Recreate that index; startup
        # backfill repopulates every durable memory with cosine-normalized data.
        if getattr(self.index, "metric_type", self.faiss.METRIC_INNER_PRODUCT) != self.faiss.METRIC_INNER_PRODUCT:
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(384))

    def _save(self) -> None:
        self.faiss.write_index(self.index, str(self.index_path))
        self.map_path.write_text(json.dumps(self.id_map))

    def upsert(self, memory_id: str, summary_text: str) -> None:
        import numpy as np
        vector = self.embedder.encode([summary_text], normalize_embeddings=True)[0].astype("float32").reshape(1, -1)
        int_id = self.id_map.setdefault(memory_id, self.next_int)
        if int_id == self.next_int: self.next_int += 1
        self.index.remove_ids(np.array([int_id], dtype="int64"))
        self.index.add_with_ids(vector, np.array([int_id], dtype="int64")); self._save()

    def delete(self, memory_id: str) -> None:
        """Remove a deliberately deleted memory from the vector projection."""
        if memory_id not in self.id_map:
            return
        import numpy as np
        int_id = int(self.id_map.pop(memory_id))
        self.index.remove_ids(np.array([int_id], dtype="int64"))
        self._save()

    def search(self, query_text: str, k: int = 5) -> list[str]:
        import numpy as np
        if not self.id_map: return []
        vector = self.embedder.encode([query_text], normalize_embeddings=True)[0].astype("float32").reshape(1, -1)
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
    def get_recent(self, participant: str, limit: int = 3, patient_id: int | None = None): return self.primary.get_recent(participant, limit, patient_id)
    def get_upcoming(self, now: str, limit: int = 10, patient_id: int | None = None): return self.primary.get_upcoming(now, limit, patient_id)
    def get_many(self, memory_ids: list[str]): return self.primary.get_many(memory_ids)
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
