# semantic_memory.py

from dataclasses import dataclass, field
from typing import Optional
import json
import uuid

@dataclass
class Fact:
    subject: str
    predicate: str
    object: str
    confidence: float = 0.9
    source: str = "learned"  # "learned", "told", "inferred"
    last_verified: float = 0.0
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_triple(self) -> str:
        return (
            f"{self.subject} -> {self.predicate} -> {self.object}"
        )

    def to_text(self) -> str:
        return f"{self.subject} {self.predicate} {self.object}"

class SemanticMemory:
    """Stores structured knowledge as a graph with
    vector-indexed retrieval."""

    def __init__(self, db_path: str = "semantic_memory.db"):
        import sqlite3
        self.conn = sqlite3.connect(db_path)
        self._init_schema()
        self._vector_store = None  # Lazy init

    def store_fact(self, fact: Fact):
        """Store or update a fact."""
        self.conn.execute("""
            INSERT OR REPLACE INTO facts
            (id, subject, predicate, object,
             confidence, source, last_verified)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (fact.id, fact.subject, fact.predicate,
              fact.object, fact.confidence, fact.source,
              fact.last_verified))
        self.conn.commit()

    def query_facts(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
    ) -> list[Fact]:
        """Query facts by subject, predicate, and/or object."""
        conditions = []
        params = []
        if subject:
            conditions.append("subject LIKE ?")
            params.append(f"%{subject}%")
        if predicate:
            conditions.append("predicate LIKE ?")
            params.append(f"%{predicate}%")
        if obj:
            conditions.append("object LIKE ?")
            params.append(f"%{obj}%")

        where = " AND ".join(conditions) if conditions else "1=1"
        rows = self.conn.execute(
            f"SELECT * FROM facts WHERE {where} "
            f"ORDER BY confidence DESC", params
        ).fetchall()

        return [
            Fact(id=r[0], subject=r[1], predicate=r[2],
                 object=r[3], confidence=r[4], source=r[5],
                 last_verified=r[6])
            for r in rows
        ]

    def search_semantic(self, query: str,
                        top_k: int = 10) -> list[Fact]:
        """Semantic search across all stored facts."""
        return (
            self.query_facts(subject=query)
            + self.query_facts(obj=query)
        )

    def correct_fact(self, fact_id: str, new_object: str,
                     reason: str = "correction"):
        """Update a fact -- the system learned something new."""
        self.conn.execute("""
            UPDATE facts SET object = ?, source = ?,
            confidence = 0.95
            WHERE id = ?
        """, (new_object, f"corrected: {reason}", fact_id))
        self.conn.commit()

    def get_related(self, entity: str,
                    max_hops: int = 2) -> list[Fact]:
        """Get facts related to an entity within N hops."""
        visited = set()
        frontier = {entity}
        all_facts = []

        for _ in range(max_hops):
            new_frontier = set()
            for e in frontier:
                if e in visited:
                    continue
                visited.add(e)
                facts = (
                    self.query_facts(subject=e)
                    + self.query_facts(obj=e)
                )
                all_facts.extend(facts)
                for f in facts:
                    new_frontier.add(f.subject)
                    new_frontier.add(f.object)
            frontier = new_frontier - visited

        return all_facts

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence REAL DEFAULT 0.9,
                source TEXT DEFAULT 'learned',
                last_verified REAL DEFAULT 0
            )
        """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_subject "
            "ON facts(subject)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_object "
            "ON facts(object)"
        )
        self.conn.commit()
