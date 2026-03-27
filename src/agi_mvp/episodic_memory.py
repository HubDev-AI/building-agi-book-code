# episodic_memory.py

from dataclasses import dataclass, field
from typing import Optional
import time
import uuid
import json

@dataclass
class Episode:
    id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )
    timestamp: float = field(default_factory=time.time)
    task_description: str = ""
    actions_taken: list[str] = field(default_factory=list)
    outcome: str = ""  # "success", "failure", "partial"
    lesson_learned: str = ""
    context: dict = field(default_factory=dict)
    importance: float = 0.5
    access_count: int = 0

    def to_text(self) -> str:
        """Serialize for embedding."""
        return (
            f"Task: {self.task_description}\n"
            f"Actions: {'; '.join(self.actions_taken)}\n"
            f"Outcome: {self.outcome}\n"
            f"Lesson: {self.lesson_learned}"
        )

class EpisodicMemory:
    """Stores and retrieves past experiences via vector similarity."""

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection: str = "episodes",
        in_memory: bool = False,
    ):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        if in_memory:
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(url=qdrant_url)
        self.collection = collection
        self._embedding_model = self._load_embedder()

        # Create collection if needed
        collections = [
            c.name
            for c in self.client.get_collections().collections
        ]
        if collection not in collections:
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(
                    size=1024, distance=Distance.COSINE
                ),
            )

    def store(self, episode: Episode):
        """Store a new episode."""
        from qdrant_client.models import PointStruct
        embedding = self._embed(episode.to_text())
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(
                id=episode.id,
                vector=embedding,
                payload={
                    "task": episode.task_description,
                    "actions": episode.actions_taken,
                    "outcome": episode.outcome,
                    "lesson": episode.lesson_learned,
                    "timestamp": episode.timestamp,
                    "importance": episode.importance,
                },
            )],
        )

    def recall(self, query: str, top_k: int = 5,
               outcome_filter: Optional[str] = None
               ) -> list[Episode]:
        """Retrieve relevant past episodes."""
        embedding = self._embed(query)

        filter_conditions = None
        if outcome_filter:
            from qdrant_client.models import (
                Filter, FieldCondition, MatchValue,
            )
            filter_conditions = Filter(must=[
                FieldCondition(
                    key="outcome",
                    match=MatchValue(value=outcome_filter),
                )
            ])

        results = self.client.query_points(
            collection_name=self.collection,
            query=embedding,
            limit=top_k,
            query_filter=filter_conditions,
        )

        episodes = []
        for hit in results.points:
            p = hit.payload
            episodes.append(Episode(
                id=hit.id,
                task_description=p["task"],
                actions_taken=p["actions"],
                outcome=p["outcome"],
                lesson_learned=p["lesson"],
                timestamp=p["timestamp"],
                importance=p["importance"],
            ))
        return episodes

    def recall_failures(self, query: str,
                        top_k: int = 3) -> list[Episode]:
        """Specifically recall past failures -- learn from mistakes."""
        return self.recall(
            query, top_k=top_k, outcome_filter="failure"
        )

    def recall_successes(self, query: str,
                         top_k: int = 3) -> list[Episode]:
        """Recall past successes -- repeat what works."""
        return self.recall(
            query, top_k=top_k, outcome_filter="success"
        )

    def _embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        return self._embedding_model.encode(text).tolist()

    def _load_embedder(self):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("BAAI/bge-large-en-v1.5")
