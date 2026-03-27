# memory_manager.py

from dataclasses import dataclass
from typing import Optional
import time

from .working_memory import WorkingMemory
from .episodic_memory import Episode, EpisodicMemory
from .semantic_memory import Fact, SemanticMemory

class MemoryManager:
    """Orchestrates the three-tier memory system."""

    def __init__(self,
                 working: WorkingMemory,
                 episodic: EpisodicMemory,
                 semantic: SemanticMemory,
                 foundation: 'FoundationModel'):
        self.working = working
        self.episodic = episodic
        self.semantic = semantic
        self.foundation = foundation

    async def remember(self, query: str,
                       context: str = "") -> str:
        """Retrieve all relevant memories for a query.

        Returns a formatted string ready for LLM context
        injection.
        """
        sections = []

        # 1. Working memory (always included)
        wm = self.working.get_context_window(max_tokens=2000)
        if wm:
            sections.append(wm)

        # 2. Relevant past episodes
        episodes = self.episodic.recall(query, top_k=3)
        if episodes:
            ep_text = "## Relevant Past Experiences\n"
            for ep in episodes:
                ep_text += (
                    f"- **{ep.task_description}** -> "
                    f"{ep.outcome}\n"
                    f"  Lesson: {ep.lesson_learned}\n"
                )
            sections.append(ep_text)

        # 3. Relevant facts
        facts = self.semantic.search_semantic(query, top_k=10)
        if facts:
            fact_text = "## Known Facts\n"
            for f in facts:
                fact_text += (
                    f"- {f.to_triple()} "
                    f"(confidence: {f.confidence:.0%})\n"
                )
            sections.append(fact_text)

        return "\n\n".join(sections)

    async def learn_from_episode(self, episode: Episode):
        """Store an episode with surprise-gated importance scoring.

        Inspired by Google's Titans architecture: high surprise
        (divergence between prediction and observation) triggers
        prioritized storage. Routine, expected events are stored
        with low importance or discarded entirely.
        """
        # Surprise gating: compute divergence between
        # prediction and reality
        surprise = await self._compute_surprise(episode)
        episode.importance = max(episode.importance, surprise)

        if surprise < 0.2 and episode.outcome == "success":
            # Routine success -- store with minimal importance
            episode.importance = 0.2
        elif surprise > 0.7:
            # High surprise -- this is novel, learn from it
            episode.importance = min(0.95, surprise)

        # Store the episode
        self.episodic.store(episode)

        # If the episode was important, try to extract
        # general knowledge
        if (episode.importance > 0.6
                and episode.outcome in ("success", "failure")):
            await self._consolidate(episode)

    async def _compute_surprise(self, episode: Episode) -> float:
        """Compute surprise: how much did reality diverge
        from prediction?

        This implements the Titans-inspired surprise mechanism:
        high surprise = store permanently,
        low surprise = forgettable.
        """
        prompt = (
            "Rate how surprising this outcome was on a scale "
            "of 0.0 to 1.0.\n\n"
            f"Task: {episode.task_description}\n"
            f"Expected outcome type: "
            f"{episode.context.get('expected', 'success')}\n"
            f"Actual outcome: {episode.outcome}\n"
            f"Lesson: {episode.lesson_learned}\n\n"
            "0.0 = completely expected, routine\n"
            "0.5 = somewhat unexpected\n"
            "1.0 = completely surprising, unprecedented\n\n"
            "Return only a number between 0.0 and 1.0."
        )
        from .foundation import Message, GenerationConfig
        result = await self.foundation.generate(
            [Message(role="user", content=prompt)],
            GenerationConfig(temperature=0.1, max_tokens=10),
        )
        try:
            return float(result.content.strip())
        except ValueError:
            return 0.5  # Default moderate surprise

    async def _consolidate(self, episode: Episode):
        """Extract semantic facts from an episodic memory.

        This is memory consolidation -- like sleep in the brain.
        """
        prompt = (
            "Analyze this experience and extract any general "
            "facts or rules that should be remembered "
            "permanently.\n\n"
            f"Experience:\n"
            f"- Task: {episode.task_description}\n"
            f"- Actions: {'; '.join(episode.actions_taken)}\n"
            f"- Outcome: {episode.outcome}\n"
            f"- Lesson: {episode.lesson_learned}\n\n"
            "Extract facts as JSON array:\n"
            '[{"subject": "...", "predicate": "...", '
            '"object": "..."}]\n\n'
            "Only include facts that are general (not specific "
            "to this one task) and likely to be useful in the "
            "future. If no general facts can be extracted, "
            "return an empty array []."
        )
        from .foundation import Message, GenerationConfig
        result = await self.foundation.generate(
            [Message(role="user", content=prompt)],
            GenerationConfig(temperature=0.3, max_tokens=500),
        )

        try:
            import json, uuid
            facts_data = json.loads(result.content)
            for fd in facts_data:
                fact = Fact(
                    id=str(uuid.uuid4()),
                    subject=fd["subject"],
                    predicate=fd["predicate"],
                    object=fd["object"],
                    source=(
                        f"consolidated from episode "
                        f"{episode.id}"
                    ),
                    confidence=0.8,
                    last_verified=time.time(),
                )
                self.semantic.store_fact(fact)
        except (json.JSONDecodeError, KeyError):
            pass  # Not every episode yields facts
