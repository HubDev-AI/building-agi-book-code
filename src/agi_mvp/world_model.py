# world_model.py

from dataclasses import dataclass, field
from typing import Any, Optional
import time
import json
import copy
from collections import defaultdict

@dataclass
class Entity:
    id: str
    type: str  # "file", "service", "variable", "person", ...
    properties: dict[str, Any] = field(default_factory=dict)
    relations: list[tuple[str, str]] = field(
        default_factory=list  # (predicate, target_id)
    )
    last_updated: float = field(default_factory=time.time)

@dataclass
class WorldState:
    entities: dict[str, Entity] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    version: int = 0

    def add_entity(self, entity: Entity):
        self.entities[entity.id] = entity
        self.version += 1

    def update_property(self, entity_id: str,
                        key: str, value: Any):
        if entity_id in self.entities:
            self.entities[entity_id].properties[key] = value
            self.entities[entity_id].last_updated = time.time()
            self.version += 1

    def add_relation(self, source_id: str,
                     predicate: str, target_id: str):
        if source_id in self.entities:
            self.entities[source_id].relations.append(
                (predicate, target_id)
            )
            self.version += 1

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self.entities.get(entity_id)

    def query_relations(
        self, entity_id: str,
        predicate: Optional[str] = None,
    ) -> list[tuple[str, str]]:
        entity = self.entities.get(entity_id)
        if not entity:
            return []
        if predicate:
            return [(p, t) for p, t in entity.relations
                    if p == predicate]
        return entity.relations

    def snapshot(self) -> 'WorldState':
        """Create a deep copy for 'what if' reasoning."""
        return copy.deepcopy(self)

    def to_text(self, max_entities: int = 20) -> str:
        """Serialize for LLM context injection."""
        lines = [f"## World State (v{self.version})"]
        sorted_entities = sorted(
            self.entities.values(),
            key=lambda e: e.last_updated,
            reverse=True,
        )[:max_entities]

        for e in sorted_entities:
            props = ", ".join(
                f"{k}={v}" for k, v in e.properties.items()
            )
            lines.append(f"- **{e.id}** ({e.type}): {props}")
            for pred, target in e.relations:
                lines.append(f"  -> {pred} -> {target}")

        return "\n".join(lines)


class StateTracker:
    """Maintains and updates the world state
    from observations."""

    def __init__(self, foundation: 'FoundationModel'):
        self.state = WorldState()
        self.foundation = foundation
        self._history: list[WorldState] = []

    async def observe(self, observation: str):
        """Process an observation and update the
        world state."""
        prompt = f"""Given the current world state and a new
observation, extract any state changes.

Current state:
{self.state.to_text()}

New observation:
{observation}

Extract changes as JSON:
{{
  "new_entities": [
    {{"id": "...", "type": "...", "properties": {{}}}}
  ],
  "updated_properties": [
    {{"entity_id": "...", "key": "...", "value": "..."}}
  ],
  "new_relations": [
    {{"source": "...", "predicate": "...", "target": "..."}}
  ],
  "removed_entities": ["entity_id", ...]
}}

Only include actual changes. Return empty lists if
nothing changed."""
        from .foundation import Message, GenerationConfig
        result = await self.foundation.generate(
            [Message(role="user", content=prompt)],
            GenerationConfig(temperature=0.2,
                             max_tokens=1000),
        )

        try:
            changes = json.loads(result.content)
            self._apply_changes(changes)
        except json.JSONDecodeError:
            pass  # Observation didn't yield parseable changes

    def _apply_changes(self, changes: dict):
        """Apply extracted changes to the world state."""
        self._history.append(self.state.snapshot())

        for ne in changes.get("new_entities", []):
            self.state.add_entity(Entity(
                id=ne["id"], type=ne["type"],
                properties=ne.get("properties", {}),
            ))

        for up in changes.get("updated_properties", []):
            self.state.update_property(
                up["entity_id"], up["key"], up["value"],
            )

        for nr in changes.get("new_relations", []):
            self.state.add_relation(
                nr["source"], nr["predicate"], nr["target"],
            )

        for eid in changes.get("removed_entities", []):
            self.state.entities.pop(eid, None)
            self.state.version += 1

    def rollback(self, steps: int = 1):
        """Undo recent state changes."""
        for _ in range(min(steps, len(self._history))):
            self.state = self._history.pop()


class DynamicsPredictor:
    """Predicts consequences of actions in the world."""

    def __init__(self, foundation: 'FoundationModel',
                 state_tracker: StateTracker):
        self.foundation = foundation
        self.state_tracker = state_tracker

    async def predict(self, action: str,
                      depth: int = 1) -> list[dict]:
        """Predict the consequences of an action.

        Args:
            action: Proposed action in natural language
            depth: How many steps ahead to predict

        Returns:
            List of predicted effects with confidence scores.
        """
        prompt = f"""Given the current world state, predict
what happens if we take this action.

Current state:
{self.state_tracker.state.to_text()}

Proposed action: {action}

Predict the effects as JSON array:
[
  {{"effect": "description", "probability": 0.0-1.0,
    "state_changes": {{...}}}},
  ...
]

Consider:
1. Direct effects (what the action immediately causes)
2. Side effects (unintended consequences)
3. Preconditions (will this action even work in the
   current state?)

Be specific and realistic. If the action would fail,
say why."""
        from .foundation import Message, GenerationConfig
        result = await self.foundation.generate(
            [Message(role="user", content=prompt)],
            GenerationConfig(
                temperature=0.4, max_tokens=1500,
                force_reasoning=True,
            ),
        )

        try:
            return json.loads(result.content)
        except json.JSONDecodeError:
            return [{"effect": result.content,
                     "probability": 0.5,
                     "state_changes": {}}]

    async def simulate(
        self, action_sequence: list[str],
    ) -> list[WorldState]:
        """Simulate a sequence of actions, returning
        predicted world states."""
        states = []
        sim_state = self.state_tracker.state.snapshot()

        for action in action_sequence:
            predictions = await self.predict(action)
            if predictions:
                best = max(
                    predictions,
                    key=lambda p: p.get("probability", 0),
                )
                changes = best.get("state_changes", {})
                if changes:
                    self.state_tracker._apply_changes(changes)
            states.append(self.state_tracker.state.snapshot())

        # Restore actual state
        self.state_tracker.state = sim_state
        return states


class CausalGraph:
    """Tracks observed cause-effect relationships."""

    def __init__(self):
        # cause -> {effect -> count}
        self._edges: dict[str, dict[str, int]] = \
            defaultdict(lambda: defaultdict(int))
        self._total_observations: dict[str, int] = \
            defaultdict(int)

    def observe_causation(self, cause: str, effect: str):
        """Record an observed cause-effect relationship."""
        self._edges[cause][effect] += 1
        self._total_observations[cause] += 1

    def get_likely_effects(
        self, cause: str, min_probability: float = 0.3,
    ) -> list[tuple[str, float]]:
        """Get likely effects of a cause with
        probabilities."""
        total = self._total_observations.get(cause, 0)
        if total == 0:
            return []

        effects = []
        for effect, count in self._edges[cause].items():
            prob = count / total
            if prob >= min_probability:
                effects.append((effect, prob))

        return sorted(effects, key=lambda x: x[1],
                      reverse=True)

    def get_likely_causes(
        self, effect: str,
    ) -> list[tuple[str, float]]:
        """Reverse lookup: what might have caused
        this effect?"""
        candidates = []
        for cause, effects in self._edges.items():
            if effect in effects:
                total = self._total_observations[cause]
                prob = effects[effect] / total
                candidates.append((cause, prob))
        return sorted(candidates, key=lambda x: x[1],
                      reverse=True)

    def to_text(self) -> str:
        """Serialize for LLM context."""
        lines = ["## Known Causal Relationships"]
        for cause in sorted(self._edges.keys()):
            effects = self.get_likely_effects(cause)
            for effect, prob in effects:
                lines.append(
                    f"- {cause} -> {effect} (p={prob:.0%})"
                )
        return "\n".join(lines)


class WorldModel:
    """Unified world model combining state tracking,
    prediction, and causation."""

    def __init__(self, foundation: 'FoundationModel'):
        self.state_tracker = StateTracker(foundation)
        self.predictor = DynamicsPredictor(
            foundation, self.state_tracker,
        )
        self.causal_graph = CausalGraph()

    async def observe(self, observation: str):
        """Update world model from an observation."""
        await self.state_tracker.observe(observation)

    async def predict_action(
        self, action: str,
    ) -> list[dict]:
        """What would happen if we did this?"""
        return await self.predictor.predict(action)

    async def simulate_plan(
        self, actions: list[str],
    ) -> list[WorldState]:
        """Simulate a plan before executing it."""
        return await self.predictor.simulate(actions)

    def record_outcome(self, action: str, outcome: str):
        """Learn a causal relationship from an
        action-outcome pair."""
        self.causal_graph.observe_causation(action, outcome)

    def get_context(self, max_tokens: int = 2000) -> str:
        """Get world model context for LLM injection."""
        parts = [
            self.state_tracker.state.to_text(
                max_entities=15
            ),
            self.causal_graph.to_text(),
        ]
        return "\n\n".join(parts)

    @property
    def state(self) -> WorldState:
        return self.state_tracker.state
