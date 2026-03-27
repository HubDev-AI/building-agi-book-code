# architecture_types.py -- Canonical layer interfaces and protocols
#
# Book reference: Chapter 4 "Architecture Overview -- The Cognitive Core"
#
# These are the interface contracts defined in Ch04's verbatim blocks.
# Each layer's detailed implementation lives in its own module; these
# types capture the cross-layer protocols and shared vocabulary.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# -----------------------------------------------------------------
# Layer 1: Foundation Model interface  (Ch04, lines 112-116)
# -----------------------------------------------------------------
# Layer1.complete(prompt, model_hint, params) -> CompletionResult
# Layer1.embed(text) -> Vector[float]
# Layer1.classify(input, categories) -> ClassificationResult

@dataclass
class CompletionResult:
    content: str
    model: str = ""
    tokens_used: int = 0
    latency_ms: float = 0.0


@dataclass
class ClassificationResult:
    category: str
    confidence: float  # 0.0 - 1.0
    scores: dict[str, float] = field(default_factory=dict)


# -----------------------------------------------------------------
# Layer 2: Memory interface  (Ch04, lines 147-155)
# -----------------------------------------------------------------
# Memory.working.get(key) -> Value
# Memory.working.set(key, value, relevance_score)
# Memory.episodic.store(episode: Episode)
# Memory.episodic.retrieve(query, top_k) -> list[Episode]
# Memory.semantic.store(fact: Fact)
# Memory.semantic.query(query, top_k) -> list[Fact]
# Memory.consolidate()  # episodic -> semantic promotion

@dataclass
class Episode:
    timestamp: float
    context: str
    action: str
    result: str
    evaluation: str = ""
    embedding: list[float] = field(default_factory=list)


@dataclass
class Fact:
    content: str
    source: str = ""
    confidence: float = 1.0
    embedding: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------
# Layer 3: Reasoning interface  (Ch04, lines 174-182)
# -----------------------------------------------------------------
# Reasoning.infer(query, context, mode) -> ReasoningResult
#   ReasoningResult contains:
#     conclusion: str
#     confidence: float (0-1)
#     verified: bool
#     verification_method: str | None
#     reasoning_trace: list[ReasoningStep]

@dataclass
class ReasoningStep:
    step_number: int
    description: str
    output: str = ""
    confidence: float = 0.0


# NOTE: ReasoningResult is defined in reasoning_types.py with
# more detailed fields.  The Ch04 interface specifies:
#   conclusion, confidence, verified, verification_method,
#   reasoning_trace.
# The reasoning_types.py version (answer, mode, confidence,
# reasoning_trace, verification, verification_detail, attempts,
# z3_model, metadata) is the implementation.


# -----------------------------------------------------------------
# Layer 4: Planner interface  (Ch04, lines 203-209)
# -----------------------------------------------------------------
# Planner.submit_goal(goal: Goal) -> PlanID
# Planner.get_plan(plan_id) -> Plan
# Planner.step(plan_id) -> NextAction
# Planner.replan(plan_id, reason: str) -> Plan
# Planner.status(plan_id) -> PlanStatus

PlanID = str


class PlanStatus(Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    REPLANNING = "replanning"
    BLOCKED = "blocked"


@dataclass
class NextAction:
    task_id: str
    task_name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------
# Layer 5: Perception & Action interface  (Ch04, lines 231-235)
# -----------------------------------------------------------------
# Action.execute(tool_name, params) -> ActionResult
# Action.observe(source) -> Observation
# Action.available_tools() -> list[ToolSpec]

@dataclass
class ActionResult:
    success: bool
    output: Any = None
    error: Optional[str] = None
    side_effects: list[str] = field(default_factory=list)
    duration_ms: float = 0.0


@dataclass
class Observation:
    source: str
    content: Any
    timestamp: float = 0.0
    structured: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters_schema: dict[str, Any] = field(
        default_factory=dict
    )


# -----------------------------------------------------------------
# Layer 6: World Model  (Ch04, lines 248-265)
# -----------------------------------------------------------------
# WorldState:
#   entities: dict[EntityID, Entity]
#   history: list[StateTransition]
#   predictions: list[Prediction]
#
# WorldModel.current_state() -> WorldState
# WorldModel.predict(action) -> PredictedStateDelta
# WorldModel.update(observation) -> StateTransition
# WorldModel.divergence() -> list[PredictionError]
# WorldModel.query(question_about_state) -> Answer

EntityID = str


@dataclass
class Relation:
    target: EntityID
    relation_type: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class Entity:
    name: str
    entity_type: str
    properties: dict[str, Any] = field(default_factory=dict)
    relations: list[Relation] = field(default_factory=list)


@dataclass
class StateTransition:
    action: str
    pre_state_hash: str
    post_state_hash: str
    timestamp: float = 0.0


@dataclass
class Prediction:
    if_action: str
    expected_state_delta: dict[str, Any] = field(
        default_factory=dict
    )
    confidence: float = 0.0


@dataclass
class PredictedStateDelta:
    changes: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class PredictionError:
    prediction: Prediction
    actual_delta: dict[str, Any] = field(default_factory=dict)
    divergence_score: float = 0.0


@dataclass
class WorldState:
    entities: dict[EntityID, Entity] = field(
        default_factory=dict
    )
    history: list[StateTransition] = field(
        default_factory=list
    )
    predictions: list[Prediction] = field(
        default_factory=list
    )


# -----------------------------------------------------------------
# Layer 7: Meta-Cognition interface  (Ch04, lines 284-290)
# -----------------------------------------------------------------
# Meta.assess_confidence(result, context) -> float
# Meta.select_model(task) -> ModelChoice
# Meta.allocate_resources(task, constraints) -> ResourceBudget
# Meta.evaluate(goal, result) -> Evaluation
# Meta.system_health() -> HealthReport

@dataclass
class ModelChoice:
    model_name: str
    reasoning: str = ""
    estimated_latency_ms: float = 0.0
    estimated_cost: float = 0.0


@dataclass
class ResourceBudget:
    max_tokens: int = 8192
    max_time_seconds: float = 30.0
    max_retries: int = 3
    reasoning_mode: str = "hybrid"


@dataclass
class Evaluation:
    goal_achieved: bool
    score: float = 0.0  # 0.0 - 1.0
    what_worked: str = ""
    what_failed: str = ""
    lessons: list[str] = field(default_factory=list)


@dataclass
class HealthReport:
    overall_status: str = "ok"  # "ok", "degraded", "error"
    component_status: dict[str, str] = field(
        default_factory=dict
    )
    metrics: dict[str, float] = field(default_factory=dict)


# -----------------------------------------------------------------
# Tool Call Protocol  (Ch04, lines 396-408)
# -----------------------------------------------------------------
# ToolRequest:
#   tool_name: str
#   parameters: dict
#   timeout_ms: int
#   sandbox: bool
#
# ToolResponse:
#   status: "success" | "error" | "timeout"
#   result: Any
#   duration_ms: int
#   side_effects: list[str]

class ToolResponseStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class ToolRequest:
    tool_name: str               # e.g., "filesystem.read_file"
    parameters: dict[str, Any] = field(
        default_factory=dict
    )                            # e.g., {"path": "/src/main.py"}
    timeout_ms: int = 30_000     # max wait time
    sandbox: bool = False        # execute in E2B sandbox?


@dataclass
class ToolResponse:
    status: ToolResponseStatus = ToolResponseStatus.SUCCESS
    result: Any = None           # tool-specific return value
    duration_ms: int = 0         # actual execution time
    side_effects: list[str] = field(
        default_factory=list
    )                            # what changed in the environment
