# metacognition.py

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import time
import json
import statistics

class Difficulty(Enum):
    TRIVIAL = "trivial"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXTREME = "extreme"

@dataclass
class CognitiveState:
    """Snapshot of the system's current cognitive state."""
    current_task: Optional[str] = None
    confidence: float = 0.5
    difficulty_estimate: Difficulty = Difficulty.MEDIUM
    tokens_spent: int = 0
    time_spent_ms: float = 0
    reasoning_depth: int = 0   # How many CoT steps used
    tools_used: int = 0
    errors_encountered: int = 0
    replans: int = 0

@dataclass
class PerformanceRecord:
    task_type: str
    strategy_used: str
    success: bool
    confidence_before: float
    time_ms: float
    tokens_used: int
    timestamp: float = field(default_factory=time.time)


class MetaCognition:
    """The self-monitoring and self-improvement layer."""

    def __init__(self, foundation: 'FoundationModel'):
        self.foundation = foundation
        self.state = CognitiveState()
        self._performance_log: list[PerformanceRecord] = []
        self._strategy_scores: dict[str, list[bool]] = {}

    # --- Confidence Estimation ---

    def estimate_confidence(
        self, reasoning_result: 'ReasoningOutput',
    ) -> float:
        """Estimate confidence in a reasoning result.

        Combines multiple signals:
        - Model's self-reported confidence
        - Number of self-corrections during reasoning
        - Historical accuracy on similar tasks
        - Consistency across approaches
        """
        signals = []

        # 1. Model's linguistic confidence
        signals.append(reasoning_result.confidence)

        # 2. Self-correction penalty
        correction_penalty = max(
            0,
            1.0 - reasoning_result.self_corrections * 0.15,
        )
        signals.append(correction_penalty)

        # 3. Historical accuracy for this task type
        task_type = self.state.current_task or "unknown"
        historical = self._get_historical_accuracy(task_type)
        if historical is not None:
            signals.append(historical)

        # 4. Reasoning depth bonus
        depth_bonus = min(
            1.0,
            0.5 + reasoning_result.steps.__len__() * 0.05,
        )
        signals.append(depth_bonus)

        # Weighted average
        weights = [0.3, 0.2, 0.3, 0.2][:len(signals)]
        total_weight = sum(weights)
        confidence = sum(
            s * w for s, w in zip(signals, weights)
        ) / total_weight

        self.state.confidence = max(0.05,
                                     min(0.99, confidence))
        return self.state.confidence

    # --- Resource Allocation ---

    def should_use_reasoning_model(
        self, task_description: str,
    ) -> bool:
        """Decide whether this task needs the
        reasoning model."""
        difficulty = self._estimate_difficulty(
            task_description,
        )
        self.state.difficulty_estimate = difficulty

        # Trivial/Easy -> fast model
        if difficulty in (Difficulty.TRIVIAL,
                          Difficulty.EASY):
            return False

        # Hard/Extreme -> always reasoning
        if difficulty in (Difficulty.HARD,
                          Difficulty.EXTREME):
            return True

        # Medium -> check historical performance
        accuracy = self._get_historical_accuracy(
            task_description,
        )
        if accuracy is not None and accuracy > 0.85:
            return False  # We're good at this
        return True  # Default to reasoning when uncertain

    def _estimate_difficulty(self, task: str) -> Difficulty:
        """Quick heuristic difficulty estimation."""
        task_lower = task.lower()

        extreme_keywords = ["prove", "novel", "research",
                            "discover"]
        hard_keywords = ["debug", "architect", "multi-step",
                         "analyze complex"]
        medium_keywords = ["implement", "explain", "compare",
                           "plan"]
        easy_keywords = ["summarize", "format", "translate",
                         "extract"]
        trivial_keywords = ["yes/no", "classify", "lookup"]

        for kw in extreme_keywords:
            if kw in task_lower:
                return Difficulty.EXTREME
        for kw in hard_keywords:
            if kw in task_lower:
                return Difficulty.HARD
        for kw in medium_keywords:
            if kw in task_lower:
                return Difficulty.MEDIUM
        for kw in easy_keywords:
            if kw in task_lower:
                return Difficulty.EASY
        for kw in trivial_keywords:
            if kw in task_lower:
                return Difficulty.TRIVIAL

        return Difficulty.MEDIUM  # Default

    # --- Strategy Selection ---

    def select_strategy(self, task_type: str) -> str:
        """Choose the best strategy for a given task type
        based on past performance."""
        if task_type in self._strategy_scores:
            best_strategy = None
            best_rate = -1
            for strategy, outcomes in \
                    self._strategy_scores.items():
                if not outcomes:
                    continue
                rate = sum(outcomes) / len(outcomes)
                if rate > best_rate:
                    best_rate = rate
                    best_strategy = strategy

            if best_strategy and best_rate > 0.5:
                return best_strategy

        # Default strategies by task type
        defaults = {
            "math": "reason_then_verify",
            "code": "generate_and_test",
            "planning": "decompose_and_execute",
            "research": "search_synthesize_verify",
            "writing": "outline_draft_refine",
        }
        return defaults.get(task_type,
                            "reason_then_verify")

    # --- Self-Evaluation ---

    async def evaluate_outcome(
        self, task: str, result: str,
        expected: Optional[str] = None,
    ) -> dict:
        """Evaluate how well a task was completed."""
        prompt = f"""Evaluate the quality of this task
completion.

Task: {task}
Result: {result[:2000]}
{"Expected: " + expected if expected else ""}

Rate on these dimensions (0-10):
1. Correctness: Is the answer right?
2. Completeness: Does it cover everything asked?
3. Efficiency: Was it done without unnecessary steps?
4. Quality: Is the output well-structured and clear?

Return JSON:
{{"correctness": N, "completeness": N,
  "efficiency": N, "quality": N,
  "overall": N, "feedback": "..."}}"""

        from .foundation import Message, GenerationConfig
        eval_result = await self.foundation.generate(
            [Message(role="user", content=prompt)],
            GenerationConfig(temperature=0.2,
                             max_tokens=300),
        )

        try:
            scores = json.loads(eval_result.content)
            success = scores.get("overall", 5) >= 7

            record = PerformanceRecord(
                task_type=task[:50],
                strategy_used=(self.state.current_task
                               or "unknown"),
                success=success,
                confidence_before=self.state.confidence,
                time_ms=self.state.time_spent_ms,
                tokens_used=self.state.tokens_spent,
            )
            self._performance_log.append(record)

            return scores
        except json.JSONDecodeError:
            return {"overall": 5,
                    "feedback": "Could not parse evaluation"}

    # --- Performance Tracking ---

    def get_performance_summary(
        self, last_n: int = 50,
    ) -> dict:
        """Get aggregate performance metrics."""
        recent = self._performance_log[-last_n:]
        if not recent:
            return {"total_tasks": 0}

        successes = sum(1 for r in recent if r.success)
        confidences = [r.confidence_before for r in recent]
        times = [r.time_ms for r in recent]

        return {
            "total_tasks": len(recent),
            "success_rate": successes / len(recent),
            "avg_confidence": statistics.mean(confidences),
            "avg_time_ms": statistics.mean(times),
            "calibration_error":
                self._calibration_error(recent),
        }

    def _calibration_error(
        self, records: list[PerformanceRecord],
    ) -> float:
        """How well-calibrated are our confidence
        estimates?

        Perfect calibration: when we say 80% confident,
        we're right 80% of the time.
        """
        if len(records) < 10:
            return -1  # Not enough data

        bins = {}
        for r in records:
            bin_key = round(r.confidence_before, 1)
            if bin_key not in bins:
                bins[bin_key] = []
            bins[bin_key].append(r.success)

        errors = []
        for confidence, outcomes in bins.items():
            if len(outcomes) >= 3:
                actual = sum(outcomes) / len(outcomes)
                errors.append(abs(confidence - actual))

        return statistics.mean(errors) if errors else -1

    # --- Learning Controller ---

    async def extract_lesson(
        self, task: str, result: str,
        success: bool,
    ) -> Optional[str]:
        """Extract a reusable lesson from a
        completed task."""
        if success and self.state.confidence > 0.8:
            return None  # Nothing surprising to learn

        status = "succeeded" if success else "failed"
        surprise = ""
        if (success and self.state.confidence < 0.5) or \
           (not success and self.state.confidence > 0.7):
            surprise = " unexpectedly"

        prompt = f"""This task {status}{surprise}.

Task: {task}
Result: {result[:1000]}
My confidence was: {self.state.confidence:.0%}

What general lesson should I learn from this?
Focus on:
- What strategy worked (or didn't)
- What I should do differently next time
- Any pattern I should recognize in similar
  future tasks

Return a single concise sentence.
If there's no useful lesson, return "NONE"."""

        from .foundation import Message, GenerationConfig
        lesson_result = await self.foundation.generate(
            [Message(role="user", content=prompt)],
            GenerationConfig(temperature=0.3,
                             max_tokens=100),
        )

        lesson = lesson_result.content.strip()
        if lesson == "NONE" or len(lesson) < 10:
            return None
        return lesson

    # --- Helpers ---

    def _get_historical_accuracy(
        self, task_type: str,
    ) -> Optional[float]:
        """Get historical accuracy for tasks
        of this type."""
        relevant = [
            r for r in self._performance_log
            if task_type.lower() in r.task_type.lower()
        ]
        if len(relevant) < 5:
            return None
        return (sum(1 for r in relevant if r.success)
                / len(relevant))
