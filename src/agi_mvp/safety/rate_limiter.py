# safety/rate_limiter.py

from dataclasses import dataclass, field
import time
from collections import deque


@dataclass
class ResourceBudget:
    """Per-task resource limits."""
    max_tool_calls: int = 50
    max_llm_calls: int = 20
    max_tokens_generated: int = 100_000
    max_wall_time_seconds: float = 300
    max_cost_dollars: float = 1.0


class TaskRateLimiter:
    """Enforces resource budgets per task execution.
    Checked before every tool call, LLM call, or
    external action."""

    def __init__(self, budget: ResourceBudget):
        self.budget = budget
        self.tool_calls = 0
        self.llm_calls = 0
        self.tokens_generated = 0
        self.start_time = time.time()
        self.estimated_cost = 0.0
        self._tool_call_times: deque = deque(maxlen=100)

    def check_tool_call(self) -> tuple[bool, str]:
        if self.tool_calls >= self.budget.max_tool_calls:
            return False, (
                f"Tool call limit reached "
                f"({self.budget.max_tool_calls})"
            )
        if self._wall_time_exceeded():
            return False, "Wall time limit reached"
        # Burst rate: max 10 tool calls per second
        now = time.time()
        self._tool_call_times.append(now)
        if (
            len(self._tool_call_times) >= 10
            and now - self._tool_call_times[0] < 1.0
        ):
            return False, (
                "Tool call rate limit (10/s) exceeded"
            )
        self.tool_calls += 1
        return True, ""

    def check_llm_call(
        self, estimated_tokens: int = 0
    ) -> tuple[bool, str]:
        if self.llm_calls >= self.budget.max_llm_calls:
            return False, "LLM call limit reached"
        if (
            self.tokens_generated + estimated_tokens
            > self.budget.max_tokens_generated
        ):
            return False, "Token budget exceeded"
        self.llm_calls += 1
        self.tokens_generated += estimated_tokens
        return True, ""

    def _wall_time_exceeded(self) -> bool:
        return (
            (time.time() - self.start_time)
            > self.budget.max_wall_time_seconds
        )
