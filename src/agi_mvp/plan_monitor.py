# plan_monitor.py -- Detecting plan degradation

from dataclasses import dataclass
from .planner_types import Task, TaskStatus, Goal
import time


@dataclass
class PlanHealth:
    completion_ratio: float
    failure_ratio: float
    time_ratio: float     # Elapsed / estimated
    is_healthy: bool
    recommendation: str   # "continue", "replan", "abort"


def assess_plan_health(goal: Goal) -> PlanHealth:
    """Evaluate whether a plan is on track."""
    if not goal.task_tree:
        return PlanHealth(0, 0, 0, False, "abort")

    leaves = goal.task_tree.all_leaves()
    total = len(leaves)
    if total == 0:
        return PlanHealth(0, 0, 0, False, "abort")

    completed = sum(
        1 for t in leaves
        if t.status == TaskStatus.COMPLETED
    )
    failed = sum(
        1 for t in leaves
        if t.status == TaskStatus.FAILED
    )

    completion_ratio = completed / total
    failure_ratio = failed / total

    estimated_total = sum(
        t.estimated_seconds for t in leaves
    )
    elapsed = time.time() - goal.created_at
    time_ratio = (
        elapsed / estimated_total
        if estimated_total > 0 else 0
    )

    # Decision logic
    if failure_ratio > 0.3:
        return PlanHealth(
            completion_ratio, failure_ratio,
            time_ratio, False, "replan",
        )

    if time_ratio > 2.0 and completion_ratio < 0.5:
        return PlanHealth(
            completion_ratio, failure_ratio,
            time_ratio, False, "replan",
        )

    if failure_ratio > 0 and completion_ratio == 0:
        return PlanHealth(
            completion_ratio, failure_ratio,
            time_ratio, False, "replan",
        )

    if time_ratio > 3.0:
        return PlanHealth(
            completion_ratio, failure_ratio,
            time_ratio, False, "abort",
        )

    return PlanHealth(
        completion_ratio, failure_ratio,
        time_ratio, True, "continue",
    )
