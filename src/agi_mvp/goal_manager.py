# goal_manager.py -- Multi-goal orchestration

from typing import Optional
import asyncio
import logging

from .planner import Planner
from .planner_types import Goal, GoalPriority, TaskStatus

logger = logging.getLogger(__name__)


class GoalManager:
    """Manages multiple concurrent goals with
    priority arbitration."""

    def __init__(self, planner: Planner,
                 max_concurrent: int = 1):
        self.planner = planner
        self.max_concurrent = max_concurrent
        self._running: set[str] = set()

    async def submit_goal(
        self,
        description: str,
        priority: GoalPriority = GoalPriority.MEDIUM,
        success_criteria: str = "",
        deadline: Optional[float] = None,
    ) -> Goal:
        """Submit a goal and start if capacity allows."""
        goal = self.planner.add_goal(
            description, priority,
            success_criteria, deadline,
        )

        if len(self._running) < self.max_concurrent:
            asyncio.create_task(self._run_goal(goal))
        else:
            logger.info(
                f"Goal '{description}' queued -- "
                f"{len(self._running)} already running"
            )

        return goal

    async def _run_goal(self, goal: Goal):
        """Execute a goal and pick up the next one."""
        self._running.add(goal.id)
        try:
            success = await self.planner.run_goal(goal)
            status = "succeeded" if success else "failed"
            logger.info(
                f"Goal '{goal.description}' {status}"
            )
        finally:
            self._running.discard(goal.id)
            next_goal = self.planner.next_goal()
            if (next_goal
                    and next_goal.status == TaskStatus.PENDING):
                asyncio.create_task(
                    self._run_goal(next_goal)
                )

    async def preempt(self, new_goal: Goal):
        """Interrupt current work for higher priority."""
        active = [
            g for g in self.planner._goals
            if g.status == TaskStatus.IN_PROGRESS
        ]
        for g in active:
            if g.priority.value < new_goal.priority.value:
                g.status = TaskStatus.PENDING
                logger.info(
                    f"Suspending '{g.description}' for "
                    f"higher-priority goal"
                )

        await self._run_goal(new_goal)

    def status_report(self) -> str:
        """Human-readable summary of all goals."""
        lines = ["## Goal Status"]
        for goal in self.planner._goals:
            progress = ""
            if goal.task_tree:
                ratio = goal.task_tree.completion_ratio()
                progress = f" ({ratio:.0%} complete)"
            overdue = (
                " [OVERDUE]" if goal.is_overdue else ""
            )
            lines.append(
                f"- [{goal.priority.name}] "
                f"{goal.description} -- "
                f"{goal.status.value}{progress}{overdue}"
            )
        return (
            "\n".join(lines)
            if len(lines) > 1
            else "No active goals."
        )
