# planner.py -- The planning system

from dataclasses import dataclass, field
from typing import Any, Optional, Callable, Awaitable
import json
import time
import logging

from .foundation import (
    FoundationModel, Message, GenerationConfig,
)
from .memory_manager import MemoryManager
from .episodic_memory import Episode
from .planner_types import (
    Task, Goal, PlanStep, TaskStatus, GoalPriority,
)

logger = logging.getLogger(__name__)


class Planner:
    """Hierarchical planner with LLM-guided decomposition
    and structured execution."""

    def __init__(
        self,
        foundation: FoundationModel,
        memory: MemoryManager,
        action_executor: Optional[
            Callable[[Task], Awaitable[Any]]
        ] = None,
        max_replan_attempts: int = 3,
    ):
        self.foundation = foundation
        self.memory = memory
        self.execute_action = (
            action_executor or self._default_executor
        )
        self.max_replan_attempts = max_replan_attempts
        self._goals: list[Goal] = []

    # -- Goal management --

    def add_goal(
        self,
        description: str,
        priority: GoalPriority = GoalPriority.MEDIUM,
        success_criteria: str = "",
        deadline: Optional[float] = None,
    ) -> Goal:
        """Register a new goal."""
        goal = Goal(
            description=description,
            priority=priority,
            success_criteria=success_criteria,
            deadline=deadline,
        )
        self._goals.append(goal)
        self._goals.sort(
            key=lambda g: g.priority.value, reverse=True
        )
        logger.info(
            f"Goal added: [{goal.priority.name}] "
            f"{description} (id={goal.id})"
        )
        return goal

    def next_goal(self) -> Optional[Goal]:
        """Pick the highest-priority active goal."""
        active = [
            g for g in self._goals
            if g.status in (
                TaskStatus.PENDING, TaskStatus.IN_PROGRESS
            )
        ]
        if not active:
            return None
        overdue = [g for g in active if g.is_overdue]
        if overdue:
            return max(
                overdue, key=lambda g: g.priority.value
            )
        return active[0]

    # -- Decomposition: goal -> task tree --

    async def decompose(self, goal: Goal,
                        context: str = "") -> Task:
        """Break a goal into a hierarchical task tree."""
        memory_context = await self.memory.remember(
            f"plan for: {goal.description}", context,
        )
        past_failures = await self._recall_failures(
            goal.description
        )

        prompt = f"""Decompose this goal into a hierarchical task tree.

GOAL: {goal.description}
SUCCESS CRITERIA: {goal.success_criteria or "Goal is fully achieved"}

{memory_context}
{past_failures}
{f"ADDITIONAL CONTEXT: {context}" if context else ""}

Return a JSON task tree. Each task has:
- "name": short action name
- "description": what to do (1-2 sentences)
- "children": subtasks (empty array for leaf tasks)
- "dependencies": list of sibling task names that must complete first
- "estimated_seconds": rough time estimate for leaf tasks

Rules:
1. Leaf tasks must be concrete, executable actions.
2. Keep depth <= 3 levels.
3. Order siblings by natural execution sequence.
4. Mark dependencies only where strict ordering is required.

Respond with ONLY the JSON object, no markdown fencing."""

        result = await self.foundation.generate(
            [Message(role="user", content=prompt)],
            GenerationConfig(
                temperature=0.4, max_tokens=4096,
                force_reasoning=True,
            ),
        )

        tree = self._parse_task_tree(result.content)
        goal.task_tree = tree
        logger.info(
            f"Decomposed goal '{goal.description}' into "
            f"{len(tree.all_leaves())} leaf tasks"
        )
        return tree

    def _parse_task_tree(self, raw_json: str) -> Task:
        """Parse LLM output into a Task tree."""
        cleaned = raw_json.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse task tree JSON, "
                "creating flat plan"
            )
            return Task(
                name="execute_goal", description=raw_json
            )

        return self._dict_to_task(data)

    def _dict_to_task(self, d: dict) -> Task:
        """Recursively convert a dict to a Task tree."""
        children = [
            self._dict_to_task(c)
            for c in d.get("children", [])
        ]
        task = Task(
            name=d.get("name", "unnamed"),
            description=d.get("description", ""),
            children=children,
            estimated_seconds=d.get("estimated_seconds", 0),
        )
        dep_names = d.get("dependencies", [])
        name_to_id = {c.name: c.id for c in children}
        task.dependencies = [
            name_to_id[n]
            for n in dep_names if n in name_to_id
        ]
        return task

    # -- Plan generation: task tree -> ordered plan --

    def generate_plan(self, tree: Task) -> list[PlanStep]:
        """Flatten a task tree into an ordered plan
        via topological sort."""
        leaves = tree.all_leaves()
        if not leaves:
            return []

        ordered = self._topological_sort(leaves, tree)

        return [
            PlanStep(
                task_id=t.id,
                task_name=t.name,
                description=t.description,
                estimated_seconds=t.estimated_seconds,
            )
            for t in ordered
        ]

    def _topological_sort(
        self, leaves: list[Task], root: Task
    ) -> list[Task]:
        """Sort leaf tasks respecting dependencies."""
        id_to_task = {t.id: t for t in leaves}
        visited: set[str] = set()
        result: list[Task] = []

        def visit(task: Task):
            if task.id in visited:
                return
            visited.add(task.id)
            for dep_id in task.dependencies:
                if dep_id in id_to_task:
                    visit(id_to_task[dep_id])
            result.append(task)

        for leaf in leaves:
            visit(leaf)

        return result

    # -- Execution: run the plan with monitoring --

    async def execute_plan(
        self, plan: list[PlanStep], goal: Goal
    ) -> bool:
        """Execute a plan step by step.
        Returns True if the goal succeeds."""
        goal.status = TaskStatus.IN_PROGRESS
        replan_count = 0
        step_index = 0

        while step_index < len(plan):
            step = plan[step_index]
            task = (
                goal.task_tree.find(step.task_id)
                if goal.task_tree else None
            )
            if task is None:
                step_index += 1
                continue

            if not self._dependencies_met(
                task, goal.task_tree
            ):
                task.status = TaskStatus.BLOCKED
                logger.warning(
                    f"Task '{task.name}' blocked "
                    f"on dependencies"
                )
                step_index += 1
                continue

            task.status = TaskStatus.IN_PROGRESS
            logger.info(
                f"Executing [{step_index+1}/{len(plan)}]: "
                f"{task.name}"
            )

            try:
                result = await self.execute_action(task)
                task.result = result
                task.status = TaskStatus.COMPLETED
                logger.info(f"Completed: {task.name}")

            except Exception as e:
                task.error = str(e)
                task.retries += 1
                logger.error(
                    f"Failed: {task.name} -- {e}"
                )

                if task.retries <= task.max_retries:
                    logger.info(
                        f"Retrying {task.name} "
                        f"({task.retries}/{task.max_retries})"
                    )
                    continue

                task.status = TaskStatus.FAILED

                if replan_count < self.max_replan_attempts:
                    logger.info(
                        f"Replanning after failure "
                        f"(attempt {replan_count+1})"
                    )
                    new_plan = await self.replan(
                        goal, task, str(e)
                    )
                    if new_plan:
                        plan = new_plan
                        step_index = 0
                        replan_count += 1
                        continue
                    else:
                        logger.error(
                            "Replanning produced "
                            "no viable alternative"
                        )

                await self._record_episode(
                    goal, "failure",
                    f"Failed at: {task.name} -- {e}",
                )
                goal.status = TaskStatus.FAILED
                return False

            step_index += 1

        success = (
            goal.task_tree.completion_ratio() > 0.8
            if goal.task_tree else False
        )
        goal.status = (
            TaskStatus.COMPLETED if success
            else TaskStatus.FAILED
        )
        outcome = "success" if success else "partial"
        await self._record_episode(
            goal, outcome,
            f"Completed "
            f"{goal.task_tree.completion_ratio():.0%}",
        )
        return success

    def _dependencies_met(
        self, task: Task, root: Task
    ) -> bool:
        """Check whether all dependencies have completed."""
        for dep_id in task.dependencies:
            dep = root.find(dep_id)
            if dep and dep.status != TaskStatus.COMPLETED:
                return False
        return True

    # -- Replanning --

    async def replan(
        self,
        goal: Goal,
        failed_task: Task,
        error: str,
    ) -> Optional[list[PlanStep]]:
        """Generate a new plan that routes around failure."""
        if not goal.task_tree:
            return None

        completed = [
            t for t in goal.task_tree.all_leaves()
            if t.status == TaskStatus.COMPLETED
        ]
        pending = [
            t for t in goal.task_tree.all_leaves()
            if t.status == TaskStatus.PENDING
        ]

        prompt = f"""A plan has partially failed. Generate a revised plan.

GOAL: {goal.description}

COMPLETED STEPS (do NOT repeat these):
{chr(10).join(f"- {t.name}: {t.description}" for t in completed) or "None yet"}

FAILED STEP:
- {failed_task.name}: {failed_task.description}
- Error: {error}

REMAINING STEPS (may need modification):
{chr(10).join(f"- {t.name}: {t.description}" for t in pending) or "None"}

Generate a revised JSON task list (flat array of objects
with "name", "description", "estimated_seconds") that:
1. Does NOT repeat completed work
2. Works around the failure
3. Still achieves the original goal

Respond with ONLY the JSON array."""

        result = await self.foundation.generate(
            [Message(role="user", content=prompt)],
            GenerationConfig(
                temperature=0.5, max_tokens=2048,
                force_reasoning=True,
            ),
        )

        try:
            cleaned = result.content.strip()
            if cleaned.startswith("```"):
                cleaned = (
                    cleaned.split("\n", 1)[1]
                    .rsplit("```", 1)[0]
                )
            steps_data = json.loads(cleaned)

            new_root = Task(
                name="revised_plan",
                description="Revised after failure",
            )
            for sd in steps_data:
                new_root.children.append(Task(
                    name=sd.get("name", "step"),
                    description=sd.get("description", ""),
                    estimated_seconds=sd.get(
                        "estimated_seconds", 0
                    ),
                ))

            goal.task_tree = new_root
            return self.generate_plan(new_root)

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(
                f"Failed to parse revised plan: {e}"
            )
            return None

    # -- Memory integration --

    async def _recall_failures(
        self, goal_description: str
    ) -> str:
        """Check memory for past failures on similar goals."""
        failures = self.memory.episodic.recall_failures(
            goal_description, top_k=3
        )
        if not failures:
            return ""
        lines = ["PAST FAILURES TO AVOID:"]
        for ep in failures:
            lines.append(
                f"- {ep.task_description}: "
                f"{ep.lesson_learned}"
            )
        return "\n".join(lines)

    async def _record_episode(
        self, goal: Goal, outcome: str, lesson: str
    ):
        """Store the planning episode for future reference."""
        actions = []
        if goal.task_tree:
            for t in goal.task_tree.all_leaves():
                marker = (
                    "+"
                    if t.status == TaskStatus.COMPLETED
                    else "-"
                )
                actions.append(f"[{marker}] {t.name}")

        episode = Episode(
            task_description=f"Plan: {goal.description}",
            actions_taken=actions,
            outcome=outcome,
            lesson_learned=lesson,
            importance=(
                0.8 if outcome == "failure" else 0.6
            ),
        )
        await self.memory.learn_from_episode(episode)

    async def _default_executor(self, task: Task) -> Any:
        """Placeholder executor."""
        logger.info(
            f"[default executor] Would execute: "
            f"{task.name} -- {task.description}"
        )
        return {"status": "simulated", "task": task.name}

    # -- High-level API --

    async def run_goal(
        self, goal: Goal, context: str = ""
    ) -> bool:
        """Full pipeline: decompose -> plan -> execute."""
        tree = await self.decompose(goal, context)
        plan = self.generate_plan(tree)

        if not plan:
            logger.warning(
                f"Decomposition produced no executable "
                f"steps for: {goal.description}"
            )
            goal.status = TaskStatus.FAILED
            return False

        logger.info(
            f"Plan for '{goal.description}': "
            f"{len(plan)} steps, "
            f"~{sum(s.estimated_seconds for s in plan):.0f}s "
            f"estimated"
        )
        for i, step in enumerate(plan, 1):
            logger.info(
                f"  {i}. {step.task_name}: "
                f"{step.description}"
            )

        return await self.execute_plan(plan, goal)
