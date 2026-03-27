# plan_memory.py -- Learning from past plans

from typing import Optional
from .memory_manager import MemoryManager
from .planner_types import Goal, Task, TaskStatus


async def enrich_decomposition_context(
    memory: MemoryManager,
    goal_description: str,
) -> str:
    """Pull all relevant planning knowledge from memory."""
    sections = []

    # 1. Successful past plans
    successes = memory.episodic.recall_successes(
        f"plan for: {goal_description}", top_k=2,
    )
    if successes:
        lines = ["SUCCESSFUL PAST APPROACHES:"]
        for ep in successes:
            lines.append(f"- Goal: {ep.task_description}")
            lines.append(
                f"  Steps: {'; '.join(ep.actions_taken)}"
            )
            lines.append(
                f"  Lesson: {ep.lesson_learned}"
            )
        sections.append("\n".join(lines))

    # 2. Failed past plans (what to avoid)
    failures = memory.episodic.recall_failures(
        f"plan for: {goal_description}", top_k=2,
    )
    if failures:
        lines = ["APPROACHES THAT FAILED (avoid these):"]
        for ep in failures:
            lines.append(f"- Goal: {ep.task_description}")
            lines.append(
                f"  What went wrong: {ep.lesson_learned}"
            )
        sections.append("\n".join(lines))

    # 3. Relevant domain knowledge
    facts = memory.semantic.search_semantic(
        goal_description, top_k=5
    )
    if facts:
        lines = ["RELEVANT KNOWLEDGE:"]
        for f in facts:
            lines.append(f"- {f.to_triple()}")
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def extract_plan_patterns(goal: Goal) -> Optional[dict]:
    """After goal completion, extract reusable patterns."""
    if (not goal.task_tree
            or goal.status != TaskStatus.COMPLETED):
        return None

    leaves = goal.task_tree.all_leaves()
    successful_steps = [
        t.name for t in leaves
        if t.status == TaskStatus.COMPLETED
    ]
    failed_steps = [
        t.name for t in leaves
        if t.status == TaskStatus.FAILED
    ]

    return {
        "goal_type": goal.description,
        "step_count": len(leaves),
        "successful_pattern": successful_steps,
        "failed_steps": failed_steps,
        "completion_ratio": (
            goal.task_tree.completion_ratio()
        ),
    }
