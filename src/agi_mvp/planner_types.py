# planner_types.py -- Core planning data structures

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import time
import uuid


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class GoalPriority(Enum):
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1


@dataclass
class Task:
    """A node in the task tree. Leaf tasks are executable."""
    id: str = field(
        default_factory=lambda: str(uuid.uuid4())[:8]
    )
    name: str = ""
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    children: list["Task"] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    estimated_seconds: float = 0.0
    result: Optional[Any] = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 2
    created_at: float = field(default_factory=time.time)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.SKIPPED,
        )

    def find(self, task_id: str) -> Optional["Task"]:
        """Find a task by ID in this subtree."""
        if self.id == task_id:
            return self
        for child in self.children:
            found = child.find(task_id)
            if found:
                return found
        return None

    def all_leaves(self) -> list["Task"]:
        """Collect all leaf tasks (executable actions)."""
        if self.is_leaf:
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.all_leaves())
        return leaves

    def completion_ratio(self) -> float:
        """Fraction of leaf tasks that are completed."""
        leaves = self.all_leaves()
        if not leaves:
            return 0.0
        done = sum(
            1 for t in leaves
            if t.status == TaskStatus.COMPLETED
        )
        return done / len(leaves)


@dataclass
class Goal:
    """A high-level objective the system is pursuing."""
    id: str = field(
        default_factory=lambda: str(uuid.uuid4())[:8]
    )
    description: str = ""
    priority: GoalPriority = GoalPriority.MEDIUM
    success_criteria: str = ""
    task_tree: Optional[Task] = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    deadline: Optional[float] = None

    @property
    def is_overdue(self) -> bool:
        return (
            self.deadline is not None
            and time.time() > self.deadline
        )


@dataclass
class PlanStep:
    """A single step in a linearized execution plan."""
    task_id: str
    task_name: str
    description: str
    dependencies_met: bool = True
    estimated_seconds: float = 0.0
