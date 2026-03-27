# working_memory.py

from dataclasses import dataclass, field
from collections import deque
from typing import Any, Optional
import time

@dataclass
class WorkingMemoryItem:
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    importance: float = 0.5  # 0-1 scale
    source: str = "unknown"  # Which layer produced this

class WorkingMemory:
    """Fast, volatile memory for current cognitive processing."""

    def __init__(self, max_items: int = 100,
                 max_history: int = 50):
        self._store: dict[str, WorkingMemoryItem] = {}
        self._history: deque[dict] = deque(maxlen=max_history)
        self._max_items = max_items

    def set(self, key: str, value: Any,
            importance: float = 0.5, source: str = "unknown"):
        """Store or update a working memory item."""
        self._store[key] = WorkingMemoryItem(
            key=key, value=value,
            importance=importance, source=source,
        )
        self._history.append({
            "action": "set", "key": key, "time": time.time(),
        })
        self._evict_if_needed()

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a working memory item."""
        item = self._store.get(key)
        if item:
            item.access_count += 1
            return item.value
        return None

    def get_context_window(self, max_tokens: int = 4000) -> str:
        """Serialize working memory for LLM context injection."""
        items = sorted(
            self._store.values(),
            key=lambda x: (x.importance, x.access_count),
            reverse=True,
        )
        lines = ["## Current Working Memory"]
        token_estimate = 10  # header

        for item in items:
            line = f"- **{item.key}**: {item.value}"
            line_tokens = len(line.split()) * 1.3
            if token_estimate + line_tokens > max_tokens:
                break
            lines.append(line)
            token_estimate += line_tokens

        return "\n".join(lines)

    def get_state(self) -> dict:
        """Full state snapshot for the meta-cognition layer."""
        return {
            "item_count": len(self._store),
            "items": {
                k: {"value": v.value, "importance": v.importance}
                for k, v in self._store.items()
            },
            "recent_actions": list(self._history)[-10:],
        }

    def clear_task(self, task_id: str):
        """Clear items associated with a completed task."""
        to_remove = [
            k for k, v in self._store.items()
            if v.source == task_id
        ]
        for k in to_remove:
            del self._store[k]

    def _evict_if_needed(self):
        """Remove lowest-importance items when at capacity."""
        while len(self._store) > self._max_items:
            least = min(
                self._store.values(),
                key=lambda x: (
                    x.importance * 0.7
                    + (x.access_count / 10) * 0.3
                ),
            )
            del self._store[least.key]
