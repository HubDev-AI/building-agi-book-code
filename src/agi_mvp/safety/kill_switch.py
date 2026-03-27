# safety/kill_switch.py

from enum import Enum
from typing import Callable
import time, threading


class SystemState(Enum):
    RUNNING = "running"
    PAUSED = "paused"       # Completes current, stops new
    SUSPENDED = "suspended"  # Interrupts immediately
    SHUTDOWN = "shutdown"    # Full graceful shutdown


class KillSwitch:
    """System-wide kill switch with graduated response
    levels. Checked at the top of every cognitive loop
    iteration."""

    def __init__(self):
        self._state = SystemState.RUNNING
        self._lock = threading.Lock()
        self._callbacks: list[Callable] = []

    @property
    def state(self) -> SystemState:
        return self._state

    def pause(
        self, reason: str, triggered_by: str = "human"
    ):
        self._transition(
            SystemState.PAUSED, reason, triggered_by
        )

    def suspend(
        self, reason: str,
        triggered_by: str = "monitor",
    ):
        self._transition(
            SystemState.SUSPENDED, reason, triggered_by
        )

    def shutdown(
        self, reason: str, triggered_by: str = "human"
    ):
        self._transition(
            SystemState.SHUTDOWN, reason, triggered_by
        )

    def resume(
        self, reason: str, triggered_by: str = "human"
    ):
        if self._state == SystemState.SHUTDOWN:
            raise RuntimeError(
                "Cannot resume from shutdown"
            )
        self._transition(
            SystemState.RUNNING, reason, triggered_by
        )

    def _transition(
        self, new_state: SystemState,
        reason: str, triggered_by: str,
    ):
        with self._lock:
            self._state = new_state
            for cb in self._callbacks:
                cb({
                    "new_state": new_state.value,
                    "reason": reason,
                    "by": triggered_by,
                })

    def check_or_wait(
        self, timeout: float = 60.0
    ) -> bool:
        """Returns True if system should proceed.
        Blocks if paused."""
        if self._state == SystemState.RUNNING:
            return True
        if self._state in (
            SystemState.SUSPENDED, SystemState.SHUTDOWN
        ):
            return False
        start = time.time()
        while self._state == SystemState.PAUSED:
            if time.time() - start > timeout:
                return False
            time.sleep(0.5)
        return self._state == SystemState.RUNNING
