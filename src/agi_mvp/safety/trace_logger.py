# safety/trace_logger.py

from dataclasses import dataclass, field
from typing import Callable
import time, json, hashlib


@dataclass
class TraceEntry:
    timestamp: float
    component: str           # Which layer produced this
    action_type: str         # "reason", "plan", "tool_call"
    input_summary: str
    output_summary: str
    safety_level: str
    confidence: float
    verification_status: str # "verified" / "unverified"
    metadata: dict = field(default_factory=dict)


class SafetyTraceLogger:
    """Append-only trace logger with hash-chain tamper
    detection and real-time alert firing for
    safety-relevant patterns."""

    def __init__(
        self,
        log_path: str = "traces/safety_trace.jsonl",
    ):
        self.log_path = log_path
        self._entries: list[TraceEntry] = []
        self._alert_callbacks: list[Callable] = []

    def log(self, entry: TraceEntry):
        self._entries.append(entry)
        # Append-only write with hash chain
        prev = (
            self._entries[-2].metadata.get(
                "hash", "genesis"
            )
            if len(self._entries) > 1
            else "genesis"
        )
        entry_hash = hashlib.sha256(
            f"{prev}:{entry.timestamp}"
            f":{entry.component}".encode()
        ).hexdigest()[:16]

        with open(self.log_path, "a") as f:
            f.write(json.dumps({
                "timestamp": entry.timestamp,
                "component": entry.component,
                "action_type": entry.action_type,
                "safety_level": entry.safety_level,
                "confidence": entry.confidence,
                "verification_status":
                    entry.verification_status,
                "hash": entry_hash,
            }) + "\n")

        self._check_alerts(entry)

    def _check_alerts(self, entry: TraceEntry):
        # Unverified consequential action
        if (
            entry.safety_level
            in ("consequential", "critical")
            and entry.verification_status == "unverified"
        ):
            self._fire(
                "UNVERIFIED_CONSEQUENTIAL",
                f"Unverified {entry.safety_level} action",
                entry,
            )
        # Low confidence
        if entry.confidence < 0.3:
            self._fire(
                "LOW_CONFIDENCE",
                f"Action at {entry.confidence:.0%} "
                f"confidence",
                entry,
            )
        # Rapid tool calls (runaway loop detection)
        recent = [
            e for e in self._entries[-10:]
            if e.action_type == "tool_call"
            and e.timestamp > time.time() - 5
        ]
        if len(recent) >= 8:
            self._fire(
                "RAPID_TOOL_CALLS",
                f"{len(recent)} tool calls in 5s",
                entry,
            )

    def _fire(
        self, alert_type: str,
        message: str, entry: TraceEntry,
    ):
        alert = {
            "type": alert_type,
            "message": message,
            "timestamp": entry.timestamp,
        }
        for cb in self._alert_callbacks:
            cb(alert)

    def on_alert(self, callback: callable):
        self._alert_callbacks.append(callback)
