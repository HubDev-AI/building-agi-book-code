# safety/human_in_the_loop.py

from dataclasses import dataclass
from typing import Optional
import asyncio


@dataclass
class ApprovalRequest:
    action_description: str
    safety_level: str
    context: str               # Why the system wants this
    alternatives: list[str]    # Other options considered
    risk_assessment: str
    reversible: bool


@dataclass
class ApprovalResponse:
    approved: bool
    modified_action: Optional[str] = None
    feedback: str = ""


class HumanApprovalGate:
    """Gates consequential actions behind human approval.
    Async to avoid blocking while waiting for human
    response. On timeout, the action is DENIED by
    default --- fail-safe."""

    def __init__(self, timeout_seconds: float = 300):
        self.timeout = timeout_seconds

    async def request_approval(
        self, request: ApprovalRequest
    ) -> ApprovalResponse:
        """Block until human responds or timeout."""
        print(f"\n{'=' * 60}")
        print(f"  HUMAN APPROVAL REQUIRED")
        print(f"  Action: {request.action_description}")
        print(
            f"  Level: {request.safety_level}  "
            f"Reversible: {request.reversible}"
        )
        print(f"  Risk: {request.risk_assessment}")
        print(f"{'=' * 60}")

        try:
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: input("  Approve? (y/n): "),
                ),
                timeout=self.timeout,
            )
            return ApprovalResponse(
                approved=response.strip()
                         .lower().startswith("y")
            )
        except asyncio.TimeoutError:
            return ApprovalResponse(
                approved=False,
                feedback="Timeout --- denied",
            )
