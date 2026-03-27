# reasoning_types.py

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ReasoningMode(Enum):
    NEURAL = "neural"      # Chain-of-thought only
    SYMBOLIC = "symbolic"  # Z3 constraint solving only
    HYBRID = "hybrid"      # Neural proposes, symbolic verifies


class VerificationStatus(Enum):
    VERIFIED = "verified"
    REFUTED = "refuted"
    UNKNOWN = "unknown"    # Z3 timed out or not expressible


@dataclass
class ReasoningResult:
    answer: str
    mode: ReasoningMode
    confidence: float                  # 0.0 - 1.0
    reasoning_trace: str = ""          # CoT from neural model
    verification: VerificationStatus = (
        VerificationStatus.UNKNOWN
    )
    verification_detail: str = ""
    attempts: int = 1
    z3_model: Optional[str] = None     # Satisfying assignment
    metadata: dict = field(default_factory=dict)


@dataclass
class Constraint:
    """A single formal constraint for Z3 verification."""
    name: str
    expression: str   # Z3 Python expression as a string
    description: str  # Human-readable explanation
