# reasoning_router.py

import re
from .reasoning_types import ReasoningMode


SYMBOLIC_PATTERNS = [
    r"schedul|allocat|assign.*constraint",
    r"satisf[iy]|feasib",
    r"prove|disprove|theorem",
    r"all\s+\w+\s+(must|should|are|satisfy)",
    r"no\s+(overlap|conflict|violation)",
    r"if\s+and\s+only\s+if",
]

NEURAL_ONLY_PATTERNS = [
    r"explain|describe|discuss|compare",
    r"write|draft|compose|suggest",
    r"summarize|opinion|perspective",
]

_symbolic_re = re.compile(
    "|".join(SYMBOLIC_PATTERNS), re.IGNORECASE
)
_neural_re = re.compile(
    "|".join(NEURAL_ONLY_PATTERNS), re.IGNORECASE
)


def select_reasoning_mode(
    question: str,
    has_constraints: bool = False,
) -> ReasoningMode:
    """Select the appropriate reasoning mode."""
    if has_constraints:
        return ReasoningMode.HYBRID

    if _symbolic_re.search(question):
        return ReasoningMode.HYBRID

    if _neural_re.search(question):
        return ReasoningMode.NEURAL

    return ReasoningMode.NEURAL
