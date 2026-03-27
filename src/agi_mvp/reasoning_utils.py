# reasoning_utils.py -- Extracting structured reasoning
#
# Book reference: Chapter 5 "Reasoning Mode Activation"

import re
from dataclasses import dataclass

from .foundation import GenerationResult


@dataclass
class ReasoningOutput:
    thinking: str          # The full reasoning trace
    conclusion: str        # The final answer
    confidence: float      # Estimated confidence (0-1)
    steps: list[str]       # Individual reasoning steps
    self_corrections: int  # Times the model corrected itself


def parse_reasoning(result: GenerationResult) -> ReasoningOutput:
    """Parse a reasoning model's output into structured components."""
    thinking = result.reasoning_trace or ""
    conclusion = result.content

    # Extract individual reasoning steps
    steps = re.split(
        r'\n(?=\d+[\.\)]\s|Step\s+\d|First,|Second,|'
        r'Third,|Next,|Finally,)',
        thinking,
    )
    steps = [s.strip() for s in steps if s.strip()]

    # Count self-corrections (indicates the model caught its own errors)
    correction_patterns = [
        r"wait,?\s",
        r"actually,?\s",
        r"let me reconsider",
        r"that's not right",
        r"I made an error",
        r"correction:",
    ]
    self_corrections = sum(
        len(re.findall(p, thinking, re.IGNORECASE))
        for p in correction_patterns
    )

    # Estimate confidence from language cues
    confidence = _estimate_confidence(thinking, conclusion)

    return ReasoningOutput(
        thinking=thinking,
        conclusion=conclusion,
        confidence=confidence,
        steps=steps,
        self_corrections=self_corrections,
    )


def _estimate_confidence(thinking: str, conclusion: str) -> float:
    """Rough confidence estimation from linguistic cues."""
    text = (thinking + " " + conclusion).lower()

    high_confidence = [
        "clearly", "definitely", "certain", "proven", "verified",
    ]
    low_confidence = [
        "uncertain", "might", "possibly", "unclear",
        "not sure", "guess",
    ]

    high_score = sum(1 for w in high_confidence if w in text)
    low_score = sum(1 for w in low_confidence if w in text)

    # Base confidence 0.7, adjusted by language cues
    base = 0.7
    adjustment = (high_score - low_score) * 0.05
    return max(0.1, min(0.99, base + adjustment))
