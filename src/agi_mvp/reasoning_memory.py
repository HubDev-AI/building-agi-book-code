# reasoning_memory.py

from .memory_manager import MemoryManager
from .episodic_memory import Episode
from .reasoning_types import ReasoningResult, ReasoningMode


async def store_reasoning_episode(
    memory: MemoryManager,
    question: str,
    result: ReasoningResult,
):
    """Store a completed reasoning episode."""
    outcome = (
        "success" if result.confidence > 0.7 else "failure"
    )

    lesson = ""
    if result.mode == ReasoningMode.HYBRID:
        if result.verification.value == "verified":
            lesson = (
                f"Hybrid reasoning verified after "
                f"{result.attempts} attempt(s). "
                f"Strategy worked."
            )
        elif result.verification.value == "refuted":
            lesson = (
                f"Neural model could not produce a "
                f"verified answer in {result.attempts} "
                f"attempts. Consider direct symbolic "
                f"solving."
            )
    elif result.mode == ReasoningMode.NEURAL:
        lesson = (
            f"Pure neural reasoning with confidence "
            f"{result.confidence:.0%}. No formal "
            f"verification applied."
        )

    episode = Episode(
        task_description=f"Reasoning: {question[:200]}",
        actions_taken=[
            f"Mode: {result.mode.value}",
            f"Attempts: {result.attempts}",
            f"Verification: {result.verification.value}",
        ],
        outcome=outcome,
        lesson_learned=lesson,
        importance=(
            0.7 if result.attempts > 1 else 0.4
        ),
    )

    await memory.learn_from_episode(episode)
