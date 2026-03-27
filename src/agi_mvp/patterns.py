# patterns.py -- Multi-model orchestration
#
# Book reference: Chapter 5 "Multi-Model Orchestration Patterns"

from .foundation import (
    FoundationModel,
    GenerationConfig,
    GenerationResult,
    Message,
)


async def generate_and_verify(
    foundation: FoundationModel,
    messages: list[Message],
) -> GenerationResult:
    """Generate with fast model, verify with reasoning model."""
    # Step 1: Fast generation
    fast_result = await foundation.generate(
        messages,
        GenerationConfig(force_reasoning=False),
    )

    # Step 2: Verify with reasoning model
    verify_messages = messages + [
        Message(role="assistant", content=fast_result.content),
        Message(role="user", content=(
            "Verify the above response. Is it correct and "
            "complete? If there are errors, provide the "
            "corrected version. If correct, respond with "
            "'VERIFIED: ' followed by the original response."
        )),
    ]

    verify_result = await foundation.generate(
        verify_messages,
        GenerationConfig(
            force_reasoning=True, max_tokens=8192
        ),
    )

    if verify_result.content.startswith("VERIFIED:"):
        return fast_result  # Original was correct
    else:
        return verify_result  # Use the corrected version
