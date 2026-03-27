# health.py -- System health checks

from .foundation import Message, GenerationConfig
from .cognitive_core import CognitiveCore

async def health_check(core: CognitiveCore) -> dict:
    """Run health checks on all components."""
    checks = {}

    # Foundation model
    try:
        result = await core.foundation.generate(
            [Message(role="user", content="Say 'ok'")],
            GenerationConfig(max_tokens=5),
        )
        checks["foundation"] = (
            "ok" if "ok" in result.content.lower()
            else "degraded"
        )
    except Exception as e:
        checks["foundation"] = f"error: {e}"

    # Memory
    checks["working_memory"] = (
        f"ok ({len(core.memory.working._store)} items)"
    )
    try:
        core.memory.episodic.recall("test", top_k=1)
        checks["episodic_memory"] = "ok"
    except Exception as e:
        checks["episodic_memory"] = f"error: {e}"

    # World model
    checks["world_model"] = (
        f"ok ({len(core.world_model.state.entities)}"
        f" entities)"
    )

    # Meta-cognition
    perf = core.meta.get_performance_summary()
    checks["metacognition"] = (
        f"ok (success_rate: "
        f"{perf.get('success_rate', 'N/A')})"
    )

    return checks
