# evaluation/scorecard.py

from dataclasses import dataclass


@dataclass
class CapabilityScore:
    """Score for a single capability dimension (0.0 - 1.0)."""
    dimension: str
    score: float
    confidence: float        # How many tasks contributed
    benchmark_sources: list[str]
    weaknesses: list[str]    # Identified failure patterns


def build_scorecard(
    report: 'EvaluationReport',
) -> dict[str, CapabilityScore]:
    """Map evaluation results to the seven architectural
    layers."""

    layer_mapping = {
        "perception":
            "Layer 1: Foundation Model",
        "reasoning":
            "Layer 3: Reasoning Engine",
        "symbolic_verification":
            "Layer 3: Reasoning Engine",
        "noise_filtering":
            "Layer 3: Reasoning Engine",
        "memory":
            "Layer 2: Memory System",
        "planning":
            "Layer 4: Planner",
        "tool_use":
            "Layer 5: Perception & Action",
        "agency":
            "Layer 5: Perception & Action",
        "world_model":
            "Layer 6: World Model",
        "metacognition":
            "Layer 7: Meta-Cognition",
    }

    scorecard = {}
    for dim, dim_score in report.dimension_scores.items():
        layer = layer_mapping.get(dim, "Unknown")
        weaknesses = []

        if dim_score.accuracy < 0.5:
            weaknesses.append(
                f"Below 50% accuracy on {dim} tasks"
            )
        if dim_score.avg_latency_ms > 30000:
            weaknesses.append(
                f"High latency: "
                f"{dim_score.avg_latency_ms:.0f}ms avg"
            )

        scorecard[dim] = CapabilityScore(
            dimension=dim,
            score=dim_score.accuracy,
            confidence=min(
                1.0, dim_score.tasks_tested / 10
            ),
            benchmark_sources=[layer],
            weaknesses=weaknesses,
        )

    return scorecard


def print_scorecard(
    scorecard: dict[str, 'CapabilityScore'],
):
    """Print a human-readable capability scorecard."""
    print("\n" + "=" * 60)
    print("         AGI MVP CAPABILITY SCORECARD")
    print("=" * 60)

    for dim, score in sorted(
        scorecard.items(),
        key=lambda x: x[1].score,
        reverse=True,
    ):
        bar_len = int(score.score * 30)
        bar = "#" * bar_len + "." * (30 - bar_len)
        confidence_marker = (
            "***" if score.confidence > 0.8 else
            "** " if score.confidence > 0.5 else
            "*  "
        )
        print(
            f"  {dim:<25} [{bar}] "
            f"{score.score:.0%} {confidence_marker}"
        )
        for w in score.weaknesses:
            print(f"    WARNING: {w}")

    print("=" * 60)
    print("  Confidence: *** high (10+ tasks) "
          " ** medium  * low")
    print("=" * 60)
