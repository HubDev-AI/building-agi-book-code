# evaluation/run_evaluation.py

import asyncio
from datetime import datetime

from .novel_task_generator import (
    NovelTaskGenerator,
)
from .harness import EvaluationHarness
from .scorecard import (
    build_scorecard, print_scorecard,
)


async def main():
    # 1. Initialize the cognitive loop (from Chapter 13)
    from cognitive_loop import CognitiveLoop
    loop = CognitiveLoop.from_config("config/default.yaml")

    # 2. Generate novel tasks
    generator = NovelTaskGenerator(
        seed=int(datetime.now().timestamp())
    )
    custom_tasks = generator.generate_suite(
        n_per_category=10
    )
    print(f"Generated {len(custom_tasks)} novel tasks")

    # 3. Run evaluation
    harness = EvaluationHarness(loop)

    print("\n--- Running Novel Task Suite ---")
    report = await harness.run_suite(
        custom_tasks, concurrency=1
    )

    # 4. Build and display scorecard
    scorecard = build_scorecard(report)
    print_scorecard(scorecard)

    # 5. Export results for analysis
    import json
    export = {
        "timestamp": report.timestamp,
        "overall_accuracy": report.overall_accuracy,
        "dimension_scores": {
            dim: {
                "accuracy": s.accuracy,
                "tasks_tested": s.tasks_tested,
                "avg_latency_ms": s.avg_latency_ms,
            }
            for dim, s in
            report.dimension_scores.items()
        },
        "category_scores": report.category_scores,
        "failures": [
            {
                "task_id": r.task_id,
                "category": r.category,
                "answer": r.answer[:200],
                "detail": r.verification_detail,
            }
            for r in report.results
            if r.correct is False
        ],
    }

    with open(
        f"eval_results_"
        f"{report.timestamp.replace(' ','_')}.json",
        "w",
    ) as f:
        json.dump(export, f, indent=2)
    print(
        f"\nResults exported. "
        f"{report.total_passed}/{report.total_tasks} passed."
    )


if __name__ == "__main__":
    asyncio.run(main())
