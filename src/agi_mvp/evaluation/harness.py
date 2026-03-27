# evaluation/harness.py

from dataclasses import dataclass, field
from typing import Optional, Callable
import time
import json
import traceback

from .novel_task_generator import EvaluationTask


@dataclass
class TaskResult:
    task_id: str
    category: str
    dimensions_tested: list[str]
    answer: str
    correct: Optional[bool]          # None if no ground truth
    verification_detail: str = ""
    latency_ms: float = 0
    tokens_used: int = 0
    tool_calls: int = 0
    reasoning_steps: int = 0
    error: Optional[str] = None
    trace: dict = field(default_factory=dict)


@dataclass
class DimensionScore:
    dimension: str
    tasks_tested: int
    tasks_passed: int
    accuracy: float
    avg_latency_ms: float
    avg_tokens: float


@dataclass
class EvaluationReport:
    total_tasks: int
    total_passed: int
    overall_accuracy: float
    dimension_scores: dict[str, DimensionScore]
    category_scores: dict[str, float]
    results: list[TaskResult]
    timestamp: str = ""


class EvaluationHarness:
    """Runs evaluation tasks against the AGI cognitive loop."""

    def __init__(self, cognitive_loop: 'CognitiveLoop'):
        self.loop = cognitive_loop
        self.results: list[TaskResult] = []

    async def run_task(
        self, task: EvaluationTask
    ) -> TaskResult:
        """Execute a single evaluation task through
        the cognitive loop."""
        start = time.monotonic()

        try:
            # Run through the full cognitive loop
            response = await self.loop.process(task.description)

            latency_ms = (time.monotonic() - start) * 1000

            answer = (
                response.final_answer
                if hasattr(response, 'final_answer')
                else str(response)
            )

            # Verify if ground truth exists
            correct = None
            detail = "No ground truth available"

            if task.expected_answer is not None:
                if task.verification_fn:
                    try:
                        verify = eval(task.verification_fn)
                        result = verify(answer)
                        if isinstance(result, tuple):
                            correct, detail = result
                        else:
                            correct = bool(result)
                            detail = (
                                "Passed" if correct else "Failed"
                            )
                    except Exception as e:
                        correct = False
                        detail = f"Verification error: {e}"
                else:
                    correct = (
                        answer.strip().lower()
                        == task.expected_answer.strip().lower()
                    )
                    detail = (
                        "Exact match" if correct else "Mismatch"
                    )

            return TaskResult(
                task_id=task.task_id,
                category=task.category,
                dimensions_tested=task.dimensions_tested,
                answer=answer,
                correct=correct,
                verification_detail=detail,
                latency_ms=latency_ms,
                tokens_used=getattr(
                    response, 'tokens_used', 0),
                tool_calls=getattr(
                    response, 'tool_calls', 0),
                reasoning_steps=getattr(
                    response, 'reasoning_steps', 0),
                trace=getattr(response, 'trace', {}),
            )

        except Exception as e:
            return TaskResult(
                task_id=task.task_id,
                category=task.category,
                dimensions_tested=task.dimensions_tested,
                answer="",
                correct=False,
                verification_detail=f"Execution error: {e}",
                latency_ms=(
                    (time.monotonic() - start) * 1000
                ),
                error=traceback.format_exc(),
            )

    async def run_suite(
        self,
        tasks: list[EvaluationTask],
        concurrency: int = 1,
    ) -> EvaluationReport:
        """Run a full evaluation suite and produce a report."""
        import asyncio

        if concurrency == 1:
            results = []
            for task in tasks:
                result = await self.run_task(task)
                results.append(result)
                status = (
                    "PASS" if result.correct else
                    ("FAIL" if result.correct is False
                     else "SKIP")
                )
                print(
                    f"  [{status}] {task.task_id} "
                    f"({task.category}, "
                    f"{result.latency_ms:.0f}ms)"
                )
        else:
            sem = asyncio.Semaphore(concurrency)

            async def _bounded(t):
                async with sem:
                    return await self.run_task(t)

            results = await asyncio.gather(
                *[_bounded(t) for t in tasks]
            )

        self.results.extend(results)
        return self._compile_report(results)

    def _compile_report(
        self, results: list[TaskResult]
    ) -> EvaluationReport:
        """Aggregate results into dimension and category scores."""
        # Dimension-level scoring
        dim_data: dict[str, list[TaskResult]] = {}
        for r in results:
            for dim in r.dimensions_tested:
                dim_data.setdefault(dim, []).append(r)

        dimension_scores = {}
        for dim, dim_results in dim_data.items():
            scoreable = [
                r for r in dim_results
                if r.correct is not None
            ]
            passed = sum(1 for r in scoreable if r.correct)
            dimension_scores[dim] = DimensionScore(
                dimension=dim,
                tasks_tested=len(scoreable),
                tasks_passed=passed,
                accuracy=(
                    passed / len(scoreable)
                    if scoreable else 0.0
                ),
                avg_latency_ms=(
                    sum(r.latency_ms for r in dim_results)
                    / len(dim_results)
                ),
                avg_tokens=(
                    sum(r.tokens_used for r in dim_results)
                    / len(dim_results)
                ),
            )

        # Category-level scoring
        cat_data: dict[str, list[TaskResult]] = {}
        for r in results:
            cat_data.setdefault(r.category, []).append(r)

        category_scores = {}
        for cat, cat_results in cat_data.items():
            scoreable = [
                r for r in cat_results
                if r.correct is not None
            ]
            if scoreable:
                category_scores[cat] = (
                    sum(1 for r in scoreable if r.correct)
                    / len(scoreable)
                )

        scoreable_all = [
            r for r in results if r.correct is not None
        ]
        total_passed = sum(
            1 for r in scoreable_all if r.correct
        )

        return EvaluationReport(
            total_tasks=len(results),
            total_passed=total_passed,
            overall_accuracy=(
                total_passed / len(scoreable_all)
                if scoreable_all else 0.0
            ),
            dimension_scores=dimension_scores,
            category_scores=category_scores,
            results=results,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
