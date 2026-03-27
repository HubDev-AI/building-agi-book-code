# reasoning_engine.py

import json
import re
from dataclasses import dataclass
from typing import Optional

from .reasoning_types import (
    Constraint, ReasoningMode, ReasoningResult,
    VerificationStatus,
)
from .symbolic_verifier import SymbolicVerifier
from .foundation import (
    FoundationModel, GenerationConfig, Message, ModelTier,
)
from .reasoning_utils import parse_reasoning
from .memory_manager import MemoryManager


@dataclass
class ReasoningConfig:
    max_retries: int = 3
    verification_timeout_ms: int = 5000
    temperature: float = 0.6       # Lower for reasoning
    max_reasoning_tokens: int = 8192
    use_memory: bool = True


class ReasoningEngine:
    """Hybrid neural-symbolic reasoning engine.

    Combines chain-of-thought reasoning (via the foundation
    model's reasoning tier) with Z3-based formal verification.
    Queries episodic memory for past reasoning strategies.
    """

    def __init__(
        self,
        foundation: FoundationModel,
        memory: Optional[MemoryManager] = None,
        config: ReasoningConfig = ReasoningConfig(),
    ):
        self.foundation = foundation
        self.memory = memory
        self.verifier = SymbolicVerifier(
            timeout_ms=config.verification_timeout_ms
        )
        self.config = config

    async def reason(
        self,
        question: str,
        context: str = "",
    ) -> ReasoningResult:
        """Pure neural reasoning using chain-of-thought."""
        memory_context = ""
        if self.memory and self.config.use_memory:
            memory_context = await self._recall_strategies(
                question
            )

        system_prompt = (
            "You are a rigorous reasoning engine. Think "
            "step by step. Show all work. State your "
            "assumptions explicitly. If you are uncertain, "
            "say so and explain why."
        )

        user_content = question
        if context:
            user_content = (
                f"Context:\n{context}\n\n"
                f"Question:\n{question}"
            )
        if memory_context:
            user_content = (
                f"{memory_context}\n\n{user_content}"
            )

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_content),
        ]

        result = await self.foundation.generate(
            messages,
            GenerationConfig(
                force_reasoning=True,
                temperature=self.config.temperature,
                max_tokens=self.config.max_reasoning_tokens,
            ),
        )

        parsed = parse_reasoning(result)

        return ReasoningResult(
            answer=parsed.conclusion,
            mode=ReasoningMode.NEURAL,
            confidence=parsed.confidence,
            reasoning_trace=parsed.thinking,
            attempts=1,
        )

    async def verify(
        self,
        claim: str,
        constraints: list[Constraint],
        context_constraints: list[Constraint] | None = None,
    ) -> ReasoningResult:
        """Verify a claim against formal constraints."""
        vresult = self.verifier.verify(
            constraints, context_constraints
        )

        return ReasoningResult(
            answer=claim,
            mode=ReasoningMode.SYMBOLIC,
            confidence=(
                1.0
                if vresult.status == VerificationStatus.VERIFIED
                else 0.0
            ),
            verification=vresult.status,
            verification_detail=vresult.detail,
            z3_model=(
                str(vresult.model) if vresult.model else None
            ),
        )

    async def reason_and_verify(
        self,
        question: str,
        constraints: list[Constraint],
        context: str = "",
        context_constraints: list[Constraint] | None = None,
    ) -> ReasoningResult:
        """The full hybrid loop: neural proposes, symbolic
        checks.

        1. Neural model reasons about the question (CoT).
        2. Neural model translates its answer into Z3
           constraints.
        3. Z3 checks those constraints against the provided
           formal constraints.
        4. If verification fails, error fed back; model retries.
        5. Returns the best result after max_retries attempts.
        """
        memory_context = ""
        if self.memory and self.config.use_memory:
            memory_context = await self._recall_strategies(
                question
            )

        best_result: ReasoningResult | None = None
        feedback = ""

        for attempt in range(1, self.config.max_retries + 1):
            system_prompt = (
                "You are a rigorous reasoning engine inside "
                "a hybrid neural-symbolic system. Your answer "
                "WILL be checked by a formal verifier.\n"
                "Think step by step, then provide:\n"
                "1. Your reasoning\n"
                "2. Your answer\n"
                "3. A JSON block with Z3 constraints:\n"
                '```z3\n[{"name": "...", "expression": '
                '"...", "description": "..."}]\n```\n'
                "Available Z3 functions: Int, Real, Bool, "
                "And, Or, Not, Implies, If, Distinct, "
                "IntVal, RealVal."
            )

            user_content = question
            if context:
                user_content = (
                    f"Context:\n{context}\n\n"
                    f"Question:\n{question}"
                )
            if memory_context:
                user_content = (
                    f"{memory_context}\n\n{user_content}"
                )

            constraint_desc = "\n".join(
                f"- {c.name}: {c.description}"
                for c in constraints
            )
            user_content += (
                f"\n\nFormal constraints to satisfy:\n"
                f"{constraint_desc}"
            )

            if feedback:
                user_content += (
                    f"\n\nPrevious attempt failed. "
                    f"Error:\n{feedback}\nPlease fix."
                )

            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_content),
            ]

            result = await self.foundation.generate(
                messages,
                GenerationConfig(
                    force_reasoning=True,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_reasoning_tokens,
                ),
            )

            parsed = parse_reasoning(result)

            answer_constraints = (
                self._extract_z3_constraints(result.content)
            )

            all_claim = constraints + answer_constraints
            vresult = self.verifier.verify(
                all_claim, context_constraints
            )

            reasoning_result = ReasoningResult(
                answer=parsed.conclusion,
                mode=ReasoningMode.HYBRID,
                confidence=(
                    parsed.confidence
                    if vresult.status
                    == VerificationStatus.VERIFIED
                    else parsed.confidence * 0.5
                ),
                reasoning_trace=parsed.thinking,
                verification=vresult.status,
                verification_detail=vresult.detail,
                attempts=attempt,
                z3_model=(
                    str(vresult.model)
                    if vresult.model else None
                ),
            )

            if (best_result is None
                    or reasoning_result.confidence
                    > best_result.confidence):
                best_result = reasoning_result

            if vresult.status == VerificationStatus.VERIFIED:
                reasoning_result.confidence = min(
                    1.0, parsed.confidence + 0.2
                )
                return reasoning_result

            feedback = vresult.detail

        return best_result

    def _extract_z3_constraints(
        self, model_output: str
    ) -> list[Constraint]:
        """Parse Z3 constraint JSON from model response."""
        pattern = r"```z3\s*(.*?)\s*```"
        match = re.search(pattern, model_output, re.DOTALL)
        if not match:
            pattern = r"```(?:json)?\s*(\[.*?\])\s*```"
            match = re.search(
                pattern, model_output, re.DOTALL
            )

        if not match:
            return []

        try:
            data = json.loads(match.group(1))
            return [
                Constraint(
                    name=item.get(
                        "name", f"model_constraint_{i}"
                    ),
                    expression=item["expression"],
                    description=item.get("description", ""),
                )
                for i, item in enumerate(data)
            ]
        except (json.JSONDecodeError, KeyError):
            return []

    async def _recall_strategies(
        self, question: str
    ) -> str:
        """Query episodic memory for past strategies."""
        if not self.memory:
            return ""

        episodes = self.memory.episodic.recall(
            f"reasoning strategy for: {question}", top_k=3
        )

        if not episodes:
            return ""

        lines = [
            "## Past Reasoning Strategies (from memory)"
        ]
        for ep in episodes:
            lines.append(
                f"- Task: {ep.task_description} | "
                f"Outcome: {ep.outcome} | "
                f"Lesson: {ep.lesson_learned}"
            )
        return "\n".join(lines)
