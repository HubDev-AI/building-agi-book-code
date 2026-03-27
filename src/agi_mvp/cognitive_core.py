# cognitive_core.py -- The heart of the AGI MVP

from dataclasses import dataclass
from typing import Optional
import time
import logging

logger = logging.getLogger("agi_mvp")

class CognitiveCore:
    """Orchestrates all seven layers into a unified
    cognitive system.

    The main loop:
    Perceive -> Remember -> Reason -> Plan -> Act -> Learn
    """

    def __init__(
        self,
        foundation: 'FoundationModel',
        memory: 'MemoryManager',
        reasoning: 'ReasoningEngine',
        planner: 'Planner',
        tools: 'ToolManager',
        world_model: 'WorldModel',
        metacognition: 'MetaCognition',
    ):
        self.foundation = foundation
        self.memory = memory
        self.reasoning = reasoning
        self.planner = planner
        self.tools = tools
        self.world_model = world_model
        self.meta = metacognition
        self._turn_count = 0

    async def process(self, user_input: str) -> str:
        """Process a user input through the full
        cognitive loop.

        This is the main entry point for the AGI MVP.
        """
        self._turn_count += 1
        start = time.monotonic()
        logger.info(
            f"Turn {self._turn_count}: "
            f"{user_input[:100]}..."
        )

        # -- 1. META-ASSESS --
        self.meta.state.current_task = user_input
        use_reasoning = \
            self.meta.should_use_reasoning_model(user_input)
        strategy = self.meta.select_strategy(
            self._classify_task(user_input)
        )
        logger.info(
            f"Strategy: {strategy}, "
            f"Reasoning: {use_reasoning}"
        )

        # -- 2. REMEMBER --
        memory_context = await self.memory.remember(
            user_input
        )
        world_context = self.world_model.get_context(
            max_tokens=1500
        )
        full_context = (
            f"{memory_context}\n\n{world_context}"
        )

        # -- 3. REASON --
        if self._is_complex_task(user_input):
            result = await self._plan_and_execute(
                user_input, full_context,
            )
        elif use_reasoning:
            reasoning_result = \
                await self.reasoning.reason_and_verify(
                    user_input,
                    constraints=[],
                    context=full_context,
                )
            result = reasoning_result.answer
        else:
            from .foundation import Message, GenerationConfig
            messages = [
                Message(
                    role="system",
                    content=self._system_prompt(
                        full_context
                    ),
                ),
                Message(role="user", content=user_input),
            ]
            gen_result = await self.foundation.generate(
                messages
            )
            result = gen_result.content

        # -- 4. LEARN --
        elapsed_ms = (time.monotonic() - start) * 1000
        self.meta.state.time_spent_ms = elapsed_ms

        await self.world_model.observe(
            f"User asked: {user_input[:200]}\n"
            f"System responded: {result[:200]}"
        )

        evaluation = await self.meta.evaluate_outcome(
            user_input, result,
        )
        success = evaluation.get("overall", 5) >= 7

        lesson = await self.meta.extract_lesson(
            user_input, result, success,
        )
        if lesson:
            from .episodic_memory import Episode
            episode = Episode(
                task_description=user_input[:200],
                actions_taken=[strategy],
                outcome="success" if success else "failure",
                lesson_learned=lesson,
                importance=0.8 if not success else 0.5,
            )
            await self.memory.learn_from_episode(episode)

        self.memory.working.set(
            "last_task", user_input, importance=0.6,
        )
        self.memory.working.set(
            "last_result", result[:500], importance=0.5,
        )
        self.memory.working.set(
            "last_success", success, importance=0.7,
        )

        logger.info(
            f"Turn {self._turn_count} complete: "
            f"{elapsed_ms:.0f}ms, success={success}, "
            f"confidence={self.meta.state.confidence:.0%}"
        )

        return result

    async def _plan_and_execute(
        self, goal: str, context: str,
    ) -> str:
        """Handle complex multi-step tasks
        via the planner."""
        from .planner import Goal, GoalPriority

        plan_goal = Goal(
            description=goal,
            priority=GoalPriority.HIGH,
            context={"memory": context},
        )

        task_tree = await self.planner.decompose(
            plan_goal, context,
        )
        plan = self.planner.generate_plan(task_tree)

        results = []
        for step in plan:
            step_result = await self._execute_step(step)
            results.append(step_result)

            if not step_result.get("success", True):
                revised = await self.planner.replan(
                    plan_goal, step,
                    step_result.get("error",
                                    "Unknown failure"),
                )
                if revised:
                    plan = revised

        # Synthesize results
        synthesis_prompt = (
            f"Goal: {goal}\n\n"
            f"Steps completed:\n"
            + "\n".join(
                f"- {r.get('step', '?')}: "
                f"{r.get('result', '?')[:200]}"
                for r in results
            )
            + "\n\nSynthesize these results into a "
              "coherent response to the original goal."
        )
        from .foundation import Message, GenerationConfig
        final = await self.foundation.generate(
            [Message(role="user",
                     content=synthesis_prompt)],
            GenerationConfig(max_tokens=4096),
        )
        return final.content

    async def _execute_step(self, step) -> dict:
        """Execute a single plan step using
        appropriate tools."""
        from .foundation import Message
        from .action_loop import execute_with_tools
        try:
            result = await execute_with_tools(
                self.foundation, self.tools,
                [Message(
                    role="user",
                    content=f"Execute this step: "
                            f"{step.description}",
                )],
                max_iterations=5,
            )
            return {"step": step.description,
                    "result": result, "success": True}
        except Exception as e:
            return {"step": step.description,
                    "error": str(e), "success": False}

    def _is_complex_task(self, task: str) -> bool:
        """Heuristic: does this task need planning?"""
        complexity_signals = [
            "and then", "after that", "step by step",
            "research", "build", "create", "implement",
            "compare and", "analyze and",
        ]
        task_lower = task.lower()
        signal_count = sum(
            1 for s in complexity_signals
            if s in task_lower
        )
        return signal_count >= 2 or len(task) > 500

    def _classify_task(self, task: str) -> str:
        """Quick task type classification."""
        task_lower = task.lower()
        if any(w in task_lower
               for w in ["code", "implement",
                          "fix", "debug"]):
            return "code"
        if any(w in task_lower
               for w in ["math", "calculate", "prove"]):
            return "math"
        if any(w in task_lower
               for w in ["plan", "strategy", "design"]):
            return "planning"
        if any(w in task_lower
               for w in ["research", "find", "search"]):
            return "research"
        if any(w in task_lower
               for w in ["write", "draft", "compose"]):
            return "writing"
        return "general"

    def _system_prompt(self, context: str) -> str:
        return (
            "You are an AGI system with access to "
            "memory, reasoning, planning, and tool "
            "use capabilities.\n\n"
            f"{context}\n\n"
            "Respond to the user's request accurately "
            "and completely. If you need to use tools, "
            "reason step-by-step, or plan multi-step "
            "actions, do so. If you're uncertain, say "
            "so and explain what you'd need to be "
            "more confident."
        )
