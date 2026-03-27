# multi_agent.py

from dataclasses import dataclass, field
from typing import Optional, Callable
import asyncio

from .foundation import (
    Message, GenerationConfig, ModelTier,
)


@dataclass
class AgentSpec:
    name: str
    role: str  # System prompt defining expertise
    model_tier: str = "fast"  # "fast" or "reasoning"
    tools: list[str] = field(default_factory=list)


@dataclass
class AgentMessage:
    from_agent: str
    to_agent: str
    content: str
    message_type: str = "task"
    # "task", "result", "question", "review"


class MultiAgentOrchestrator:
    """Manages multiple specialized agents working
    on a shared goal."""

    def __init__(self, foundation: 'FoundationModel',
                 tools: 'ToolManager'):
        self.foundation = foundation
        self.tools = tools
        self.agents: dict[str, AgentSpec] = {}
        self.message_log: list[AgentMessage] = []

    def register_agent(self, spec: AgentSpec):
        self.agents[spec.name] = spec

    async def delegate(
        self, task: str, agent_name: str,
    ) -> str:
        """Send a task to a specific agent
        and get the result."""
        agent = self.agents[agent_name]

        messages = [
            Message(role="system", content=agent.role),
            Message(role="user", content=task),
        ]

        result = await self.foundation.generate(
            messages,
            GenerationConfig(
                force_reasoning=(
                    agent.model_tier == "reasoning"
                ),
                tools=[
                    t for t
                    in self.tools.get_tool_definitions()
                    if t["function"]["name"] in agent.tools
                ] if agent.tools else [],
            ),
        )

        self.message_log.append(AgentMessage(
            from_agent="orchestrator",
            to_agent=agent_name,
            content=task,
        ))
        self.message_log.append(AgentMessage(
            from_agent=agent_name,
            to_agent="orchestrator",
            content=result.content,
            message_type="result",
        ))

        return result.content

    async def parallel_delegate(
        self, tasks: dict[str, str],
    ) -> dict[str, str]:
        """Send tasks to multiple agents in parallel.

        Args:
            tasks: {agent_name: task_description}
        Returns:
            {agent_name: result}
        """
        async def run_one(
            name: str, task: str,
        ) -> tuple[str, str]:
            result = await self.delegate(task, name)
            return name, result

        results = await asyncio.gather(
            *[run_one(name, task)
              for name, task in tasks.items()]
        )
        return dict(results)

    async def debate(
        self, question: str,
        agents: list[str],
        rounds: int = 2,
    ) -> str:
        """Have agents debate a question and
        converge on an answer."""
        positions = {}

        # Round 1: Independent positions
        for agent_name in agents:
            positions[agent_name] = await self.delegate(
                f"Answer this question independently:\n"
                f"{question}",
                agent_name,
            )

        # Subsequent rounds: respond to each other
        for round_num in range(1, rounds):
            context = "\n\n".join(
                f"**{name}** says: {pos}"
                for name, pos in positions.items()
            )
            for agent_name in agents:
                positions[agent_name] = \
                    await self.delegate(
                        f"Other agents have shared their "
                        f"views:\n{context}\n\n"
                        f"Revise your answer if needed. "
                        f"The question is:\n{question}",
                        agent_name,
                    )

        # Synthesize
        final_context = "\n\n".join(
            f"**{name}**: {pos}"
            for name, pos in positions.items()
        )
        synthesis = await self.delegate(
            f"Synthesize these expert opinions into "
            f"a final answer:\n{final_context}",
            agents[0],
        )
        return synthesis


# Pre-built agent specs

RESEARCHER = AgentSpec(
    name="researcher",
    role="You are a research specialist. You find, "
         "analyze, and synthesize information from "
         "multiple sources. Be thorough and cite "
         "your sources.",
    tools=["web_search", "read_file"],
)

CODER = AgentSpec(
    name="coder",
    role="You are a senior software engineer. Write "
         "clean, tested, production-quality code. "
         "Follow best practices.",
    model_tier="reasoning",
    tools=["execute_python", "read_file", "write_file"],
)

REVIEWER = AgentSpec(
    name="reviewer",
    role="You are a critical code/design reviewer. "
         "Find bugs, security issues, and suggest "
         "improvements. Be specific and constructive.",
    model_tier="reasoning",
    tools=["read_file"],
)

WRITER = AgentSpec(
    name="writer",
    role="You are a technical writer. Create clear, "
         "well-structured documentation that engineers "
         "will actually read.",
    tools=["read_file", "write_file"],
)


async def build_feature(
    orchestrator: MultiAgentOrchestrator,
    spec: str,
):
    """Multi-agent workflow:
    research -> code -> review -> document."""

    # Step 1: Research (parallel with planning)
    research, plan = await asyncio.gather(
        orchestrator.delegate(
            f"Research best practices for: {spec}",
            "researcher",
        ),
        orchestrator.delegate(
            f"Create an implementation plan for: {spec}",
            "coder",
        ),
    )

    # Step 2: Code (informed by research)
    code = await orchestrator.delegate(
        f"Implement this feature:\n{spec}\n\n"
        f"Research findings:\n{research}\n\n"
        f"Plan:\n{plan}",
        "coder",
    )

    # Step 3: Review
    review = await orchestrator.delegate(
        f"Review this code:\n{code}\n\n"
        f"Original spec:\n{spec}",
        "reviewer",
    )

    # Step 4: Fix issues and document (parallel)
    fixed_code, docs = await asyncio.gather(
        orchestrator.delegate(
            f"Fix these review issues:\n{review}\n\n"
            f"Original code:\n{code}",
            "coder",
        ),
        orchestrator.delegate(
            f"Write documentation for this feature:\n"
            f"{spec}\n\nCode:\n{code}",
            "writer",
        ),
    )

    return {
        "code": fixed_code,
        "docs": docs,
        "review": review,
    }
