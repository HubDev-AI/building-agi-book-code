# main.py -- Bootstrap the AGI MVP

import asyncio
from .foundation import FoundationModel, ModelConfig, \
    ModelRouter
from .working_memory import WorkingMemory
from .episodic_memory import EpisodicMemory
from .semantic_memory import SemanticMemory
from .memory_manager import MemoryManager
from .reasoning_engine import ReasoningEngine
from .planner import Planner
from .tool_manager import ToolManager
from .world_model import WorldModel
from .metacognition import MetaCognition
from .cognitive_core import CognitiveCore

async def main():
    # Layer 1: Foundation Model
    config = ModelConfig()
    router = ModelRouter()
    foundation = FoundationModel(config, router)

    # Layer 2: Memory
    working = WorkingMemory()
    episodic = EpisodicMemory()
    semantic = SemanticMemory()
    memory = MemoryManager(
        working, episodic, semantic, foundation,
    )

    # Layer 3: Reasoning
    reasoning = ReasoningEngine(foundation, memory)

    # Layer 4: Planner
    planner = Planner(foundation, memory)

    # Layer 5: Tools
    tools = ToolManager()
    from tools.code_execution import code_execution_tool
    from tools.web_search import web_search_tool
    tools.register(code_execution_tool)
    tools.register(web_search_tool)

    # Layer 6: World Model
    world_model = WorldModel(foundation)

    # Layer 7: Meta-Cognition
    meta = MetaCognition(foundation)

    # Assemble the Cognitive Core
    core = CognitiveCore(
        foundation=foundation,
        memory=memory,
        reasoning=reasoning,
        planner=planner,
        tools=tools,
        world_model=world_model,
        metacognition=meta,
    )

    # Interactive loop
    print("AGI MVP ready. Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        response = await core.process(user_input)
        print(f"\nAGI: {response}\n")

if __name__ == "__main__":
    asyncio.run(main())
