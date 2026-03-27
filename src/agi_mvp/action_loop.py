# action_loop.py

import json
from .foundation import Message, GenerationConfig
from .tool_manager import ToolManager, ToolCall


async def execute_with_tools(
    foundation: 'FoundationModel',
    tool_manager: ToolManager,
    messages: list[Message],
    max_iterations: int = 10,
) -> str:
    """Run the foundation model with tool use until
    task completion."""
    for i in range(max_iterations):
        result = await foundation.generate(
            messages,
            GenerationConfig(
                tools=tool_manager.get_tool_definitions()
            ),
        )

        # No tool calls -> model is done
        if not result.tool_calls:
            return result.content

        # Execute tool calls
        tool_results = []
        for tc in result.tool_calls:
            call = ToolCall(
                tool_name=tc.function.name,
                arguments=json.loads(tc.function.arguments),
                call_id=tc.id,
            )
            tr = await tool_manager.execute(call)
            tool_results.append(tr)

        # Append results and continue
        messages.append(Message(
            role="assistant", content=result.content,
            tool_calls=result.tool_calls,
        ))
        for tr in tool_results:
            messages.append(Message(
                role="tool", content=tr.output,
                tool_call_id=tr.call_id,
            ))

    return "Max tool iterations reached"
