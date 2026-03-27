# tool_manager.py

from dataclasses import dataclass, field
from typing import Callable, Any, Optional
import json

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema for parameters
    handler: Callable  # Async function to execute
    category: str = "general"
    risk_level: str = "low"  # "low", "medium", "high"
    requires_approval: bool = False

@dataclass
class ToolCall:
    tool_name: str
    arguments: dict
    call_id: str = ""

@dataclass
class ToolResult:
    call_id: str
    output: str
    success: bool
    error: Optional[str] = None
    side_effects: list[str] = field(default_factory=list)

class ToolManager:
    """Manages available tools and executes tool calls safely."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._execution_log: list[dict] = []

    def register(self, tool: Tool):
        """Register a new tool."""
        self._tools[tool.name] = tool

    def get_tool_definitions(self) -> list[dict]:
        """Get tool definitions in OpenAI/MCP format for LLM context."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    async def execute(self, call: ToolCall) -> ToolResult:
        """Execute a tool call with safety checks."""
        tool = self._tools.get(call.tool_name)
        if not tool:
            return ToolResult(
                call_id=call.call_id,
                output="",
                success=False,
                error=f"Unknown tool: {call.tool_name}",
            )

        # Safety check for high-risk tools
        if tool.requires_approval:
            # In production, this would prompt the user
            pass

        try:
            result = await tool.handler(**call.arguments)
            self._execution_log.append({
                "tool": call.tool_name,
                "args": call.arguments,
                "success": True,
            })
            return ToolResult(
                call_id=call.call_id,
                output=str(result),
                success=True,
            )
        except Exception as e:
            self._execution_log.append({
                "tool": call.tool_name,
                "args": call.arguments,
                "success": False,
                "error": str(e),
            })
            return ToolResult(
                call_id=call.call_id,
                output="",
                success=False,
                error=str(e),
            )

    def get_execution_history(self, last_n: int = 10) -> list[dict]:
        return self._execution_log[-last_n:]
