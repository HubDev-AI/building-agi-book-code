# tools/code_execution.py

import asyncio
import tempfile
import os

from ..tool_manager import Tool

async def execute_python(code: str, timeout: int = 30) -> str:
    """Execute Python code in a sandboxed subprocess."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                     delete=False) as f:
        f.write(code)
        f.flush()

        try:
            proc = await asyncio.create_subprocess_exec(
                'python', f.name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
            output = stdout.decode()
            if stderr:
                output += f"\nSTDERR: {stderr.decode()}"
            return output
        except asyncio.TimeoutError:
            proc.kill()
            return "ERROR: Execution timed out"
        finally:
            os.unlink(f.name)

# Register as a tool
code_execution_tool = Tool(
    name="execute_python",
    description="Execute Python code and return the output. "
                "Use for calculations, data processing, or "
                "testing hypotheses.",
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute",
            },
            "timeout": {
                "type": "integer",
                "description": "Max execution time in seconds",
                "default": 30,
            },
        },
        "required": ["code"],
    },
    handler=execute_python,
    category="computation",
    risk_level="medium",
)
