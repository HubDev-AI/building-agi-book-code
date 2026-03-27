# tools/file_operations.py

import os
from pathlib import Path

WORKSPACE = Path("./workspace")  # Sandboxed directory

async def read_file(path: str) -> str:
    """Read a file from the workspace."""
    full_path = WORKSPACE / path
    if not full_path.resolve().is_relative_to(
        WORKSPACE.resolve()
    ):
        return "ERROR: Access denied -- path outside workspace"
    return full_path.read_text()

async def write_file(path: str, content: str) -> str:
    """Write content to a file in the workspace."""
    full_path = WORKSPACE / path
    if not full_path.resolve().is_relative_to(
        WORKSPACE.resolve()
    ):
        return "ERROR: Access denied -- path outside workspace"
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content)
    return f"Written {len(content)} bytes to {path}"

async def list_files(directory: str = ".") -> str:
    """List files in a workspace directory."""
    full_path = WORKSPACE / directory
    if not full_path.resolve().is_relative_to(
        WORKSPACE.resolve()
    ):
        return "ERROR: Access denied"
    entries = sorted(full_path.iterdir())
    return "\n".join(
        f"{'[DIR]' if e.is_dir() else '[FILE]'} {e.name}"
        for e in entries
    )
