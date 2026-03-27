# safety/sandbox.py

from dataclasses import dataclass, field
from typing import Optional
import subprocess
import tempfile
import os
import resource
import signal


@dataclass
class SandboxConfig:
    """Configuration for the code execution sandbox."""
    max_memory_mb: int = 512
    max_cpu_seconds: int = 30
    max_file_size_mb: int = 10
    allow_network: bool = False
    allow_filesystem_write: bool = False
    allowed_imports: list[str] = field(
        default_factory=lambda: [
            "math", "json", "re", "datetime",
            "collections", "itertools", "functools",
            "statistics", "typing", "dataclasses",
            "enum", "hashlib", "csv", "textwrap",
        ]
    )
    blocked_imports: list[str] = field(
        default_factory=lambda: [
            "os", "sys", "subprocess", "shutil",
            "socket", "http", "urllib", "requests",
            "pathlib", "importlib", "ctypes", "signal",
            "threading", "multiprocessing", "pickle",
            "shelve",
        ]
    )


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False
    memory_exceeded: bool = False
    blocked_imports: list[str] = field(
        default_factory=list
    )
    error: Optional[str] = None


class CodeSandbox:
    """Executes code in a restricted environment.

    For production use, replace the subprocess approach
    with a container-based sandbox (E2B, Modal, or Docker
    with seccomp profiles). The subprocess approach here
    demonstrates the principle and works for development.
    """

    def __init__(
        self, config: Optional[SandboxConfig] = None
    ):
        self.config = config or SandboxConfig()

    def execute(
        self, code: str, language: str = "python"
    ) -> SandboxResult:
        """Execute code in a sandboxed environment."""
        # Pre-execution safety: scan for blocked imports
        blocked = self._scan_imports(code)
        if blocked:
            return SandboxResult(
                stdout="",
                stderr=(
                    f"Blocked imports detected: "
                    f"{', '.join(blocked)}"
                ),
                exit_code=1,
                blocked_imports=blocked,
                error="Import policy violation",
            )

        wrapped = self._wrap_code(code)

        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(
                tmpdir, "sandbox_script.py"
            )
            with open(script_path, "w") as f:
                f.write(wrapped)

            try:
                result = subprocess.run(
                    ["python3", script_path],
                    capture_output=True,
                    text=True,
                    timeout=self.config.max_cpu_seconds,
                    cwd=tmpdir,
                    env=self._sandbox_env(),
                )
                return SandboxResult(
                    stdout=result.stdout[:10000],
                    stderr=result.stderr[:5000],
                    exit_code=result.returncode,
                )

            except subprocess.TimeoutExpired:
                return SandboxResult(
                    stdout="",
                    stderr="Execution timed out",
                    exit_code=-1,
                    timed_out=True,
                    error=(
                        f"Exceeded "
                        f"{self.config.max_cpu_seconds}s "
                        f"limit"
                    ),
                )

    def _scan_imports(self, code: str) -> list[str]:
        """Scan code for blocked imports via AST."""
        import ast
        blocked = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root = alias.name.split(".")[0]
                        if root in (
                            self.config.blocked_imports
                        ):
                            blocked.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        root = node.module.split(".")[0]
                        if root in (
                            self.config.blocked_imports
                        ):
                            blocked.append(node.module)
        except SyntaxError:
            blocked.append("__syntax_error__")
        return blocked

    def _wrap_code(self, code: str) -> str:
        """Wrap user code with resource limits."""
        mem = self.config.max_memory_mb * 1024 * 1024
        cpu = self.config.max_cpu_seconds
        fsize = self.config.max_file_size_mb * 1024 * 1024
        return f"""
import resource
import sys

# Memory limit
resource.setrlimit(resource.RLIMIT_AS, ({mem}, {mem}))
# CPU time limit
resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu}))
# File size limit
resource.setrlimit(
    resource.RLIMIT_FSIZE, ({fsize}, {fsize})
)

# Execute the sandboxed code
{code}
"""

    def _sandbox_env(self) -> dict:
        """Minimal environment variables."""
        return {
            "PATH": "/usr/bin:/bin",
            "HOME": "/tmp",
            "LANG": "en_US.UTF-8",
            # No cloud credentials, API keys, or tokens
        }
