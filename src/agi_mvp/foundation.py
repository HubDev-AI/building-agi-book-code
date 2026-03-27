# foundation.py -- Unified foundation model interface

from dataclasses import dataclass, field
from typing import AsyncIterator, Optional
from enum import Enum
import json
import re


# --- Model Configuration (config.py) ---

@dataclass
class ModelConfig:
    fast_model: str = "deepseek-chat"           # DeepSeek V3
    reasoning_model: str = "deepseek-reasoner"   # DeepSeek R1
    fast_base_url: str = "https://api.deepseek.com"
    reasoning_base_url: str = "https://api.deepseek.com"
    fast_max_tokens: int = 4096
    reasoning_max_tokens: int = 16384


# --- Model Router (router.py) ---

class ModelTier(Enum):
    FAST = "fast"
    REASONING = "reasoning"


class ModelRouter:
    """Routes requests to the appropriate model based on task complexity."""

    # Patterns that indicate reasoning is needed
    REASONING_PATTERNS = [
        r"prove|proof|theorem",
        r"step.by.step|think.carefully|reason.through",
        r"plan|strategy|decompose",
        r"why.does|explain.why|root.cause",
        r"math|calcul|equation|solve",
        r"verify|check.if|validate",
        r"compar[ei]|tradeoff|analyze",
        r"code.review|debug|fix.this.bug",
        r"multi.step|complex",
    ]

    # Patterns that indicate fast model suffices
    FAST_PATTERNS = [
        r"classif|categorize|label",
        r"extract|parse|convert",
        r"summar|tldr|brief",
        r"translate|rephrase",
        r"yes.or.no|true.or.false",
        r"format|rewrite|clean.up",
    ]

    def __init__(self):
        self._reasoning_re = re.compile(
            "|".join(self.REASONING_PATTERNS), re.IGNORECASE
        )
        self._fast_re = re.compile(
            "|".join(self.FAST_PATTERNS), re.IGNORECASE
        )

    def route(
        self,
        prompt: str,
        force_tier: Optional[ModelTier] = None,
        context_length: int = 0,
    ) -> ModelTier:
        """Determine which model tier should handle this request."""
        if force_tier:
            return force_tier

        # Long context + reasoning patterns -> reasoning model
        if context_length > 8000 and self._reasoning_re.search(prompt):
            return ModelTier.REASONING

        # Explicit reasoning indicators
        if self._reasoning_re.search(prompt):
            return ModelTier.REASONING

        # Explicit fast indicators
        if self._fast_re.search(prompt):
            return ModelTier.FAST

        # Default to fast -- only escalate when needed
        return ModelTier.FAST


# --- Core Data Classes (foundation.py) ---

@dataclass
class Message:
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_calls: list = field(default_factory=list)
    tool_call_id: Optional[str] = None


@dataclass
class GenerationConfig:
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.95
    stop: list = field(default_factory=list)
    tools: list = field(default_factory=list)  # MCP tool definitions
    force_reasoning: bool = False  # Override router


@dataclass
class GenerationResult:
    content: str
    model_used: str
    tier: ModelTier
    tokens_in: int
    tokens_out: int
    latency_ms: float
    tool_calls: list = field(default_factory=list)
    reasoning_trace: Optional[str] = None  # For reasoning models


# --- Foundation Model (foundation.py) ---

class FoundationModel:
    """Unified interface to the dual-model foundation layer."""

    def __init__(self, config: ModelConfig, router: ModelRouter):
        self.config = config
        self.router = router
        self._fast_client = self._init_client(config.fast_base_url)
        self._reasoning_client = self._init_client(
            config.reasoning_base_url
        )

    async def generate(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """Generate a response using the appropriate model."""
        if config is None:
            config = GenerationConfig()

        # Route to the right model
        prompt_text = messages[-1].content if messages else ""
        context_len = sum(len(m.content) for m in messages)

        tier = self.router.route(
            prompt_text,
            force_tier=(
                ModelTier.REASONING if config.force_reasoning else None
            ),
            context_length=context_len,
        )

        client = (
            self._reasoning_client
            if tier == ModelTier.REASONING
            else self._fast_client
        )
        model_name = (
            self.config.reasoning_model
            if tier == ModelTier.REASONING
            else self.config.fast_model
        )

        # Make the API call
        import time
        start = time.monotonic()

        # Retry on transient errors (429, 529)
        import asyncio as _aio
        for _attempt in range(5):
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": m.role, "content": m.content}
                        for m in messages
                    ],
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    tools=config.tools or None,
                )
                break
            except Exception as _e:
                if "529" in str(_e) or "429" in str(_e):
                    await _aio.sleep(2 ** _attempt)
                else:
                    raise
        else:
            raise RuntimeError(
                "API unavailable after retries"
            )

        elapsed = (time.monotonic() - start) * 1000
        choice = response.choices[0]

        return GenerationResult(
            content=choice.message.content or "",
            model_used=model_name,
            tier=tier,
            tokens_in=response.usage.prompt_tokens,
            tokens_out=response.usage.completion_tokens,
            latency_ms=elapsed,
            tool_calls=choice.message.tool_calls or [],
            reasoning_trace=getattr(
                choice.message, 'reasoning_content', None
            ),
        )

    async def generate_stream(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[str]:
        """Stream a response token by token."""
        if config is None:
            config = GenerationConfig()
        prompt_text = messages[-1].content if messages else ""
        tier = self.router.route(
            prompt_text,
            force_tier=(
                ModelTier.REASONING
                if config.force_reasoning else None
            ),
        )
        client = (
            self._reasoning_client
            if tier == ModelTier.REASONING
            else self._fast_client
        )
        model_name = (
            self.config.reasoning_model
            if tier == ModelTier.REASONING
            else self.config.fast_model
        )
        stream = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": m.role, "content": m.content}
                for m in messages
            ],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def _init_client(self, base_url: str):
        from openai import AsyncOpenAI
        return AsyncOpenAI(base_url=base_url)
