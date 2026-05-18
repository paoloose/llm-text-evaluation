"""OpenCode Go provider: subscription-based cloud inference.

OpenCode Go provides access to curated open coding models via two API formats:
- OpenAI-compatible /v1/chat/completions (most models)
- Anthropic Messages /v1/messages (MiniMax M2.5, M2.7)

Base URL: https://opencode.ai/zen/go/v1
Auth: Bearer token via API key.
"""

from __future__ import annotations

import json
import logging

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from .base import BaseProvider

logger = logging.getLogger("llm_verbal_framework")


ANTHROPIC_MODELS = frozenset({"minimax-m2.7", "minimax-m2.5"})


class OpencodeGo(BaseProvider):
    """OpenCode Go inference provider.

    Handles two API formats transparently:
    - OpenAI-compatible for most models (GLM, Kimi, DeepSeek, MiMo, Qwen, Hy3).
    - Anthropic Messages for MiniMax models.

    Args:
        model: Model identifier (e.g. "kimi-k2.6", "minimax-m2.7").
        api_key: OpenCode Go API key from https://opencode.ai/auth.
        batch: Number of questions per prompt/request.
        temperature: Sampling temperature. 0.0 = deterministic.
        max_tokens: Maximum tokens in the response (None = provider default).
        enforce_json: Whether to enforce structured JSON output (default True).
            OpenAI models → json_schema.
            Anthropic models → prompt-level schema instruction.
        retry_times: Max retries per sample on API error (default 1).
        max_errors: Max total API errors before aborting the model (default 3).
        label: Optional tag for this configuration variant (e.g. "temp=0.7").
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        batch: int = 1,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        enforce_json: bool = True,
        retry_times: int = 1,
        max_errors: int = 3,
        label: str | None = None,
    ) -> None:
        self.model = model
        self.label = label
        self.batch_size = batch
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enforce_json = enforce_json
        self.retry_times = retry_times
        self.max_errors = max_errors

        self._is_anthropic = model.lower() in ANTHROPIC_MODELS

        if self._is_anthropic:
            self._client = AsyncAnthropic(
                base_url="https://opencode.ai/zen/go",
                api_key=api_key,
                max_retries=0,
                timeout=120.0,
            )
        else:
            self._client = AsyncOpenAI(
                base_url="https://opencode.ai/zen/go/v1",
                api_key=api_key,
                max_retries=0,
                timeout=120.0,
            )

    @property
    def provider_name(self) -> str:
        return "opencode-go"

    async def complete(
        self,
        messages: list[dict[str, str]],
        response_format: dict | None = None,
    ) -> tuple[str, int, int]:
        if self._is_anthropic:
            return await self._complete_anthropic(messages, response_format)
        return await self._complete_openai(messages, response_format)

    async def _complete_openai(
        self,
        messages: list[dict[str, str]],
        response_format: dict | None,
    ) -> tuple[str, int, int]:
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        if response_format:
            if self.enforce_json:
                kwargs["response_format"] = response_format
            else:
                for msg in messages:
                    if msg["role"] == "system":
                        msg["content"] += "\n\nExpected response schema:\n" + json.dumps(response_format, ensure_ascii=False)
                        break

        response = await self._client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content or ""
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        return content, prompt_tokens, completion_tokens

    async def _complete_anthropic(
        self,
        messages: list[dict[str, str]],
        response_format: dict | None,
    ) -> tuple[str, int, int]:
        system = None
        api_messages: list[dict] = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                system = content
            else:
                api_messages.append({"role": role, "content": content})

        if response_format and self.enforce_json:
            schema_instruction = (
                "\n\nIMPORTANT: You must respond with valid JSON matching this schema:\n"
                + json.dumps(response_format, ensure_ascii=False)
            )
            if system:
                system += schema_instruction
            else:
                system = "Respond with valid JSON.\n" + schema_instruction

        kwargs: dict = {
            "model": self.model,
            "messages": api_messages,
            "temperature": self.temperature,
        }

        if system:
            kwargs["system"] = system

        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        response = await self._client.messages.create(**kwargs)

        content = ""
        if response.content:
            for block in response.content:
                if hasattr(block, "text"):
                    content = block.text
                    break

        prompt_tokens = getattr(response.usage, "input_tokens", 0) if response.usage else 0
        completion_tokens = getattr(response.usage, "output_tokens", 0) if response.usage else 0

        return content, prompt_tokens, completion_tokens
