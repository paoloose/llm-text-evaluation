"""OpenCode Go provider: subscription-based cloud inference.

OpenCode Go provides access to curated open coding models via two API formats:
- OpenAI-compatible /v1/chat/completions (most models)
- Anthropic-compatible /v1/messages (MiniMax M2.5, M2.7)

Base URL: https://opencode.ai/zen/go/v1
Auth: Bearer token via API key.
"""

from __future__ import annotations

import json
import logging

import httpx
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
        response_format_mode: Structured output mode.
            "json_schema" (default): JSON schema enforcement (OpenAI only).
            "json_object": Basic JSON mode.
            "none": No format enforcement, rely on prompt.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        batch: int = 1,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format_mode: str = "json_schema",
    ) -> None:
        self.model = model
        self.batch_size = batch
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.response_format_mode = response_format_mode

        self._is_anthropic = model.lower() in ANTHROPIC_MODELS

        auth_headers = {
            "Authorization": f"Bearer {api_key}",
        }

        if self._is_anthropic:
            self._http = httpx.AsyncClient(
                base_url="https://opencode.ai/zen/go/v1",
                headers={
                    **auth_headers,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                },
                timeout=httpx.Timeout(120.0),
            )
            self._client = None
        else:
            self._client = AsyncOpenAI(
                base_url="https://opencode.ai/zen/go/v1",
                api_key=api_key,
                max_retries=2,
                timeout=120.0,
            )
            self._http = None

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

        if response_format and self.response_format_mode == "json_schema":
            kwargs["response_format"] = response_format
        elif self.response_format_mode == "json_object":
            kwargs["response_format"] = {"type": "json_object"}

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

        if response_format and self.response_format_mode == "json_schema":
            schema_instruction = (
                "\n\nIMPORTANT: You must respond with valid JSON matching this schema:\n"
                + json.dumps(response_format, ensure_ascii=False)
            )
            if system:
                system += schema_instruction
            else:
                system = "Respond with valid JSON.\n" + schema_instruction

        body: dict = {
            "model": self.model,
            "messages": api_messages,
            "temperature": self.temperature,
        }

        if system:
            body["system"] = system

        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens

        response = await self._http.post("/messages", json=body)
        response.raise_for_status()
        data = response.json()

        content = ""
        for block in data.get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                content += block.get("text", "")

        usage = data.get("usage", {})
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)

        return content, prompt_tokens, completion_tokens

    async def close(self) -> None:
        if self._http:
            await self._http.aclose()
        if self._client:
            await self._client.close()
