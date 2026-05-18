"""OpenRouter provider: cloud LLM inference via OpenAI-compatible API.

OpenRouter provides a unified API to access hundreds of AI models.
Base URL: https://openrouter.ai/api/v1
Auth: Bearer token via API key.
Structured output: json_schema (model-dependent).
"""

from __future__ import annotations

import json

from openai import AsyncOpenAI

from .base import BaseProvider


class OpenRouter(BaseProvider):
    """OpenRouter inference provider.

    Args:
        model: Model identifier as known by OpenRouter
               (e.g. "nvidia/nemotron_3_super").
        api_key: OpenRouter API key.
        batch: Number of questions per prompt/request.
        temperature: Sampling temperature. 0.0 = deterministic.
        max_tokens: Maximum tokens in the response (None = provider default).
        enforce_json: Whether to enforce structured JSON output (default True).
            True → json_schema (model-dependent support).
            False → no format enforcement, rely on prompt.
        site_url: Optional HTTP-Referer header for OpenRouter rankings.
        site_name: Optional X-Title header for OpenRouter rankings.
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
        site_url: str | None = None,
        site_name: str | None = None,
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

        extra_headers: dict[str, str] = {}
        if site_url:
            extra_headers["HTTP-Referer"] = site_url
        if site_name:
            extra_headers["X-Title"] = site_name

        self._client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers=extra_headers if extra_headers else None,
            max_retries=0,
            timeout=120.0,
        )

    @property
    def provider_name(self) -> str:
        return "openrouter"

    async def complete(
        self,
        messages: list[dict[str, str]],
        response_format: dict | None = None,
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
