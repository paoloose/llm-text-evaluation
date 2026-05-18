"""Ollama provider: local or remote Ollama server via OpenAI-compatible API.

Ollama exposes an OpenAI-compatible /v1/chat/completions endpoint.
Structured output uses llama.cpp grammar masking for guaranteed valid JSON.

Base URL pattern: http://{host}:{port}/v1
Default: http://localhost:11434/v1
Auth: Not required locally. For remote servers behind a reverse proxy,
      pass the proxy token via api_key.
"""

from __future__ import annotations

import json

from openai import AsyncOpenAI

from .base import BaseProvider


class Ollama(BaseProvider):
    """Ollama inference provider.

    Args:
        model: Model name as known by Ollama (e.g. "qwen2.5:7b-instruct").
        batch: Number of questions per prompt/request.
        url: Host and port of the Ollama server (default: "localhost:11434").
             Supports formats: "host:port", "http://host:port", "https://host:port".
        api_key: API key for remote servers behind auth proxies.
                 Ignored by local Ollama instances (any non-empty string accepted).
        temperature: Sampling temperature. 0.0 = deterministic.
        max_tokens: Maximum tokens in the response (None = provider default).
        enforce_json: Whether to enforce structured JSON output (default True).
            True → json_schema (grammar-enforced).
            False → no format enforcement, rely on prompt.
        retry_times: Max retries per sample on API error (default 1).
        max_errors: Max total API errors before aborting the model (default 3).
        label: Optional tag for this configuration variant (e.g. "temp=0.7").
    """

    def __init__(
        self,
        model: str,
        batch: int = 1,
        url: str = "localhost:11434",
        api_key: str = "ollama",
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

        base = url if url.startswith("http") else f"http://{url}"
        base = base.rstrip("/")

        self._client = AsyncOpenAI(
            base_url=f"{base}/v1",
            api_key=api_key,
            max_retries=0,
            timeout=600.0,
        )

    @property
    def provider_name(self) -> str:
        return "ollama"

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
