"""Abstract base provider for LLM inference."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import ChoiceLogprobs


class BaseProvider(ABC):
    """Abstract base class for LLM inference providers.

    Subclasses must implement:
        - complete(): Send a chat completion request
        - provider_name: Human-readable provider identifier

    Configurable per-provider options (set in constructor):
        - label: Human-readable tag for this configuration variant
            (e.g. "temp=0.7").  When set it appears in reports and logs
            alongside the model name.
        - temperature: Sampling temperature (default 0.0 for deterministic output)
        - max_tokens: Maximum tokens in the response
        - enforce_json: Whether to enforce structured JSON output.
            Each provider chooses the best strategy for its model:
            json_schema, json_object, or prompt-level instruction.
        - retry_times: Maximum retries per sample when API requests fail.
            After retry_times+1 total failures for a sample, it is skipped
            for the rest of the session (default 1, i.e. 2 total attempts).
        - max_errors: Maximum total API errors before aborting the model
            for the current session. Counted per failed batch attempt
            (default 3).
        - logprobs: Whether to request token logprobs from the API.
            Providers that don't support it silently return None.
        - top_logprobs: Number of top logprobs per token (only when
            logprobs=True).  Providers may cap or ignore this value.
    """

    model: str
    label: str | None
    batch_size: int
    temperature: float
    max_tokens: int | None
    enforce_json: bool
    retry_times: int
    max_errors: int
    logprobs: bool
    top_logprobs: int | None

    @property
    def display_name(self) -> str:
        """Name shown in reports — model plus optional label."""
        if self.label:
            return f"{self.model} ({self.label})"
        return self.model

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        response_format: dict | None = None,
    ) -> tuple[str, int, int, ChoiceLogprobs | None]:
        """Send a chat completion request.

        Args:
            messages: List of chat messages (role + content dicts).
            response_format: Optional structured output format spec.

        Returns:
            Tuple of (content_text, prompt_tokens, completion_tokens, logprobs).
            logprobs is None when the provider does not support it or
            logprobs was not requested.

        Raises:
            openai.APIError: On API communication failure.
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider identifier for reports."""
        ...

    @property
    def model_slug(self) -> str:
        """URL-safe model identifier for file naming."""
        return self.model.replace("/", "_").replace(":", "_").replace(".", "-")

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(model={self.model!r}, batch={self.batch_size}, "
            f"temperature={self.temperature})"
            + (f", label={self.label!r}" if self.label else "")
        )
