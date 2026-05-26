"""Provider pool: round-robin across multiple providers with failover."""

from __future__ import annotations

import itertools
import logging
import threading
from typing import TYPE_CHECKING

from .base import BaseProvider

if TYPE_CHECKING:
    from ..types import ChoiceLogprobs

logger = logging.getLogger("llm_verbal_framework")


class ProviderPool(BaseProvider):
    """Pool of providers that serve the same model.

    Distributes ``complete()`` calls across inner providers using
    round-robin.  If a provider fails, the call is retried on the
    next provider in the pool (failover).  All providers must use
    the same ``batch_size`` to avoid sample misalignment.

    Args:
        providers: Two or more :class:`BaseProvider` instances.
        name: Canonical model name used for display, file naming,
            and report generation.  Overrides each inner provider's
            ``model`` attribute so the pool appears as a single model.
        label: Optional label (same semantics as :class:`BaseProvider`).
    """

    def __init__(
        self,
        providers: list[BaseProvider],
        name: str,
        label: str | None = None,
    ) -> None:
        if len(providers) < 2:
            raise ValueError("ProviderPool requires at least 2 providers")
        batches = {p.batch_size for p in providers}
        if len(batches) > 1:
            raise ValueError(
                f"All providers in a pool must use the same batch_size, "
                f"got: {', '.join(str(b) for b in sorted(batches))}"
            )

        self._providers = list(providers)
        self._rr_counter = itertools.count()

        self.model = name
        self.label = label
        self.batch_size = providers[0].batch_size
        self.temperature = providers[0].temperature
        self.max_tokens = providers[0].max_tokens
        self.enforce_json = providers[0].enforce_json
        self.retry_times = providers[0].retry_times
        self.max_errors = providers[0].max_errors
        self.logprobs = providers[0].logprobs
        self.top_logprobs = providers[0].top_logprobs

    @property
    def provider_name(self) -> str:
        return "+".join(p.provider_name for p in self._providers)

    @property
    def display_name(self) -> str:
        base = f"{self.model} (pool:{len(self._providers)})"
        if self.label:
            return f"{base} ({self.label})"
        return base

    def _pick_provider(self) -> BaseProvider:
        idx = next(self._rr_counter) % len(self._providers)
        return self._providers[idx]

    async def complete(
        self,
        messages: list[dict[str, str]],
        response_format: dict | None = None,
    ) -> tuple[str, int, int, ChoiceLogprobs | None]:
        """Route the call to one provider, failover on error.

        Tries each provider exactly once.  The first provider is
        chosen by round-robin; on failure, the remaining providers
        are tried in order.  If all providers fail, the last
        exception is raised.
        """
        start_idx = next(self._rr_counter) % len(self._providers)
        last_exc: Exception | None = None

        for offset in range(len(self._providers)):
            idx = (start_idx + offset) % len(self._providers)
            provider = self._providers[idx]
            try:
                result = await provider.complete(messages, response_format)
                if offset > 0:
                    logger.info(
                        "Pool failover: %s succeeded on provider %s (attempt %d)",
                        self.model, provider.provider_name, offset + 1,
                    )
                return result
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Pool failover: %s failed on provider %s: %s: %s",
                    self.model, provider.provider_name,
                    type(exc).__name__, exc,
                )

        raise last_exc  # type: ignore[misc]
