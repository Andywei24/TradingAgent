from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from tradeagent.config import get_settings


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    api_key: str
    base_url: str | None
    model: str


def _provider_config(provider: str) -> ProviderConfig:
    """Resolve credentials + default model for a named provider from settings."""
    settings = get_settings()
    provider = provider.lower()
    if provider == "deepseek":
        return ProviderConfig(
            name="deepseek",
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model=settings.deepseek_model,
        )
    if provider == "openai":
        return ProviderConfig(
            name="openai",
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.openai_model,
        )
    raise ValueError(f"unknown LLM provider {provider!r}; supported: deepseek, openai")


class LLMClient:
    """Provider-agnostic wrapper over any OpenAI-compatible chat-completions API.

    Deepseek and OpenAI share the same wire protocol (the ``openai`` SDK), so a single
    client serves both — only api_key / base_url / model differ. Add another provider by
    extending ``_provider_config``.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None,
        model: str,
        provider: str = "custom",
        client: OpenAI | None = None,
    ):
        self.provider = provider
        self.model = model
        self.client = client or OpenAI(api_key=api_key or "missing", base_url=base_url)

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict = "auto",
        temperature: float = 0.0,
        response_format: dict | None = None,
    ) -> Any:
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        if response_format:
            kwargs["response_format"] = response_format
        return self.client.chat.completions.create(**kwargs)


def make_llm_client(
    provider: str | None = None,
    *,
    model: str | None = None,
    client: OpenAI | None = None,
) -> LLMClient:
    """Build an LLMClient for the given provider (defaults to settings.llm_provider).

    ``model`` overrides the provider's configured default; ``client`` injects a
    pre-built OpenAI instance (used by tests).
    """
    provider = provider or get_settings().llm_provider
    cfg = _provider_config(provider)
    return LLMClient(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        model=model or cfg.model,
        provider=cfg.name,
        client=client,
    )
