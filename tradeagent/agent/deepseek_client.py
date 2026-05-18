from __future__ import annotations

from typing import Any

from openai import OpenAI

from tradeagent.config import get_settings


class DeepseekClient:
    """Thin wrapper over the OpenAI-compatible Deepseek chat-completions API."""

    def __init__(self, client: OpenAI | None = None):
        settings = get_settings()
        self.model = settings.deepseek_model
        self.client = client or OpenAI(
            api_key=settings.deepseek_api_key or "missing",
            base_url=settings.deepseek_base_url,
        )

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
