from __future__ import annotations

from openai import OpenAI

from tradeagent.agent.llm import LLMClient, make_llm_client


class DeepseekClient(LLMClient):
    """Backwards-compatible Deepseek client.

    Kept so existing imports keep working; prefer ``make_llm_client(provider)`` for
    multi-provider support.
    """

    def __init__(self, client: OpenAI | None = None):
        base = make_llm_client("deepseek", client=client)
        super().__init__(
            api_key="",
            base_url=None,
            model=base.model,
            provider="deepseek",
            client=base.client,
        )
