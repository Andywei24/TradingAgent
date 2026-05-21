from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # LLM — pick the active provider; both speak the OpenAI chat-completions protocol.
    llm_provider: str = "deepseek"  # deepseek | openai

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    # Alpha Vantage
    alphavantage_api_key: str = ""
    alphavantage_base_url: str = "https://www.alphavantage.co/query"
    alphavantage_rate_limit_per_min: int = 5
    alphavantage_default_exchange: str = "NASDAQ"
    # The free tier blocks several features: adjusted closes (TIME_SERIES_DAILY_ADJUSTED)
    # and outputsize=full. Leave False on a free key — we then use TIME_SERIES_DAILY with
    # outputsize=compact (latest ~100 bars, adj_close == close). Set True with a premium key.
    alphavantage_premium: bool = False

    # Storage
    db_url: str = "sqlite:///data/market.db"
    data_dir: Path = Field(default=Path("data"))
    rag_index_dir: Path = Field(default=Path("data/faiss_index"))

    # RAG
    embed_model: str = "BAAI/bge-small-en-v1.5"

    # Misc
    log_level: str = "INFO"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "csv").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "reports").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "knowledge_base").mkdir(parents=True, exist_ok=True)
        self.rag_index_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
