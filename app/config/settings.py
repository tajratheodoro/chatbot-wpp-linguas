"""Application settings loaded from environment variables."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for external services and application behavior."""

    app_name: str = "chatbot-wpp-linguas"
    app_env: str = Field(default="development", validation_alias="APP_ENV")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    api_prefix: str = "/api"

    database_url: str = Field(default="", validation_alias="DATABASE_URL")
    evolution_api_url: str = Field(
        default="http://localhost:8080",
        validation_alias="EVOLUTION_API_URL",
    )
    evolution_api_key: str = Field(default="", validation_alias="EVOLUTION_API_KEY")
    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")
    edge_tts_voice: str = Field(default="en-US-AriaNeural", validation_alias="EDGE_TTS_VOICE")
    nemo_guardrails_config_path: str = Field(
        default="app/core/guardrails_config",
        validation_alias="NEMO_GUARDRAILS_CONFIG_PATH",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
