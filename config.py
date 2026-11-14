from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent / ".env"


@dataclass(frozen=True)
class CoreConfig:
    host: str
    port: int


@dataclass(frozen=True)
class LlmConfig:
    api_key: str
    base_url: str
    model: str
    api_mode: str
    provider: str = "custom"
    max_output_tokens: int = 4096
    thinking: str = "default"

    @property
    def profile(self) -> str:
        return f"{self.provider}|{self.api_mode}|{self.base_url}|{self.model}"


def load_core_config() -> CoreConfig:
    load_dotenv(ENV_PATH, override=False)
    host = os.getenv("AGENT_ASSISTANT_HOST", "127.0.0.1")
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Core is local-only; AGENT_ASSISTANT_HOST must be a loopback address")
    return CoreConfig(host, int(os.getenv("AGENT_ASSISTANT_PORT", "7438")))


def state_directory() -> Path:
    return Path(
        os.getenv("AGENT_ASSISTANT_STATE_DIR", str(ENV_PATH.parent / ".agent-state"))
    ).resolve()


def load_llm_config() -> LlmConfig:
    load_dotenv(ENV_PATH, override=False)
    provider = os.getenv("AGENT_ASSISTANT_PROVIDER", "").lower()
    keys = {
        "deepseek": os.getenv("DEEPSEEK_API_KEY") or os.getenv("AGENT_ASSISTANT_DEEPSEEK_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "custom": os.getenv("AGENT_ASSISTANT_API_KEY"),
    }
    if not provider:
        present = [name for name, key in keys.items() if key]
        if len(present) != 1:
            raise ValueError("Set AGENT_ASSISTANT_PROVIDER when zero/multiple keys exist")
        provider = present[0]
    if provider not in keys:
        raise ValueError("Provider must be deepseek, openai, anthropic, or custom")
    api_key = keys[provider]
    if not api_key:
        raise ValueError(f"Missing API key for selected provider: {provider}")
    defaults = {
        "deepseek": ("https://api.deepseek.com", "chat_completions"),
        "openai": ("https://api.openai.com/v1", "chat_completions"),
        "anthropic": ("https://api.anthropic.com", "anthropic"),
        "custom": ("", "chat_completions"),
    }
    default_url, default_mode = defaults[provider]
    prefix = provider.upper()
    base_url = (
        os.getenv("AGENT_ASSISTANT_LLM_BASE_URL") or os.getenv(f"{prefix}_BASE_URL") or default_url
    ).rstrip("/")
    model = os.getenv("AGENT_ASSISTANT_LLM_MODEL") or os.getenv(f"{prefix}_MODEL")
    model = model or os.getenv("KAMA_LLM_DEFAULT_MODEL")
    if not base_url or not model:
        raise ValueError("Set a model explicitly; custom providers also require a base URL")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username:
        raise ValueError("Invalid model base URL")
    mode = os.getenv("AGENT_ASSISTANT_API_MODE", default_mode)
    if mode not in {"chat_completions", "anthropic"}:
        raise ValueError("API mode must be chat_completions or anthropic")
    thinking = os.getenv("AGENT_ASSISTANT_THINKING", "default")
    if thinking not in {"default", "enabled", "disabled"}:
        raise ValueError("Thinking must be default, enabled, or disabled")
    max_tokens = int(os.getenv("AGENT_ASSISTANT_MAX_OUTPUT_TOKENS", "4096"))
    if max_tokens < 1:
        raise ValueError("Output token budget must be positive")
    return LlmConfig(api_key, base_url, model, mode, provider, max_tokens, thinking)
