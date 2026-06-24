from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

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


def load_core_config() -> CoreConfig:
    load_dotenv(ENV_PATH, override=False)
    host = os.getenv("AGENT_ASISTANT_HOST") or "127.0.0.1"
    port = int(os.getenv("AGENT_ASISTANT_PORT") or "7438")
    return CoreConfig(host=host, port=port)


def load_llm_config() -> LlmConfig:
    load_dotenv(ENV_PATH, override=False)
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    anthropic_base = os.getenv("ANTHROPIC_BASE_URL")
    api_key = (
        anthropic_key
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("AGENT_ASISTANT_DEEPSEEK_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "Missing ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, or "
            "AGENT_ASISTANT_DEEPSEEK_API_KEY in .env"
        )
    api_mode = "anthropic" if anthropic_key or anthropic_base else "chat_completions"
    if api_mode == "anthropic":
        base_url = (anthropic_base or "https://api.deepseek.com/anthropic").rstrip("/")
    else:
        base_url = (
            os.getenv("DEEPSEEK_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("AGENT_ASISTANT_LLM_BASE_URL")
            or "https://api.deepseek.com"
        ).rstrip("/")
    if api_mode == "anthropic":
        model = (
            os.getenv("ANTHROPIC_MODEL")
            or os.getenv("KAMA_LLM_DEFAULT_MODEL")
            or os.getenv("AGENT_ASISTANT_LLM_MODEL")
            or os.getenv("DEEPSEEK_MODEL")
            or "deepseek-v4-flash"
        )
    else:
        model = (
            os.getenv("DEEPSEEK_MODEL")
            or os.getenv("AGENT_ASISTANT_LLM_MODEL")
            or os.getenv("KAMA_LLM_DEFAULT_MODEL")
            or "deepseek-chat"
        )
    return LlmConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        api_mode=api_mode,
    )
