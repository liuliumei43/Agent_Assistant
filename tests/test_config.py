import os
from unittest.mock import patch

import pytest
from Agent_Assistant import config


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: None)
    with patch.dict(os.environ, {}, clear=True):
        yield


@pytest.mark.parametrize(
    "provider,url,mode",
    [
        ("openai", "https://api.openai.com/v1", "chat_completions"),
        ("anthropic", "https://api.anthropic.com", "anthropic"),
        ("deepseek", "https://api.deepseek.com", "chat_completions"),
    ],
)
def test_provider_key_and_endpoint_stay_together(monkeypatch, provider, url, mode):
    monkeypatch.setenv(f"{provider.upper()}_API_KEY", "fake")
    monkeypatch.setenv(f"{provider.upper()}_MODEL", "fixture-model")
    result = config.load_llm_config()
    assert (result.provider, result.base_url, result.api_mode) == (provider, url, mode)


def test_multiple_keys_require_selection(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic")
    monkeypatch.setenv("AGENT_ASSISTANT_LLM_MODEL", "fixture-model")
    with pytest.raises(ValueError, match="PROVIDER"):
        config.load_llm_config()
    monkeypatch.setenv("AGENT_ASSISTANT_PROVIDER", "openai")
    assert config.load_llm_config().api_key == "fake-openai"


def test_explicit_model_required(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
    with pytest.raises(ValueError, match="model"):
        config.load_llm_config()


def test_custom_gateway_is_explicit(monkeypatch):
    for key, value in {
        "AGENT_ASSISTANT_PROVIDER": "custom",
        "AGENT_ASSISTANT_API_KEY": "fake",
        "AGENT_ASSISTANT_LLM_MODEL": "fixture-model",
        "AGENT_ASSISTANT_LLM_BASE_URL": "https://gateway.example/anthropic",
        "AGENT_ASSISTANT_API_MODE": "anthropic",
    }.items():
        monkeypatch.setenv(key, value)
    result = config.load_llm_config()
    assert result.api_mode == "anthropic" and result.base_url.endswith("/anthropic")


def test_core_rejects_public_bind(monkeypatch):
    monkeypatch.setenv("AGENT_ASSISTANT_HOST", "0.0.0.0")
    with pytest.raises(ValueError, match="loopback"):
        config.load_core_config()
