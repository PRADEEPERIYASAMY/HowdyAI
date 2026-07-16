"""
tests/test_guardrail.py – Tests for guardrail logic.

Covers both the legacy QueryGuardrail (still used standalone in some paths)
and the new merged GuardrailRewriter. Adversarial spot-checks are the
critical section: merging the two prompts can dilute safety focus if the
model starts optimizing for the rewrite task over the safety judgment.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from src.guardrail import QueryGuardrail


# ── Helpers ────────────────────────────────────────────────────────────────────

class MockResponse:
    def __init__(self, text):
        self.content = text


def make_json_response(safe: bool, rewritten: str | None = None) -> MockResponse:
    payload = {"safe": safe, "rewritten_query": rewritten}
    return MockResponse(json.dumps(payload))


# ── Legacy QueryGuardrail ──────────────────────────────────────────────────────

@pytest.fixture
def mock_config_legacy():
    config = MagicMock()
    config.GUARDRAIL_TEMPLATE_PATH = "dummy/path.txt"
    return config


def test_guardrail_in_scope(mock_config_legacy, monkeypatch):
    mock_model_class = MagicMock()
    mock_model_instance = MagicMock()
    mock_model_instance.invoke.return_value = MockResponse("IN_SCOPE")
    mock_model_class.return_value = mock_model_instance
    monkeypatch.setattr("src.guardrail.OpenAILanguageModel", mock_model_class)

    guardrail = QueryGuardrail(mock_config_legacy)
    is_in_scope, msg = guardrail.check("Where is the library?")
    assert is_in_scope is True
    assert msg == ""


def test_guardrail_out_of_scope(mock_config_legacy, monkeypatch):
    mock_model_class = MagicMock()
    mock_model_instance = MagicMock()
    mock_model_instance.invoke.return_value = MockResponse("OUT_OF_SCOPE")
    mock_model_class.return_value = mock_model_instance
    monkeypatch.setattr("src.guardrail.OpenAILanguageModel", mock_model_class)

    guardrail = QueryGuardrail(mock_config_legacy)
    is_in_scope, msg = guardrail.check("What is the capital of France?")
    assert is_in_scope is False
    assert "specifically for Texas A&M" in msg


def test_guardrail_api_error_fallback(mock_config_legacy, monkeypatch):
    mock_model_class = MagicMock()
    mock_model_instance = MagicMock()
    mock_model_instance.invoke.side_effect = Exception("API Down")
    mock_model_class.return_value = mock_model_instance
    monkeypatch.setattr("src.guardrail.OpenAILanguageModel", mock_model_class)

    guardrail = QueryGuardrail(mock_config_legacy)
    is_in_scope, msg = guardrail.check("Where is the library?")
    assert is_in_scope is True
    assert msg == ""


