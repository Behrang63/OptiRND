"""
Phase 3, Prompt 1 tests: OllamaLLMProvider, schema-bound generation,
factory routing, and the no-crash fallback guarantee.

No live Ollama server is required - every network interaction is monkeypatched
at the _chat() boundary.
"""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import BaseModel, ValidationError

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from agents.llm_factory import get_provider as get_provider_facade  # noqa: E402
from agents.ollama_provider import OllamaLLMProvider  # noqa: E402
from core.contracts import QualitativeAssessment  # noqa: E402
from core.llm_provider import (  # noqa: E402
    BaseLLMProvider,
    LLMFactory,
    MockLLMProvider,
    SchemaValidationError,
)


VALID_PAYLOAD = {
    "technical_success_probability": 0.9,
    "recommended_action": "APPROVE",
    "risk_summary": "ریسک قابل مدیریت است",
}


@pytest.fixture()
def provider():
    return OllamaLLMProvider(model="llama3", host="http://localhost:11434",
                             timeout_seconds=1.0, max_retries=2)


# --------------------------------------------------------------------------- #
# Factory routing
# --------------------------------------------------------------------------- #
def test_factory_mock_default():
    assert isinstance(LLMFactory.get_provider(), MockLLMProvider)
    assert isinstance(LLMFactory.get_provider("mock"), MockLLMProvider)


def test_factory_ollama_returns_provider_with_mock_guarantee():
    """Requesting ollama returns an OllamaLLMProvider (mock built in), never None/crash."""
    p = LLMFactory.get_provider("ollama")
    assert isinstance(p, BaseLLMProvider)


def test_factory_unknown_type_falls_back_to_mock():
    assert isinstance(LLMFactory.get_provider("totally-unknown"), MockLLMProvider)


def test_facade_env_routing(monkeypatch):
    monkeypatch.setenv("OPTIRND_LLM_PROVIDER", "mock")
    assert isinstance(get_provider_facade(), MockLLMProvider)
    # Explicit argument beats the env var:
    assert isinstance(get_provider_facade("unknown-x"), MockLLMProvider)


# --------------------------------------------------------------------------- #
# generate_validated contract (base + mock)
# --------------------------------------------------------------------------- #
def test_base_generate_validated_accepts_valid_payload():
    mock = MockLLMProvider()
    out = mock.generate_validated("ارزیابی پروژه", "schema", QualitativeAssessment)
    assert 0.0 <= out["technical_success_probability"] <= 1.0


def test_base_generate_validated_raises_schema_error_on_violation(monkeypatch):
    mock = MockLLMProvider()

    def broken_json(prompt, schema_description):
        return {"technical_success_probability": "not-a-number"}

    monkeypatch.setattr(mock, "generate_json", broken_json)
    with pytest.raises(SchemaValidationError):
        mock.generate_validated("p", "schema", QualitativeAssessment)


# --------------------------------------------------------------------------- #
# OllamaLLMProvider - successful payload processing
# --------------------------------------------------------------------------- #
def test_generate_validated_success_path(provider):
    with patch.object(provider, "_chat", return_value='{"technical_success_probability": 0.9, "recommended_action": "APPROVE", "risk_summary": "ok"}'):
        out = provider.generate_validated("prompt", "schema", QualitativeAssessment)
    assert out["recommended_action"] == "APPROVE"
    assert out["technical_success_probability"] == 0.9


def test_json_extraction_tolerates_markdown_fences_and_prose(provider):
    wrapped = 'Here you go:\n```json\n{"technical_success_probability": 0.5, "recommended_action": "REJECT", "risk_summary": "s"}\n```'
    with patch.object(provider, "_chat", return_value=wrapped):
        out = provider.generate_validated("p", "schema", QualitativeAssessment)
    assert out["recommended_action"] == "REJECT"


# --------------------------------------------------------------------------- #
# Retry guards & fallback behavior (the no-crash contract)
# --------------------------------------------------------------------------- #
def test_malformed_json_retried_then_falls_back_to_mock(provider):
    calls = {"n": 0}

    def flaky_chat(prompt):
        calls["n"] += 1
        return "definitely not json {{"

    with patch.object(provider, "_chat", side_effect=flaky_chat):
        out = provider.generate_validated("p", "schema", QualitativeAssessment)

    assert calls["n"] == 1 + provider.max_retries      # bounded retries
    assert "technical_success_probability" in out      # deterministic mock output


def test_schema_violation_retried_then_falls_back_to_mock(provider):
    calls = {"n": 0}

    def invalid_json(prompt):
        calls["n"] += 1
        return '{"technical_success_probability": 7.5}'  # out of [0, 1]

    with patch.object(provider, "_chat", side_effect=invalid_json):
        out = provider.generate_validated("p", "schema", QualitativeAssessment)

    assert calls["n"] == 1 + provider.max_retries
    assert 0.0 <= out["technical_success_probability"] <= 1.0


def test_unreachable_server_falls_back_to_mock(provider):
    """Connection-refused style failure -> deterministic mock, never an exception."""
    with patch.object(provider, "_chat", return_value=None):
        out = provider.generate_validated("p", "schema", QualitativeAssessment)
    assert isinstance(out, dict)
    assert "risk_summary" in out


def test_generate_text_falls_back_to_mock(provider):
    with patch.object(provider, "_chat", return_value=None):
        text = provider.generate_text("تحلیل ریسک")
    assert "[پاسخ آفلاین]" in text


def test_generate_json_without_schema_returns_mock_contract(provider):
    """
    Contract-first rule: unvalidated LLM JSON must not enter the pipeline.
    The legacy generate_json() path degrades to the deterministic mock contract
    (schema-validated output shape), never raw model output.
    """
    with patch.object(provider, "_chat", return_value='{"anything": true}'):
        out = provider.generate_json("p", "schema")
    # Must match the mock contract shape - NOT the raw model payload.
    assert "technical_success_probability" in out
    assert "anything" not in out
    assert 0.0 <= out["technical_success_probability"] <= 1.0


def test_is_available_false_when_server_down(provider):
    with patch.object(provider, "_chat", return_value=None), \
         patch("agents.ollama_provider.urllib.request.urlopen",
               side_effect=OSError("connection refused")):
        assert provider.is_available() is False


def test_chat_swallows_network_errors(provider):
    with patch("agents.ollama_provider.urllib.request.urlopen",
               side_effect=OSError("connection refused")):
        assert provider._chat("prompt") is None


# --------------------------------------------------------------------------- #
# Agent integration - schema-bound assessment inside ExAnteAgent
# --------------------------------------------------------------------------- #
def _minimal_proposal():
    return {
        "title": "پروژه تست",
        "cost_p10": 3000.0, "cost_p50": 4000.0, "cost_p90": 5500.0,
        "benefit_p10": 6000.0, "benefit_p50": 8000.0, "benefit_p90": 11000.0,
        "p_success": 0.85,
    }


def test_agent_produces_valid_assessment_with_ollama_provider():
    """Agent works with the Ollama provider even when the server is absent."""
    from agents.main_agents.ex_ante_agent import ExAnteAgent

    agent = ExAnteAgent(llm_provider=LLMFactory.get_provider("ollama"))
    result = agent.evaluate_comprehensive_proposal(_minimal_proposal())
    # Output contract enforcement still holds on the degraded path:
    q = result["qualitative_assessment"]
    assert 0.0 <= q["technical_success_probability"] <= 1.0
    assert isinstance(q["risk_summary"], str) and q["risk_summary"]


def test_agent_uses_provider_retries_not_local_only(monkeypatch):
    """Agent prefers generate_validated when the provider offers it."""
    from agents.main_agents.ex_ante_agent import ExAnteAgent

    agent = ExAnteAgent(llm_provider=MockLLMProvider())
    called = {"validated": False}

    def fake_validated(prompt, schema_description, schema_model):
        called["validated"] = True
        return schema_model.model_validate(VALID_PAYLOAD).model_dump()

    monkeypatch.setattr(agent.llm_provider, "generate_validated", fake_validated,
                        raising=False)
    result = agent.evaluate_comprehensive_proposal(_minimal_proposal())
    assert called["validated"] is True
    assert result["qualitative_assessment"]["recommended_action"] == "APPROVE"


def test_agent_survives_provider_raising_schema_error():
    """A provider that violates the schema is contained; mock assessment is used."""
    from agents.main_agents.ex_ante_agent import ExAnteAgent

    class RogueProvider(BaseLLMProvider):
        def generate_json(self, prompt, schema_description):
            return {"garbage": True}

        def generate_text(self, prompt):
            return "x"

        def interpret_monte_carlo(self, metrics, p_success):
            return "x"

    agent = ExAnteAgent(llm_provider=RogueProvider())
    result = agent.evaluate_comprehensive_proposal(_minimal_proposal())
    q = result["qualitative_assessment"]
    assert 0.0 <= q["technical_success_probability"] <= 1.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
