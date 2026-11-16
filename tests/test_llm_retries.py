"""Reintentos de Gemini: se cuentan, se muestran y no se repite lo inútil (cuota diaria)."""
from types import SimpleNamespace

import pytest

from llm import GeminiLLM, LLMError


def fake_response():
    part = SimpleNamespace(text="hola", function_call=None, thought=False)
    return SimpleNamespace(text="hola", candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))],
                           usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=2, thoughts_token_count=0))


def make(errors, monkeypatch):
    llm = GeminiLLM(api_key="x")
    pending = list(errors)

    def generate_content(**kwargs):
        if pending:
            raise Exception(pending.pop(0))
        return fake_response()
    llm._client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    sleeps = []
    monkeypatch.setattr("llm.time.sleep", lambda s: sleeps.append(s))
    return llm, sleeps


def test_retries_are_counted_and_waits_recorded(monkeypatch):
    llm, sleeps = make(["503 UNAVAILABLE", "429 RESOURCE_EXHAUSTED PerMinute"], monkeypatch)
    assert llm.generate("sys", "user") == "hola"
    assert llm.usage["retries"] == 2 and llm.usage["wait_s"] == 9 and sleeps == [3, 6] and llm.usage["calls"] == 1


def test_daily_quota_fails_fast_without_waiting(monkeypatch):
    llm, sleeps = make(["429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel-FreeTier"], monkeypatch)
    with pytest.raises(LLMError, match="DIARIA"):
        llm.generate("sys", "user")
    assert sleeps == [] and llm.usage["retries"] == 0


def test_non_retryable_error_raises_immediately(monkeypatch):
    llm, sleeps = make(["400 INVALID_ARGUMENT bad request"], monkeypatch)
    with pytest.raises(LLMError):
        llm.generate("sys", "user")
    assert sleeps == []
