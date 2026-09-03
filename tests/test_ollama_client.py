"""Tests for the local Ollama LLM client."""

from unittest import mock

import pytest

from src.aegis.ollama_client import OllamaClient
from src.aegis.schemas import DataQualityReport


def _make_fake_response(response_text: str = "synthetic response") -> dict:
    return {"response": response_text}


def test_default_model_and_host() -> None:
    """OllamaClient uses the expected defaults."""
    with mock.patch("src.aegis.ollama_client.Client"):
        client = OllamaClient()

    assert client.model == "qwen3:1.7b"
    assert client.host == "http://localhost:11434"


def test_custom_model_can_be_supplied() -> None:
    """A custom model can be passed to the constructor."""
    with mock.patch("src.aegis.ollama_client.Client"):
        client = OllamaClient(model="custom-model")

    assert client.model == "custom-model"
    assert client.host == "http://localhost:11434"


def test_generate_sends_prompt_and_returns_response() -> None:
    """generate() forwards the prompt and returns the response text."""
    fake_client = mock.MagicMock()
    fake_client.generate.return_value = _make_fake_response("local result")

    with mock.patch("src.aegis.ollama_client.Client", return_value=fake_client):
        client = OllamaClient()

    text = client.generate("test prompt")

    assert text == "local result"
    fake_client.generate.assert_called_once_with(
        model="qwen3:1.7b",
        prompt="test prompt",
        stream=False,
        format="json",
        options={"temperature": 0.1},
    )


def test_custom_model_is_used_in_generate() -> None:
    """A custom model is passed through to the Ollama generate call."""
    fake_client = mock.MagicMock()
    fake_client.generate.return_value = _make_fake_response("custom model result")

    with mock.patch("src.aegis.ollama_client.Client", return_value=fake_client):
        client = OllamaClient(model="custom-model")

    text = client.generate("prompt for custom model")

    assert text == "custom model result"
    fake_client.generate.assert_called_once_with(
        model="custom-model",
        prompt="prompt for custom model",
        stream=False,
        format="json",
        options={"temperature": 0.1},
    )


def test_provider_error_becomes_runtime_error() -> None:
    """Exceptions from Ollama are wrapped in a beginner-friendly RuntimeError."""
    fake_client = mock.MagicMock()
    fake_client.generate.side_effect = ConnectionError("connection refused")

    with mock.patch("src.aegis.ollama_client.Client", return_value=fake_client):
        client = OllamaClient()

    with pytest.raises(RuntimeError, match="verify that") as exc:
        client.generate("boom")

    assert "Ollama is running" in str(exc.value)
    assert "qwen3:1.7b is installed" in str(exc.value)
    assert "localhost:11434 is reachable" in str(exc.value)


def test_missing_response_field_becomes_runtime_error() -> None:
    """Unexpected Ollama responses are surfaced as RuntimeError."""
    fake_client = mock.MagicMock()
    fake_client.generate.return_value = {"not_response": "oops"}

    with mock.patch("src.aegis.ollama_client.Client", return_value=fake_client):
        client = OllamaClient()

    with pytest.raises(RuntimeError, match="unexpected response"):
        client.generate("prompt")


def test_response_schema_json_schema_is_passed_to_ollama() -> None:
    """When response_schema is provided, its JSON schema is passed as format."""
    fake_client = mock.MagicMock()
    fake_client.generate.return_value = _make_fake_response("structured result")

    with mock.patch("src.aegis.ollama_client.Client", return_value=fake_client):
        client = OllamaClient()

    client.generate("prompt", response_schema=DataQualityReport)

    call_kwargs = fake_client.generate.call_args.kwargs
    assert call_kwargs["format"] == DataQualityReport.model_json_schema()
