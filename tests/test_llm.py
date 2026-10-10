"""Generator selection and model fallback / escolha do gerador e alternância entre modelos."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from biorxiv_rag.config import Settings
from biorxiv_rag.llm import (
    API_FALLBACK_MODELS,
    CONNECTION_ATTEMPTS,
    ChatApiGenerator,
    GenerationError,
    build_generator,
)

NOTEBOOK_MODEL = "Qwen/Qwen2.5-14B-Instruct"
MESSAGES = [{"role": "user", "content": "question"}]


def settings(**changes):
    base = replace(Settings.from_env(), llm_backend="auto", llm_model="", llm_base_url="", llm_api_key="", hf_token="")
    return replace(base, **changes)


def chunk(text):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


class FakeClient:
    """Answers per model: a list of pieces or an exception / respostas por modelo: lista de trechos ou exceção."""

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def chat_completion(self, messages, model, max_tokens, temperature, stream):
        self.calls.append(model)
        answer = self.answers[model]
        if isinstance(answer, Exception):
            raise answer
        return iter([SimpleNamespace(choices=[]), *[chunk(piece) for piece in answer]])


def api_generator(answers):
    generator = ChatApiGenerator(list(answers), "hf_api", token="hf_test")
    generator.client = FakeClient(answers)
    return generator


def test_auto_backend_without_credentials_is_retrieval_only():
    assert build_generator(settings(), NOTEBOOK_MODEL) is None
    assert build_generator(settings(llm_backend="none", hf_token="hf_test"), NOTEBOOK_MODEL) is None


def test_auto_backend_with_token_tries_the_notebook_model_first():
    generator = build_generator(settings(hf_token="hf_test"), NOTEBOOK_MODEL)
    assert generator.backend == "hf_api"
    assert generator.models == [NOTEBOOK_MODEL, *API_FALLBACK_MODELS]
    assert build_generator(settings(hf_token="hf_test", llm_model="org/model"), NOTEBOOK_MODEL).models == ["org/model"]


def test_base_url_selects_the_openai_compatible_backend():
    generator = build_generator(settings(llm_base_url="http://ollama:11434/v1", llm_model="qwen2.5:7b"), NOTEBOOK_MODEL)
    assert generator.backend == "openai"
    assert generator.models == ["qwen2.5:7b"]
    with pytest.raises(ValueError):
        build_generator(settings(llm_backend="openai"), NOTEBOOK_MODEL)


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError):
        build_generator(settings(llm_backend="magic"), NOTEBOOK_MODEL)


def test_generation_streams_pieces_and_remembers_the_model():
    generator = api_generator({"first": ["Hello", " world"]})
    generation = generator.start(MESSAGES, 50)
    assert list(generation) == ["Hello", " world"]
    assert (generation.text, generation.model) == ("Hello world", "first")


def http_error(status, message):
    error = RuntimeError(message)
    error.response = SimpleNamespace(status_code=status)
    return error


def test_generation_falls_back_to_the_next_model():
    generator = api_generator({"first": http_error(400, "no provider"), "second": ["ok"]})
    generation = generator.start(MESSAGES, 50)
    assert list(generation) == ["ok"]
    assert generation.model == "second"
    assert generator.client.calls == ["first", "second"]


def test_generation_retries_a_connection_that_never_reached_the_server():
    generator = api_generator({"first": ConnectionError("certificate verify failed"), "second": ["ok"]})
    assert list(generator.start(MESSAGES, 50)) == ["ok"]
    assert generator.client.calls == ["first"] * CONNECTION_ATTEMPTS + ["second"]


def test_generation_reports_every_failure():
    generator = api_generator({"first": http_error(402, "payment required"), "second": []})
    with pytest.raises(GenerationError) as failure:
        list(generator.start(MESSAGES, 50))
    assert "first: HTTP 402" in str(failure.value)
    assert "second: empty answer" in str(failure.value)
