"""A localhost embedding server with no API key indexes end to end.

The indexing chain is: Settings catalog -> EmbeddingConfig -> EmbeddingClient ->
provider adapter. Every link runs here for real; the only thing replaced is the
socket, by an httpx MockTransport that records the request. That is how "no API
key, fully local" is proven without a network and without a downloaded model.

What this does NOT prove: that a real Ollama/vLLM/LM Studio/Lemonade server
answers this way, or that its vectors match the fake ones.
"""

from __future__ import annotations

import json

import httpx
import pytest

from deeptutor.services.embedding.client import EmbeddingClient
from deeptutor.services.embedding.config import get_embedding_config
from deeptutor.services.rag.embedding_binding import binding_status
from deeptutor.services.rag.embedding_signature import signature_from_config

LOCAL_ENDPOINT = "http://localhost:11434/api/embed"
PUBLIC_ENDPOINT = "https://api.openai.com/v1/embeddings"
MODEL = "nomic-embed-text"


def _catalog(*, base_url: str = LOCAL_ENDPOINT, api_key: str = "") -> dict:
    profile = {
        "id": "profile-embedding",
        "name": "Embedding",
        "binding": "openai",
        "base_url": base_url,
        "api_key": api_key,
        "models": [{"id": "model-embedding", "name": MODEL, "model": MODEL}],
    }
    return {
        "version": 1,
        "services": {
            "llm": {"active_profile_id": None, "active_model_id": None, "profiles": []},
            "embedding": {
                "active_profile_id": "profile-embedding",
                "active_model_id": "model-embedding",
                "profiles": [profile],
            },
            "search": {"active_profile_id": None, "profiles": []},
        },
    }


def _selection() -> dict:
    return {"profile_id": "profile-embedding", "model_id": "model-embedding"}


def test_a_local_endpoint_resolves_with_a_completely_empty_api_key() -> None:
    config = get_embedding_config(catalog=_catalog(api_key=""))

    assert config.api_key == ""
    assert config.provider_mode == "local"
    assert config.binding == "ollama"
    assert config.effective_url == LOCAL_ENDPOINT
    assert config.model == MODEL


@pytest.mark.asyncio
async def test_indexing_embeds_through_the_local_server_without_an_auth_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real client/adapter path, with only the socket replaced."""
    recorded: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        return httpx.Response(
            200,
            json={"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], "model": MODEL},
        )

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs),
    )

    vectors = await EmbeddingClient(get_embedding_config(catalog=_catalog())).embed(
        ["first chunk", "second chunk"]
    )

    assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert len(recorded) == 1
    request = recorded[0]
    assert str(request.url) == LOCAL_ENDPOINT
    # No key configured means no key sent: nothing to leak, nothing to demand.
    assert "authorization" not in {name.lower() for name in request.headers}
    sent = json.loads(request.content)
    assert sent["input"] == ["first chunk", "second chunk"]
    assert sent["model"] == MODEL


def test_a_knowledge_base_bound_to_that_model_is_ready_not_blocked() -> None:
    """The gate that stops uploads/imports must let the keyless local binding in."""
    catalog = _catalog()
    config = get_embedding_config(catalog=catalog)
    entry = {
        "rag_provider": "llamaindex",
        "embedding_selection": _selection(),
        "embedding_signature": signature_from_config(config).hash(),
    }

    state, bound = binding_status(entry, catalog=catalog)

    assert state == "ready"
    assert bound is not None
    assert bound.api_key == ""
    assert bound.effective_url == LOCAL_ENDPOINT


def test_the_same_empty_key_is_still_refused_for_a_remote_endpoint() -> None:
    """The exemption is the endpoint, not a hole in the key check."""
    with pytest.raises(ValueError, match="Embedding API key not set"):
        get_embedding_config(catalog=_catalog(base_url=PUBLIC_ENDPOINT, api_key=""))
