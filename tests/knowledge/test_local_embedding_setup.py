"""The local, no-API-key way to embed, and the honest verdict about it.

A knowledge base indexes through an embedding model resolved from Settings.
Nothing in DeepTutor runs a model in-process, so the fully local option is an
embedding server on this machine; those servers do not use API keys. This file
pins the single source of that answer: which endpoints count as local, whether
an endpoint needs network access, and the sentence a blocked user is shown.

What this does NOT prove: that any embedding server is running. Nothing here
opens a socket.
"""

from __future__ import annotations

import pytest

from deeptutor.knowledge import embedding_setup

PUBLIC_ENDPOINT = "https://api.openai.com/v1/embeddings"

# Loopback, container gateway, LAN and mDNS endpoints: all reachable without an
# external service, and all of them are the same "local, keyless" class the
# resolver already treats as provider_mode == "local".
LOCAL_ENDPOINTS = [
    "http://localhost:11434/api/embed",
    "http://127.0.0.1:8000/v1/embeddings",
    "http://[::1]:1234/v1/embeddings",
    "http://host.docker.internal:13305/v1/embeddings",
    "http://192.168.1.40:1234/v1/embeddings",
    "http://ollama:11434/api/embed",
    "http://gpu-box.local:8000/v1/embeddings",
]

DEFAULT_MODEL = "nomic-embed-text"


def _catalog(
    *,
    binding: str = "openai",
    base_url: str = "",
    api_key: str = "",
    model: str = DEFAULT_MODEL,
) -> dict:
    """One active embedding profile/model, shaped like the real catalog."""
    profile = {
        "id": "profile-embedding",
        "name": "Embedding",
        "binding": binding,
        "base_url": base_url,
        "api_key": api_key,
        "models": [{"id": "model-embedding", "name": model, "model": model}],
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


@pytest.mark.parametrize("endpoint", LOCAL_ENDPOINTS)
def test_a_local_endpoint_needs_no_api_key_and_no_network(endpoint: str) -> None:
    target = embedding_setup.describe_embedding_target(
        catalog=_catalog(base_url=endpoint, api_key="")
    )

    assert target.configured is True, target.error
    assert target.endpoint == endpoint
    assert target.is_local is True
    assert target.requires_network is False
    assert target.needs_api_key is False


def test_a_remote_endpoint_is_marked_as_requiring_network_access() -> None:
    """Not a vague "unavailable": the endpoint is a remote service."""
    target = embedding_setup.describe_embedding_target(
        catalog=_catalog(base_url=PUBLIC_ENDPOINT, api_key="")
    )

    assert target.configured is False
    assert target.error and "API key" in target.error
    assert target.is_local is False
    assert target.requires_network is True
    assert target.needs_api_key is True
    assert PUBLIC_ENDPOINT in target.message
    assert "needs network access" in target.message


def test_nothing_configured_says_so_instead_of_raising() -> None:
    target = embedding_setup.describe_embedding_target(catalog={})

    assert target.configured is False
    assert target.endpoint is None
    assert target.requires_network is None
    assert target.needs_api_key is False
    assert "No embedding model is configured" in target.message


def test_the_local_option_names_the_servers_and_says_no_key_is_needed() -> None:
    option = embedding_setup.LOCAL_EMBEDDING_OPTION

    for server in ("Ollama", "vLLM", "LM Studio", "Lemonade"):
        assert server in option
    assert "no API key" in option
    assert "localhost" in option.lower()
    assert "fully local" in option.lower()


def test_the_unavailable_binding_sentence_covers_both_local_and_remote() -> None:
    text = embedding_setup.EMBEDDING_BINDING_UNAVAILABLE

    assert "unavailable" in text
    for server in ("Ollama", "vLLM", "LM Studio", "Lemonade"):
        assert server in text
    assert "no API key" in text
    assert "network access" in text.lower()


def test_explaining_a_failure_keeps_the_problem_and_adds_the_way_out() -> None:
    detail = embedding_setup.explain_embedding_problem(
        "Embedding API key not set.", catalog=_catalog(base_url=PUBLIC_ENDPOINT)
    )

    assert detail.startswith("Embedding API key not set.")
    assert "needs network access" in detail
    assert embedding_setup.LOCAL_EMBEDDING_OPTION in detail


def test_a_local_endpoint_failure_is_not_blamed_on_the_network() -> None:
    """A stale binding on a localhost profile is a missing model, not a key."""
    detail = embedding_setup.explain_embedding_problem(
        "The embedding model bound to this knowledge base was deleted.",
        selection={"profile_id": "gone", "model_id": "gone"},
        catalog=_catalog(base_url="http://localhost:11434/api/embed", api_key=""),
    )

    assert "needs network access" not in detail
    assert "http://localhost:11434/api/embed" in detail
    assert "no API key" in detail


def test_host_classification_agrees_with_the_provider_runtime_twin() -> None:
    """Our host rule must not drift from the one the resolver already uses."""
    from deeptutor.services.config.provider_runtime import _is_nonpublic_embedding_host

    hosts = [
        "localhost",
        "127.0.0.1",
        "::1",
        "host.docker.internal",
        "lemonade",
        "ollama",
        "192.168.1.40",
        "10.0.0.5",
        "172.16.0.9",
        "169.254.1.1",
        "gpu-box.local",
        "svc.internal",
        "api.openai.com",
        "api.example.com",
        "8.8.8.8",
    ]
    for host in hosts:
        assert embedding_setup.host_is_local(host) is _is_nonpublic_embedding_host(host), host
