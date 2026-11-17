"""The local, no-API-key way to embed, and the honest verdict about it.

A knowledge base indexes through one embedding model resolved from Settings.
Nothing in DeepTutor runs a model in-process, so a fully local deployment means
an embedding *server* on this machine: Ollama, vLLM, LM Studio or Lemonade.
Those servers do not use API keys, and the resolver already treats them as
provider_mode == "local" -- but the failure paths used to say only that
something was "unavailable", which told a user without a key nothing about
what to do next.

This module is the single source of that answer: which endpoints count as
local, whether an endpoint needs network access, and the sentence a blocked
user is shown. It adds no provider, adapter or model runtime; it only explains
the ones that already exist.
"""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
from typing import Any
from urllib.parse import urlparse

# The embedding servers that run beside DeepTutor instead of behind someone
# else's API. Names only: no endpoint here implies a new provider.
LOCAL_EMBEDDING_SERVERS = ("Ollama", "vLLM", "LM Studio", "Lemonade")

# The way out of "no API key configured", in one sentence a user can act on.
LOCAL_EMBEDDING_OPTION = (
    "A localhost embedding server satisfies a fully local deployment and needs no API key: "
    "point the embedding profile's endpoint at Ollama (http://localhost:11434/api/embed), "
    "vLLM or LM Studio (http://localhost:8000/v1/embeddings), or Lemonade "
    "(http://localhost:13305/v1/embeddings), leave the API key empty, and select that model."
)

# Nothing resolved at all: no profile, no model, nothing to send a request to.
NO_EMBEDDING_MODEL = (
    "No embedding model is configured, so this knowledge base cannot be indexed yet."
)

# The endpoint is somebody else's service: indexing needs the internet and the
# provider's key, which is exactly what a "fully local" user does not have.
REMOTE_ENDPOINT_NEEDS_NETWORK = (
    "The configured embedding endpoint {endpoint} is a remote service, so indexing needs "
    "network access"
)

# The endpoint is on this machine or the local network: no key and no internet.
LOCAL_ENDPOINT_NEEDS_NO_KEY = (
    "The configured embedding endpoint {endpoint} is local, so it needs no API key and no "
    "internet access."
)

# The batch gate's reason. The import endpoint's per-file reason is this same
# sentence verbatim (a paraphrase would tell the user something the knowledge
# base never said), so both surfaces must be edited together.
EMBEDDING_BINDING_UNAVAILABLE = (
    "The bound embedding model is unavailable or changed: the model was deleted, its settings "
    "changed, or it no longer resolves. Restore it in Settings > Embedding models, or select a "
    "model to re-index this knowledge base. A localhost embedding server (Ollama, vLLM, "
    "LM Studio or Lemonade) satisfies a fully local deployment and needs no API key; a model on "
    "a remote API needs network access and its API key."
)

# Loopback, container-gateway and orchestrator names. Deliberately the same set
# the resolver uses in provider_runtime._is_nonpublic_embedding_host;
# test_host_classification_agrees_with_the_provider_runtime_twin keeps the two
# from drifting.
_NONPUBLIC_HOSTS = frozenset(
    {
        "localhost",
        "127.0.0.1",
        "::1",
        "lemonade",
        "host.docker.internal",
        "gateway.docker.internal",
    }
)


def host_is_local(hostname: str | None) -> bool:
    """Whether a host is reachable without leaving the machine or local network.

    Loopback, LAN ranges, container gateways, mDNS names and single-label
    service names all count: those are the endpoints a keyless local embedding
    server actually answers on.
    """
    host = str(hostname or "").strip().lower().strip("[]").rstrip(".")
    if not host:
        return False
    if host in _NONPUBLIC_HOSTS:
        return True
    if host.endswith(".local") or host.endswith(".internal"):
        return True
    # Docker Compose service names are a single DNS label.
    if "." not in host:
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return bool(address.is_private or address.is_loopback or address.is_link_local)


def _split_endpoint(endpoint: str | None):
    raw = str(endpoint or "").strip()
    if not raw:
        return None
    return urlparse(raw if "://" in raw else f"http://{raw}")


def endpoint_host(endpoint: str | None) -> str | None:
    """The host of a configured embedding endpoint, or None when unreadable."""
    parsed = _split_endpoint(endpoint)
    return (parsed.hostname or None) if parsed is not None else None


def endpoint_is_local(endpoint: str | None) -> bool | None:
    """True local, False remote, None when there is no endpoint to judge."""
    host = endpoint_host(endpoint)
    if host is None:
        return None
    return host_is_local(host)


@dataclass(frozen=True)
class EmbeddingTarget:
    """What a knowledge base would embed with, and why that may not work."""

    model: str | None = None
    provider: str | None = None
    provider_mode: str | None = None
    endpoint: str | None = None
    # Why a usable configuration could not be resolved. Kept verbatim: it is
    # usually the resolver's own, more specific sentence.
    error: str | None = None

    @property
    def configured(self) -> bool:
        return self.error is None and self.endpoint is not None and self.model is not None

    @property
    def is_local(self) -> bool | None:
        return endpoint_is_local(self.endpoint)

    @property
    def requires_network(self) -> bool | None:
        local = self.is_local
        return None if local is None else not local

    @property
    def needs_api_key(self) -> bool:
        """Whether the resolver will refuse this profile without a key."""
        return bool(self.provider_mode) and self.provider_mode != "local"

    @property
    def message(self) -> str:
        """The honest verdict in one sentence; no guidance bolted on."""
        if self.endpoint is None:
            return NO_EMBEDDING_MODEL
        if self.requires_network:
            return REMOTE_ENDPOINT_NEEDS_NETWORK.format(endpoint=self.endpoint) + (
                " and its API key." if self.needs_api_key else "."
            )
        return LOCAL_ENDPOINT_NEEDS_NO_KEY.format(endpoint=self.endpoint)

    def as_dict(self) -> dict[str, Any]:
        return {
            "configured": self.configured,
            "model": self.model,
            "provider": self.provider,
            "provider_mode": self.provider_mode,
            "endpoint": self.endpoint,
            "is_local": self.is_local,
            "requires_network": self.requires_network,
            "needs_api_key": self.needs_api_key,
            "error": self.error,
            "message": self.message,
        }


def _selected_catalog(catalog: dict | None) -> dict | None:
    if catalog is not None:
        return catalog
    try:
        from deeptutor.services.rag.embedding_binding import load_catalog

        return load_catalog()
    except Exception:  # noqa: BLE001 - reporting must never fail on a config read
        return None


def _resolved_runtime(catalog: dict | None):
    try:
        from deeptutor.services.config.provider_runtime import resolve_embedding_runtime_config

        if catalog is not None:
            return resolve_embedding_runtime_config(catalog=catalog)
        return resolve_embedding_runtime_config()
    except Exception:  # noqa: BLE001 - a failure here is the thing being reported
        return None


def _resolve_config(selection: dict | None, catalog: dict | None):
    from deeptutor.services.embedding.config import get_embedding_config

    if catalog is not None:
        return get_embedding_config(selection, catalog=catalog)
    return get_embedding_config(selection)


def describe_embedding_target(
    *,
    entry: dict | None = None,
    selection: dict | None = None,
    catalog: dict | None = None,
) -> EmbeddingTarget:
    """Resolve the embedding target and classify it. Never raises.

    An explicit selection wins over entry["embedding_selection"]; with neither,
    the globally active profile is described, which is what an unbound
    knowledge base would index with.
    """
    loaded = _selected_catalog(catalog)
    if selection is None and isinstance(entry, dict):
        candidate = entry.get("embedding_selection")
        selection = candidate if isinstance(candidate, dict) else None

    try:
        config = _resolve_config(selection, loaded)
    except Exception as exc:  # noqa: BLE001 - the message is the product here
        error = str(exc) or type(exc).__name__
        # The profile may still be readable even when the key/model is not:
        # that is what separates "needs network + key" from "nothing is set".
        resolved = _resolved_runtime(loaded)
        if resolved is None:
            return EmbeddingTarget(error=error)
        return EmbeddingTarget(
            model=resolved.model,
            provider=resolved.binding,
            provider_mode=resolved.provider_mode,
            endpoint=resolved.effective_url or resolved.base_url,
            error=error,
        )

    return EmbeddingTarget(
        model=config.model,
        provider=config.binding,
        provider_mode=config.provider_mode,
        endpoint=config.effective_url or config.base_url,
    )


def explain_embedding_problem(
    problem: str | None = None,
    *,
    entry: dict | None = None,
    selection: dict | None = None,
    catalog: dict | None = None,
) -> str:
    """The resolver's own sentence, the honest verdict, and the way out.

    Used wherever a request is refused because embedding cannot be resolved,
    so every one of those refusals says how to run fully local without a key.
    """
    target = describe_embedding_target(entry=entry, selection=selection, catalog=catalog)
    parts: list[str] = []
    text = (problem or "").strip()
    if text:
        if not text.endswith((".", "!", "?")):
            text += "."
        parts.append(text)
    parts.append(target.message)
    parts.append(LOCAL_EMBEDDING_OPTION)
    seen: set[str] = set()
    unique: list[str] = []
    for part in parts:
        if part not in seen:
            seen.add(part)
            unique.append(part)
    return " ".join(unique)
