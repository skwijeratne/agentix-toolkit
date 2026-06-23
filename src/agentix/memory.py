"""Pluggable long-term / semantic memory.

The agent loop is single-conversation: it forgets everything between runs. Many
apps need **cross-session recall** — "remember the user prefers metric units",
"recall the design decision from last week". That recall is usually backed by a
vector DB + embeddings (semantic) or a search index (keyword) — *infrastructure
that belongs to your app*, not the toolkit. So agentix owns the **interface**, not
the storage.

:class:`Memory` is that interface: ``recall`` fetches records relevant to the
current request (the loop injects them as system context before the run), and
``write`` stores new ones. Implement it over your own backend — Pinecone, pgvector,
Chroma, Elasticsearch, a file — and pass it to ``Agent(memory=...)``.

:class:`InMemoryMemory` is a dependency-free default with simple keyword recall:
good for tests, demos, and small apps; swap in a semantic backend for production.

Security note: recalled records are injected as **trusted system context**. Only
write curated content to memory — never raw, unvetted tool output — or you
reopen the prompt-injection boundary the guards exist to protect.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class MemoryRecord:
    """One stored memory. ``score`` is set by a backend that ranks recall."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str | None = None
    score: float | None = None


@runtime_checkable
class Memory(Protocol):
    """Cross-session recall. Implement over any store/index you like.

    The loop calls :meth:`recall` once per ``run``/``stream`` (with the user's
    request as the query) and injects the results as system context; call
    :meth:`write` yourself, or set ``Agent(remember_exchange=True)`` to persist
    each completed exchange.
    """

    async def recall(self, query: str, *, limit: int = 5) -> list[MemoryRecord]:
        """Return up to ``limit`` records relevant to ``query`` (most relevant
        first). Relevance is the backend's call — semantic, keyword, recency."""
        ...

    async def write(self, content: str, *, metadata: dict[str, Any] | None = None) -> None:
        """Store a new memory."""
        ...


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


class InMemoryMemory:
    """A dependency-free :class:`Memory` with keyword-overlap recall.

    Ranks records by how many query words they contain (ties keep insertion
    order). Not semantic — it won't match synonyms — but enough for tests, demos,
    and small apps. Use :meth:`dump`/:meth:`load` to persist records across
    sessions (e.g. via a :class:`~agentix.store.Store`).
    """

    def __init__(self, records: Iterable[MemoryRecord] | None = None) -> None:
        self._records: list[MemoryRecord] = list(records or [])

    @property
    def records(self) -> list[MemoryRecord]:
        return list(self._records)

    async def recall(self, query: str, *, limit: int = 5) -> list[MemoryRecord]:
        q = _tokens(query)
        if not q:
            return []
        scored: list[tuple[int, MemoryRecord]] = []
        for rec in self._records:
            overlap = len(q & _tokens(rec.content))
            if overlap:
                scored.append((overlap, rec))
        scored.sort(key=lambda pair: -pair[0])  # stable: ties keep insertion order
        return [
            MemoryRecord(rec.content, dict(rec.metadata), rec.id, float(score))
            for score, rec in scored[:limit]
        ]

    async def write(self, content: str, *, metadata: dict[str, Any] | None = None) -> None:
        self._records.append(MemoryRecord(content=content, metadata=dict(metadata or {})))

    def dump(self) -> list[dict[str, Any]]:
        """Serialize records to JSON-able dicts (for persistence)."""
        return [
            {"content": r.content, "metadata": r.metadata, "id": r.id}
            for r in self._records
        ]

    @classmethod
    def load(cls, dicts: Sequence[dict[str, Any]]) -> InMemoryMemory:
        """Rebuild from :meth:`dump` output."""
        return cls(
            MemoryRecord(
                content=d["content"],
                metadata=dict(d.get("metadata") or {}),
                id=d.get("id"),
            )
            for d in dicts
        )
