"""Run persistence — a pluggable key/value store for checkpointed run state.

agentix doesn't assume a database. ``Store`` is a two-method protocol; core ships
``MemoryStore`` (a dict, the default) and ``FileStore`` (one JSON file per run).
Bring your own Redis/Postgres/S3 backend by implementing the same two methods.

The persisted ``state`` is a plain JSON-able dict (see :mod:`agentix.serde`):
``{"run_id", "steps", "tokens_used", "messages": [...]}``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Store(Protocol):
    async def save(self, run_id: str, state: dict[str, Any]) -> None: ...

    async def load(self, run_id: str) -> dict[str, Any] | None: ...


class MemoryStore:
    """In-process store backed by a dict. Default; ideal for tests."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}

    async def save(self, run_id: str, state: dict[str, Any]) -> None:
        # Round-trip through JSON so callers can't mutate stored state by ref.
        self._data[run_id] = json.loads(json.dumps(state))

    async def load(self, run_id: str) -> dict[str, Any] | None:
        state = self._data.get(run_id)
        return json.loads(json.dumps(state)) if state is not None else None


class FileStore:
    """One JSON file per run under ``path`` (created on first save).

    Writes are **atomic** (temp file + fsync + ``os.replace``) so a crash
    mid-write can't corrupt or truncate a checkpoint — a reader always sees the
    previous complete file or the new complete file, never a partial one. File
    I/O is offloaded to a worker thread so it doesn't block the event loop.

    Concurrency: safe for concurrent readers/writers in the sense that no reader
    ever sees a torn file. Two writers racing on the **same** ``run_id`` resolve
    last-writer-wins (no corruption) — but you should still keep a single writer
    per run, especially across processes (there is no cross-process lock).
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _file(self, run_id: str) -> Path:
        # Guard against path traversal / separators in the run id.
        safe = run_id.replace("/", "_").replace("\\", "_")
        return self.path / f"{safe}.json"

    async def save(self, run_id: str, state: dict[str, Any]) -> None:
        await asyncio.to_thread(self._save_sync, run_id, state)

    async def load(self, run_id: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._load_sync, run_id)

    def _save_sync(self, run_id: str, state: dict[str, Any]) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        target = self._file(run_id)
        data = json.dumps(state, indent=2)
        # Unique temp file in the same dir -> os.replace is atomic on one volume.
        fd, tmp = tempfile.mkstemp(dir=str(self.path), prefix=target.name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())  # durable before we swap it in
            os.replace(tmp, target)  # atomic rename
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    def _load_sync(self, run_id: str) -> dict[str, Any] | None:
        f = self._file(run_id)
        if not f.exists():
            return None
        data: dict[str, Any] = json.loads(f.read_text(encoding="utf-8"))
        return data
