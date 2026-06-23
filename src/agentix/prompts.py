"""Lightweight prompt registry & versioning.

Keep named prompts under version control in-process so you can **roll back a
prompt change that regressed**. Each ``register`` adds a new version (the latest
becomes active); ``get`` returns the active (or a pinned) version; ``rollback``
re-points the active version to an earlier one.

    prompts = PromptRegistry()
    prompts.register("assistant", "You are a helpful assistant.")        # v1
    prompts.register("assistant", "You are a terse, helpful assistant.") # v2 (active)

    agent = Agent(model=..., system_prompt=prompts.get("assistant"))     # uses v2
    prompts.rollback("assistant", 1)                                     # back to v1

Pair with eval (`agentix.evals`) to compare versions, and `to_dict`/`from_dict`
to persist the registry (e.g. via a `Store`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PromptVersion:
    version: int
    template: str


@dataclass
class _Entry:
    versions: list[PromptVersion] = field(default_factory=list)
    active: int = 0


class PromptRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}

    def register(self, name: str, template: str) -> int:
        """Add a new version of ``name`` and make it active. Returns its version
        number. A no-op (identical to the current active text) returns the
        existing version without creating a duplicate."""
        entry = self._entries.setdefault(name, _Entry())
        if entry.versions and entry.versions[entry.active - 1].template == template:
            return entry.active
        version = len(entry.versions) + 1
        entry.versions.append(PromptVersion(version, template))
        entry.active = version
        return version

    def get(self, name: str, *, version: int | None = None) -> str:
        """Return the active version's text (or a specific ``version``)."""
        entry = self._require(name)
        target = entry.active if version is None else version
        for pv in entry.versions:
            if pv.version == target:
                return pv.template
        raise KeyError(f"prompt {name!r} has no version {target}")

    def render(self, name: str, /, *, version: int | None = None, **values: Any) -> str:
        """``get`` + ``str.format(**values)``. (Escape literal braces as ``{{}}``.)"""
        return self.get(name, version=version).format(**values)

    def versions(self, name: str) -> list[int]:
        return [pv.version for pv in self._require(name).versions]

    def active(self, name: str) -> int:
        return self._require(name).active

    def rollback(self, name: str, version: int) -> None:
        """Re-point the active version to an earlier (existing) one."""
        entry = self._require(name)
        if version not in (pv.version for pv in entry.versions):
            raise KeyError(f"prompt {name!r} has no version {version}")
        entry.active = version

    def names(self) -> list[str]:
        return list(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def _require(self, name: str) -> _Entry:
        if name not in self._entries:
            raise KeyError(f"no prompt registered under {name!r}")
        return self._entries[name]

    # ── persistence ───────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            name: {
                "active": e.active,
                "versions": [{"version": v.version, "template": v.template} for v in e.versions],
            }
            for name, e in self._entries.items()
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptRegistry:
        registry = cls()
        for name, e in data.items():
            entry = _Entry(
                versions=[PromptVersion(v["version"], v["template"]) for v in e["versions"]],
                active=int(e["active"]),
            )
            registry._entries[name] = entry
        return registry
