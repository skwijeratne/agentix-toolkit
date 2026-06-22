"""Defining tools.

The ``@tool`` decorator turns a plain, typed Python function into a registered
:class:`Tool` whose JSON Schema is derived from the function's type hints and
docstring. One decorated function is the single source of truth: the name, the
description, the parameter schema the model sees, *and* the executable body —
so the schema can't drift from the implementation.

    @tool
    def get_weather(city: str) -> str:
        '''Get the current weather for a city.

        Args:
            city: City name, e.g. 'Paris'.
        '''
        return f"{city}: 21C"

A :class:`ToolRegistry` collects tools, exposes their ``schemas`` for the model,
and doubles as a :class:`~agentix.executors.ToolExecutor` for the loop.
"""

from __future__ import annotations

import collections.abc as cabc
import functools
import inspect
import types as _types
import typing
from collections.abc import Iterable, Sequence
from typing import Any, Literal, Union

from .executors import LocalToolExecutor, ToolFn
from .model import ToolSchema
from .types import ToolCall, ToolResult

__all__ = ["Tool", "ToolRegistry", "tool"]


class Tool:
    """A callable wrapped with the metadata the agent loop needs.

    Stays callable, so a decorated function can still be invoked directly
    (handy in tests and for non-agent use).
    """

    def __init__(
        self,
        func: ToolFn,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
    ) -> None:
        self.func = func
        self.name = name
        self.description = description
        self.parameters = parameters
        functools.update_wrapper(self, func)  # carry __name__, __doc__, ...

    @property
    def schema(self) -> ToolSchema:
        """The provider-neutral schema dict (name, description, parameters)."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Tool(name={self.name!r})"


def tool(
    func: ToolFn | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Any:
    """Decorator that turns a function into a :class:`Tool`.

    Usable bare (``@tool``) or with overrides (``@tool(name="...", ...)``).
    The schema is generated from type hints; the description defaults to the
    first paragraph of the docstring, and per-parameter descriptions are read
    from a Google-style ``Args:`` section.
    """

    def wrap(fn: ToolFn) -> Tool:
        return _build_tool(fn, name=name, description=description)

    return wrap if func is None else wrap(func)


# ─────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────


class ToolRegistry:
    """Collects tools; provides ``schemas`` for the model and acts as the
    executor for the loop (delegating to a :class:`LocalToolExecutor`)."""

    def __init__(self, tools: Iterable[Tool | ToolFn] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        self._executor: LocalToolExecutor | None = None
        for t in tools:
            self.add(t)

    def add(self, t: Tool | ToolFn) -> Tool:
        wrapped = t if isinstance(t, Tool) else _build_tool(t, name=None, description=None)
        self._tools[wrapped.name] = wrapped
        self._executor = None  # invalidate cached executor
        return wrapped

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    @property
    def schemas(self) -> list[ToolSchema]:
        return [t.schema for t in self._tools.values()]

    def _exec(self) -> LocalToolExecutor:
        if self._executor is None:
            mapping: dict[str, ToolFn] = {n: t.func for n, t in self._tools.items()}
            self._executor = LocalToolExecutor(mapping)
        return self._executor

    async def __call__(
        self,
        call: ToolCall,
        *,
        network_allowlist: Sequence[str] = (),
        timeout_s: float = 30.0,
    ) -> ToolResult:
        return await self._exec()(
            call, network_allowlist=network_allowlist, timeout_s=timeout_s
        )


# ─────────────────────────────────────────────────────────────────────────
# Schema generation
# ─────────────────────────────────────────────────────────────────────────

_PRIMITIVES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _build_tool(
    fn: ToolFn, *, name: str | None, description: str | None
) -> Tool:
    sig = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn, include_extras=True)
    except Exception:  # noqa: BLE001 - unresolvable annotations -> treat as untyped
        hints = {}

    summary, param_docs = _parse_docstring(inspect.getdoc(fn) or "")

    properties: dict[str, Any] = {}
    required: list[str] = []

    for pname, param in sig.parameters.items():
        if pname == "self":
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        annotation = hints.get(pname, str)
        prop = _json_schema_for(annotation)
        if pname in param_docs:
            prop = {**prop, "description": param_docs[pname]}
        properties[pname] = prop
        if param.default is inspect.Parameter.empty:
            required.append(pname)

    parameters: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = required

    return Tool(
        fn,
        name=name or fn.__name__,
        description=description or summary,
        parameters=parameters,
    )


def _json_schema_for(annotation: Any) -> dict[str, Any]:
    if annotation in _PRIMITIVES:
        return {"type": _PRIMITIVES[annotation]}
    if annotation is type(None):
        return {"type": "null"}

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if origin is Literal:
        values = list(args)
        elem_type = _PRIMITIVES.get(type(values[0]), "string") if values else "string"
        return {"type": elem_type, "enum": values}

    # Optional[X] / X | None / Union[...]
    if origin is Union or origin is _types.UnionType:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _json_schema_for(non_none[0])
        return {"anyOf": [_json_schema_for(a) for a in non_none]}

    if origin in (list, set, frozenset, tuple, cabc.Sequence):
        item = _json_schema_for(args[0]) if args else {}
        return {"type": "array", "items": item}

    if origin in (dict, cabc.Mapping):
        return {"type": "object"}

    return {"type": "string"}  # conservative fallback for unknown annotations


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Return (summary, {param: description}) from a Google-style docstring."""
    if not doc:
        return "", {}

    lines = doc.splitlines()

    # Summary = text up to the first blank line.
    summary_lines: list[str] = []
    for line in lines:
        if not line.strip():
            break
        summary_lines.append(line.strip())
    summary = " ".join(summary_lines).strip()

    # Parameter docs from an Args/Arguments/Parameters section.
    param_docs: dict[str, str] = {}
    in_args = False
    closers = {"returns", "return", "raises", "yields", "examples", "example", "note", "notes"}
    for line in lines:
        s = line.strip()
        head = s.rstrip(":").lower()
        if head in {"args", "arguments", "parameters"}:
            in_args = True
            continue
        if in_args:
            if head in closers:
                break
            # match  "name: desc"  or  "name (type): desc"
            colon = s.find(":")
            if colon > 0:
                key = s[:colon].split("(")[0].strip()
                if key.isidentifier():
                    param_docs[key] = s[colon + 1 :].strip()
    return summary, param_docs
