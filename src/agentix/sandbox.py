"""A sandboxing tool executor for untrusted tools and model-generated code.

:class:`~agentix.executors.LocalToolExecutor` runs tool functions in-process —
fine for *trusted* tools, but it cannot contain code you don't trust (a
code-interpreter tool, a shell command the model composed). It also can't honour
``AgentPolicy.network_allowlist``: an in-process Python call can open any socket.

:class:`SubprocessExecutor` runs each tool as a **separate OS process** and
applies the limits the loop hands it:

* **Network** — when the effective allowlist is empty, egress is *denied* by
  launching the process in a fresh network namespace (Linux ``unshare`` with an
  unprivileged user namespace, auto-detected). If isolation can't be established
  and ``require_network_isolation`` is set (the default), the call **fails
  closed** rather than running untrusted code with network access. (Per-host
  allowlisting — a *non-empty* list — is not enforced here; that needs a
  filtering proxy or firewall. A non-empty list is treated as "network allowed",
  documented as such.)
* **CPU / memory / file size / processes** — POSIX ``setrlimit`` in the child.
* **Filesystem** — each call runs in a fresh temporary working directory that is
  removed afterwards. (This is cwd isolation, not a chroot; true FS confinement
  still wants a container or mount namespace.)
* **Environment** — the child gets a *minimal* env (just ``PATH``) plus any names
  you explicitly pass through, so secrets in the parent process don't leak.
* **Timeout** — the process group is killed if it overruns ``timeout_s``.

Pair it with ``tool_schemas`` describing the tools to the model, exactly like any
other :class:`~agentix.executors.ToolExecutor`.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from .types import ToolCall, ToolResult

try:  # POSIX-only; absent on Windows
    import resource
except ImportError:  # pragma: no cover - platform guard
    resource = None  # type: ignore[assignment]

#: Builds the process ``argv`` from a tool call's args.
ArgvBuilder = Callable[[Mapping[str, Any]], Sequence[str]]


@dataclass
class Command:
    """How a tool name maps to a subprocess.

    ``argv`` is either a fixed ``argv`` list or a callable that builds one from
    the tool-call args (never passed through a shell). If ``stdin`` is set, the
    named arg's value is fed to the process on standard input — handy for a
    "run this code" tool (``argv=[python, "-"], stdin="code"``).
    """

    argv: ArgvBuilder | Sequence[str]
    stdin: str | None = None

    def build(self, args: Mapping[str, Any]) -> list[str]:
        raw = self.argv(args) if callable(self.argv) else self.argv
        return [str(a) for a in raw]


@dataclass
class SandboxPolicy:
    """Resource and isolation limits applied to every sandboxed process."""

    cpu_seconds: float | None = 5.0
    memory_bytes: int | None = 512 * 1024 * 1024
    file_size_bytes: int | None = 16 * 1024 * 1024
    max_processes: int | None = 64
    max_output_bytes: int = 64 * 1024  # cap on captured stdout+stderr

    #: Exact environment for the child. If None, a minimal env (PATH) is used
    #: plus the names listed in ``env_passthrough`` copied from the parent.
    env: Mapping[str, str] | None = None
    env_passthrough: Sequence[str] = ()

    #: Base directory for the per-call temp workdir (default: system temp).
    workdir: str | os.PathLike[str] | None = None

    #: Fail closed: if network must be denied but can't be isolated, refuse.
    require_network_isolation: bool = True
    #: Override the network-isolation wrapper (e.g. ``["firejail", "--net=none"]``
    #: or ``["bwrap", "--unshare-net", "--"]``). None = auto-detect (Linux netns).
    isolator: Sequence[str] | None = None


@lru_cache(maxsize=1)
def detect_network_isolator() -> tuple[str, ...] | None:
    """Return a working argv prefix that drops a child into an empty network
    namespace, or ``None`` if none is available. Probed once and cached.

    Uses ``unshare`` with an unprivileged user namespace, which needs no root on
    Linux hosts that allow user namespaces. The probe actually *runs* it (a bare
    binary on PATH isn't proof the kernel permits it).
    """
    candidate = ["unshare", "--user", "--map-root-user", "--net"]
    if shutil.which("unshare") is None:
        return None
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [*candidate, "true"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return tuple(candidate) if proc.returncode == 0 else None


class SubprocessExecutor:
    """A sandboxing :class:`~agentix.executors.ToolExecutor`.

    ``commands`` maps tool name -> :class:`Command` (a bare ``argv`` list or
    :class:`ArgvBuilder` is accepted as shorthand). Example::

        import sys
        from agentix.sandbox import SubprocessExecutor, Command

        executor = SubprocessExecutor(
            {"run_python": Command(argv=[sys.executable, "-"], stdin="code")}
        )
        # With AgentPolicy().network_allowlist empty (the default), run_python
        # executes with no network access — or refuses if it can't guarantee that.
    """

    def __init__(
        self,
        commands: Mapping[str, Command | ArgvBuilder | Sequence[str]],
        *,
        sandbox: SandboxPolicy | None = None,
    ) -> None:
        self._commands: dict[str, Command] = {
            name: spec if isinstance(spec, Command) else Command(argv=spec)
            for name, spec in commands.items()
        }
        self.sandbox = sandbox or SandboxPolicy()

    @property
    def names(self) -> list[str]:
        return list(self._commands)

    async def __call__(
        self,
        call: ToolCall,
        *,
        network_allowlist: Sequence[str] = (),
        timeout_s: float = 30.0,
    ) -> ToolResult:
        command = self._commands.get(call.name)
        if command is None:
            return ToolResult(call.name, f"unknown tool: {call.name}", call.id, ok=False)

        try:
            argv = command.build(call.args)
        except Exception as exc:  # noqa: BLE001 - bad args surface as data
            return ToolResult(call.name, f"ERROR building command: {exc}", call.id, ok=False)

        # Empty allowlist => deny all egress. Non-empty => network allowed
        # (per-host filtering is not enforced by this executor).
        deny_network = len(network_allowlist) == 0
        prefix: list[str] = []
        if deny_network:
            isolator = self.sandbox.isolator
            isolator = list(isolator) if isolator is not None else _detect_list()
            if not isolator:
                if self.sandbox.require_network_isolation:
                    return ToolResult(
                        call.name,
                        "REFUSED: network isolation is unavailable on this host "
                        "(no working unshare/network namespace) and the policy "
                        "requires it; refusing to run untrusted code with network "
                        "access. Set SandboxPolicy(require_network_isolation=False) "
                        "to override, or provide an `isolator`.",
                        call.id,
                        ok=False,
                    )
            else:
                prefix = isolator

        stdin_bytes: bytes | None = None
        if command.stdin is not None:
            stdin_bytes = str(call.args.get(command.stdin, "")).encode()

        workdir = tempfile.mkdtemp(dir=self.sandbox.workdir and os.fspath(self.sandbox.workdir))
        try:
            return await self._spawn(call, [*prefix, *argv], stdin_bytes, workdir, timeout_s)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    async def _spawn(
        self,
        call: ToolCall,
        argv: list[str],
        stdin_bytes: bytes | None,
        workdir: str,
        timeout_s: float,
    ) -> ToolResult:
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=subprocess.PIPE if stdin_bytes is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=workdir,
                env=self._child_env(),
                start_new_session=True,  # own process group, so we can kill it all
                preexec_fn=self._preexec if os.name == "posix" else None,
            )
        except OSError as exc:
            return ToolResult(call.name, f"ERROR launching tool: {exc}", call.id, ok=False)

        try:
            out, err = await asyncio.wait_for(proc.communicate(stdin_bytes), timeout=timeout_s)
        except asyncio.TimeoutError:
            _kill_group(proc)
            await proc.wait()
            return ToolResult(
                call.name, f"tool timed out after {timeout_s}s", call.id, ok=False
            )

        cap = self.sandbox.max_output_bytes
        stdout = (out or b"").decode(errors="replace")[:cap]
        stderr = (err or b"").decode(errors="replace")[:cap]
        if proc.returncode == 0:
            return ToolResult(call.name, stdout, call.id, ok=True)
        detail = stderr.strip() or stdout.strip() or "(no output)"
        return ToolResult(
            call.name,
            f"tool exited with code {proc.returncode}: {detail}",
            call.id,
            ok=False,
        )

    def _child_env(self) -> dict[str, str]:
        if self.sandbox.env is not None:
            return dict(self.sandbox.env)
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
        for name in self.sandbox.env_passthrough:
            if name in os.environ:
                env[name] = os.environ[name]
        return env

    def _preexec(self) -> None:  # pragma: no cover - runs in the forked child
        if resource is None:
            return
        s = self.sandbox
        if s.cpu_seconds is not None:
            cpu = int(s.cpu_seconds)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        if s.memory_bytes is not None:
            resource.setrlimit(resource.RLIMIT_AS, (s.memory_bytes, s.memory_bytes))
        if s.file_size_bytes is not None:
            resource.setrlimit(resource.RLIMIT_FSIZE, (s.file_size_bytes, s.file_size_bytes))
        if s.max_processes is not None and hasattr(resource, "RLIMIT_NPROC"):
            resource.setrlimit(resource.RLIMIT_NPROC, (s.max_processes, s.max_processes))


def _detect_list() -> list[str]:
    found = detect_network_isolator()
    return list(found) if found else []


def _kill_group(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):  # pragma: no cover
        proc.kill()
