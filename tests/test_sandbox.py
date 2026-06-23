"""SubprocessExecutor: real process isolation, limits, and the network gate.

These spawn real subprocesses (POSIX). Functional tests pass a non-empty
`network_allowlist` (the "network allowed" path) so they don't depend on whether
this host actually permits network namespaces; the deny / fail-closed logic is
exercised by monkeypatching the isolator probe.
"""

from __future__ import annotations

import sys
import time

import agentix.sandbox as sb
from agentix import Command, SandboxPolicy, SubprocessExecutor
from agentix.types import ToolCall

PY = sys.executable
ALLOW = ["example.com"]  # non-empty => network allowed => no netns needed


def _call(name: str, **args: object) -> ToolCall:
    return ToolCall(name=name, args=args, id="c1")


def _py() -> SubprocessExecutor:
    return SubprocessExecutor({"run": Command(argv=[PY, "-"], stdin="code")})


async def test_stdout_capture_and_ok() -> None:
    ex = _py()
    res = await ex(_call("run", code="print('hello world')"), network_allowlist=ALLOW)
    assert res.ok
    assert "hello world" in res.content


async def test_nonzero_exit_is_failure_with_stderr() -> None:
    ex = _py()
    code = "import sys; sys.stderr.write('boom'); sys.exit(3)"
    res = await ex(_call("run", code=code), network_allowlist=ALLOW)
    assert not res.ok
    assert "code 3" in res.content
    assert "boom" in res.content


async def test_timeout_kills_process_quickly() -> None:
    ex = _py()
    start = time.monotonic()
    res = await ex(
        _call("run", code="import time; time.sleep(30)"),
        network_allowlist=ALLOW,
        timeout_s=0.5,
    )
    elapsed = time.monotonic() - start
    assert not res.ok
    assert "timed out" in res.content
    assert elapsed < 10  # didn't wait out the full 30s sleep


async def test_environment_is_scrubbed_by_default(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("AGENTIX_SECRET", "topsecret")
    code = "import os; print(os.environ.get('AGENTIX_SECRET', 'MISSING'))"

    scrubbed = SubprocessExecutor({"run": Command(argv=[PY, "-"], stdin="code")})
    res = await scrubbed(_call("run", code=code), network_allowlist=ALLOW)
    assert res.ok and res.content.strip() == "MISSING"

    passed = SubprocessExecutor(
        {"run": Command(argv=[PY, "-"], stdin="code")},
        sandbox=SandboxPolicy(env_passthrough=["AGENTIX_SECRET"]),
    )
    res2 = await passed(_call("run", code=code), network_allowlist=ALLOW)
    assert res2.ok and res2.content.strip() == "topsecret"


async def test_workdir_is_isolated_and_cleaned(tmp_path) -> None:  # type: ignore[no-untyped-def]
    ex = SubprocessExecutor(
        {"run": Command(argv=[PY, "-"], stdin="code")},
        sandbox=SandboxPolicy(workdir=tmp_path),
    )
    code = "open('out.txt', 'w').write('x'); import os; print(os.getcwd())"
    res = await ex(_call("run", code=code), network_allowlist=ALLOW)
    assert res.ok
    assert res.content.strip() != str(tmp_path)  # ran in a fresh subdir
    assert list(tmp_path.iterdir()) == []  # the subdir was removed afterwards


async def test_file_size_rlimit_is_enforced() -> None:
    ex = SubprocessExecutor(
        {"run": Command(argv=[PY, "-"], stdin="code")},
        sandbox=SandboxPolicy(file_size_bytes=10),
    )
    code = "open('big', 'w').write('x' * 100_000)"
    res = await ex(_call("run", code=code), network_allowlist=ALLOW)
    assert not res.ok  # writing past the 10-byte cap is killed/errored


async def test_output_is_capped() -> None:
    ex = SubprocessExecutor(
        {"run": Command(argv=[PY, "-"], stdin="code")},
        sandbox=SandboxPolicy(max_output_bytes=100),
    )
    res = await ex(_call("run", code="print('A' * 10_000)"), network_allowlist=ALLOW)
    assert res.ok
    assert len(res.content) <= 100


async def test_stdin_feeding() -> None:
    ex = SubprocessExecutor({"cat": Command(argv=["cat"], stdin="data")})
    res = await ex(_call("cat", data="piped text"), network_allowlist=ALLOW)
    assert res.ok and res.content.strip() == "piped text"


async def test_unknown_tool() -> None:
    ex = _py()
    res = await ex(_call("nope"), network_allowlist=ALLOW)
    assert not res.ok and "unknown tool" in res.content


async def test_bad_args_surface_as_error() -> None:
    ex = SubprocessExecutor({"run": Command(argv=lambda a: [PY, "-c", a["missing"]])})
    res = await ex(_call("run"), network_allowlist=ALLOW)  # no 'missing' arg
    assert not res.ok and "ERROR building command" in res.content


# ── the network gate ──────────────────────────────────────────────────────


async def test_empty_allowlist_fails_closed_without_isolation(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(sb, "_detect_list", lambda: [])
    ex = _py()  # default policy: require_network_isolation=True
    res = await ex(_call("run", code="print(1)"), network_allowlist=[])
    assert not res.ok
    assert "network isolation" in res.content.lower()


async def test_empty_allowlist_runs_when_isolator_available(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # `env` is a harmless wrapper that just runs the command — stands in for the
    # network-namespace wrapper so the deny path is exercised end to end.
    monkeypatch.setattr(sb, "_detect_list", lambda: ["env"])
    ex = _py()
    res = await ex(_call("run", code="print('ran sandboxed')"), network_allowlist=[])
    assert res.ok and "ran sandboxed" in res.content


async def test_opt_out_of_isolation_runs_without_it(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(sb, "_detect_list", lambda: [])
    ex = SubprocessExecutor(
        {"run": Command(argv=[PY, "-"], stdin="code")},
        sandbox=SandboxPolicy(require_network_isolation=False),
    )
    res = await ex(_call("run", code="print('ok')"), network_allowlist=[])
    assert res.ok and "ok" in res.content


async def test_custom_isolator_is_used(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(sb, "_detect_list", lambda: [])  # auto-detect would fail
    ex = SubprocessExecutor(
        {"run": Command(argv=[PY, "-"], stdin="code")},
        sandbox=SandboxPolicy(isolator=["env"]),  # explicit wrapper wins
    )
    res = await ex(_call("run", code="print('via custom')"), network_allowlist=[])
    assert res.ok and "via custom" in res.content
