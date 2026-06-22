import asyncio
import tempfile
import threading
import time
from pathlib import Path

from agentix import (
    Agent,
    FileStore,
    Limiter,
    LocalToolExecutor,
    MockModel,
    ModelResponse,
    ToolCall,
    bounded_gather,
)

MAIN_THREAD = threading.get_ident()


# ── #1 sync tools run off the event loop; timeout works ──────────────────


async def test_sync_tool_runs_in_worker_thread() -> None:
    seen: dict[str, int] = {}

    def blocking(x: int) -> str:
        seen["tid"] = threading.get_ident()
        return str(x)

    ex = LocalToolExecutor({"blocking": blocking})
    result = await ex(ToolCall("blocking", {"x": 1}))
    assert result.ok is True and result.content == "1"
    assert seen["tid"] != MAIN_THREAD  # did NOT run on the loop thread


async def test_async_tool_runs_on_loop_thread() -> None:
    async def atool(x: int) -> str:
        return str(threading.get_ident())

    ex = LocalToolExecutor({"atool": atool})
    result = await ex(ToolCall("atool", {"x": 1}))
    assert result.content == str(MAIN_THREAD)  # async tools are NOT needlessly threaded


async def test_blocking_sync_tool_does_not_stall_the_loop() -> None:
    # While a blocking sync tool runs, a concurrent loop task must keep ticking.
    def blocking(x: int) -> str:
        time.sleep(0.2)
        return "done"

    ex = LocalToolExecutor({"blocking": blocking})
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        for _ in range(20):
            await asyncio.sleep(0.01)
            ticks += 1

    tool_task = asyncio.create_task(ex(ToolCall("blocking", {"x": 1})))
    await asyncio.gather(tool_task, ticker())
    # If the sync tool had blocked the loop, ticks would be ~0; offloaded -> many.
    assert ticks >= 10


async def test_timeout_fires_for_blocking_sync_tool() -> None:
    def slow(x: int) -> str:
        time.sleep(0.2)
        return "late"

    ex = LocalToolExecutor({"slow": slow})
    result = await ex(ToolCall("slow", {"x": 1}), timeout_s=0.02)
    assert result.ok is False
    assert "timed out" in result.content


# ── #2 FileStore is atomic and leaves no temp files ──────────────────────


async def test_filestore_atomic_no_temp_leftover() -> None:
    with tempfile.TemporaryDirectory() as d:
        store = FileStore(d)
        await store.save("r", {"a": 1})
        files = sorted(p.name for p in Path(d).iterdir())
        assert files == ["r.json"]  # no *.tmp left behind
        assert await store.load("r") == {"a": 1}


async def test_filestore_concurrent_saves_same_run_no_corruption() -> None:
    with tempfile.TemporaryDirectory() as d:
        store = FileStore(d)
        await asyncio.gather(*(store.save("r", {"n": n}) for n in range(25)))
        loaded = await store.load("r")
        # Last-writer-wins, but always a complete, valid document.
        assert isinstance(loaded, dict)
        assert loaded["n"] in range(25)
        assert sorted(p.suffix for p in Path(d).iterdir()) == [".json"]


# ── #3 Limiter / bounded_gather actually bound concurrency ───────────────


async def test_limiter_bounds_peak_concurrency() -> None:
    limiter = Limiter(3)
    active = 0
    peak = 0

    async def task() -> None:
        nonlocal active, peak
        async with limiter:
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1

    await asyncio.gather(*(task() for _ in range(20)))
    assert peak <= 3


async def test_bounded_gather_caps_and_preserves_order() -> None:
    active = 0
    peak = 0

    async def work(i: int) -> int:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.005)
        active -= 1
        return i

    results = await bounded_gather([work(i) for i in range(15)], limit=4)
    assert results == list(range(15))  # original order preserved
    assert peak <= 4


def test_limiter_rejects_bad_size() -> None:
    try:
        Limiter(0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


async def test_agent_with_model_limiter_runs_and_bounds_calls() -> None:
    limiter = Limiter(2)
    active = 0
    peak = 0

    class CountingModel:
        async def __call__(self, messages, *, tools=()):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return ModelResponse(text="ok")

    agents = [
        Agent(model=CountingModel(), system_prompt="sys", model_limiter=limiter)
        for _ in range(10)
    ]
    outcomes = await bounded_gather([a.run("hi") for a in agents], limit=10)
    assert all(o.answer == "ok" for o in outcomes)
    assert peak <= 2  # the shared limiter capped concurrent model calls


async def test_agent_without_limiter_still_runs() -> None:
    agent = Agent(model=MockModel([ModelResponse(text="ok")]), system_prompt="sys")
    outcome = await agent.run("hi")
    assert outcome.answer == "ok"
