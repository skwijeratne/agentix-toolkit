"""Run control — cooperative interruption.

Pass an :class:`Interrupt` to ``Agent.run`` / ``Agent.stream`` and call
``trigger()`` from anywhere (another task, a signal handler, a UI button) to
stop that run. The loop checks at the top of each step — a *safe boundary*,
after a complete model+tools round — so an in-flight model call or tool finishes
first, then the run ends with ``status="aborted"``, ``reason="interrupted"``.

This is per-run state, so one ``Interrupt`` controls exactly one run. For hard,
immediate cancellation, cancel the asyncio task instead.
"""

from __future__ import annotations

import asyncio


class Interrupt:
    def __init__(self) -> None:
        self._event = asyncio.Event()

    def trigger(self) -> None:
        """Request the run to stop at its next safe boundary."""
        self._event.set()

    @property
    def triggered(self) -> bool:
        return self._event.is_set()
