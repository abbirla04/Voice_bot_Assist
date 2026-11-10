"""Send live events to the dashboard (dashboard_hub.py). Never in the way of a call.

    feed().send(session_id, "bot_text", text="...", id="...")

Events go into a bounded in-memory queue; one background task posts them in small
batches to http://127.0.0.1:8767/events. If the dashboard isn't running, events are
dropped (one log line, then silence) and nothing waits on it.
Off unless DASHBOARD=1 (the voice bot runs without the dashboard by default).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading
import time
import weakref
from typing import Any

logger = logging.getLogger("frontdesk.feed")
MAX_QUEUE = 2000


def dashboard_enabled() -> bool:
    return os.getenv("DASHBOARD", "0") == "1"


class Feed:
    def __init__(self, url: str | None = None) -> None:
        self.url = url or os.getenv("DASHBOARD_URL", "http://127.0.0.1:8767/events")
        self.enabled = dashboard_enabled()
        self._queue: asyncio.Queue | None = None
        self._task: asyncio.Task | None = None
        self._down_since: float | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self.sent = 0

    def send(self, session: str, type: str, **data: Any) -> None:
        if not self.enabled or not session:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no event loop (a plain script): nothing to do
        if self._loop is not loop:  # first use, or a new loop (tests, a new process)
            self._loop, self._queue, self._task = loop, asyncio.Queue(MAX_QUEUE), None
        if self._task is None or self._task.done():
            self._task = loop.create_task(self._run())
        ev = {"session": session, "type": type, "at": time.time(), **data}
        with contextlib.suppress(asyncio.QueueFull):  # dashboard far behind: drop, never grow
            self._queue.put_nowait(ev)

    async def _run(self) -> None:
        import aiohttp

        assert self._queue is not None
        timeout = aiohttp.ClientTimeout(total=2)
        async with aiohttp.ClientSession(timeout=timeout) as http:
            while True:
                batch = [await self._queue.get()]
                while not self._queue.empty() and len(batch) < 50:
                    batch.append(self._queue.get_nowait())
                if self._down_since and time.time() - self._down_since < 5:
                    continue  # dashboard was down a moment ago: drop, try again in a few seconds
                try:
                    async with http.post(self.url, json=batch) as r:
                        await r.read()
                    self.sent += len(batch)
                    if self._down_since:
                        logger.info("dashboard reachable again")
                    self._down_since = None
                except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
                    if self._down_since is None:
                        logger.info("dashboard not running (%s): live events skipped", self.url)
                    self._down_since = time.time()

    async def aclose(self) -> None:
        """Stop the sender (end of a call's process): no task left pending at shutdown.
        The next send() starts a new one."""
        task, self._task = self._task, None
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def drain(self, timeout: float = 2.0) -> None:
        """Give queued events a moment to go out (end of a call)."""
        end = time.time() + timeout
        while self._queue is not None and not self._queue.empty() and time.time() < end:
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.05)


_FEED: Feed | None = None  # outside any event loop (send() is a no-op there)
# One feed per event loop. On Windows LiveKit runs every call as a thread with its own loop
# in one process: a shared feed swapped its queue under another call's sender, and one call
# ending (DashboardTap.end -> aclose) cancelled the other call's sender task.
_FEEDS: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, Feed] = weakref.WeakKeyDictionary()
_FEEDS_LOCK = threading.Lock()


def feed() -> Feed:
    global _FEED
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        if _FEED is None:
            _FEED = Feed()
        return _FEED
    with _FEEDS_LOCK:
        f = _FEEDS.get(loop)
        if f is None:
            f = _FEEDS[loop] = Feed()
        return f
