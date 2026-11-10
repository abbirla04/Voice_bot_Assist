"""Pre-recorded greetings.

The greeting is the first thing a caller hears and it is fixed text, but it is spoken on a
TTS connection that was opened a moment earlier: on the first test calls from India the
greeting's first audio took 3.3-4.3 s. So greetings are synthesised once with the caller-facing
voice, stored on disk, and played from the recording (first audio in a few ms).

    python src/agent.py warmup      # fills the cache (the phone-test starter runs it)

A missing recording is not an error: the greeting is spoken live and recorded in the
background after the call's greeting has played, for the next caller.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
import wave
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from livekit import rtc

logger = logging.getLogger("frontdesk.greetings")

ROOT = Path(__file__).resolve().parent.parent
FRAME_MS = 20


def voice_identity(tts_spec: str) -> str:
    """Recordings are only valid for the voice (and voice settings) that made them."""
    parts = [tts_spec, os.getenv("ELEVEN_VOICE_ID", ""), os.getenv("CARTESIA_VOICE_ID", "")]
    # other providers' voice choices; only when set, so existing recordings keep their key
    parts += [f"{k}={v}" for k in ("LIVEKIT_TTS_VOICE", "SARVAM_SPEAKER") if (v := os.getenv(k))]
    if tts_spec.startswith("elevenlabs/"):
        from providers import eleven_settings_tag  # lazy: providers loads the chains' plugins

        parts.append(eleven_settings_tag())
    return "|".join(parts)


_TASKS: set[asyncio.Task] = set()
_LOADED: dict[Path, tuple[int, int, int, bytes]] = {}  # path -> (mtime, rate, channels, pcm)


class GreetingCache:
    def __init__(
        self,
        voice: str,
        folder: Path | None = None,
        make_tts: Callable[[], Any] | None = None,
    ) -> None:
        self.voice = voice
        self.folder = folder or Path(os.getenv("GREETING_CACHE_DIR", ROOT / ".cache" / "greetings"))
        self.make_tts = make_tts  # for recording a missing greeting after the call's one played

    def record_after(self, handle: Any, text: str) -> None:
        if self.make_tts is None:
            return
        t = asyncio.create_task(self._record_after(handle, text))
        _TASKS.add(t)
        t.add_done_callback(_TASKS.discard)

    async def _record_after(self, handle: Any, text: str) -> None:
        try:
            await handle.wait_for_playout()  # never compete with the live greeting
            if self.path(text).exists():
                return
            tts = self.make_tts()
            try:
                _, frames = await synthesize(tts, text)
            finally:
                await tts.aclose()
            if self.save(text, frames):
                logger.info("greeting recorded for the next callers: %r", text[:50])
        except Exception:
            logger.warning("could not record the greeting", exc_info=True)

    def path(self, text: str) -> Path:
        digest = hashlib.sha1(f"{self.voice}\n{text}".encode()).hexdigest()[:16]
        return self.folder / f"{digest}.wav"

    def load(self, text: str) -> list[rtc.AudioFrame] | None:
        p = self.path(text)
        try:
            mtime = p.stat().st_mtime_ns
        except OSError:
            return None
        hit = _LOADED.get(p)
        if hit is None or hit[0] != mtime:
            # calls share the worker process (threads on Windows): read each recording once
            try:
                with wave.open(str(p), "rb") as w:
                    rate, ch = w.getframerate(), w.getnchannels()
                    pcm = w.readframes(w.getnframes())
            except (OSError, wave.Error, EOFError):
                logger.warning("unreadable greeting recording %s: ignoring it", p.name)
                return None
            hit = _LOADED[p] = (mtime, rate, ch, pcm)
        _, rate, ch, pcm = hit
        # fresh frames for every call: two calls never share frame objects
        step = rate * FRAME_MS // 1000 * 2 * ch
        frames = []
        for i in range(0, len(pcm), step):
            chunk = pcm[i : i + step]
            frames.append(rtc.AudioFrame(chunk, rate, ch, len(chunk) // (2 * ch)))
        return frames or None

    def save(self, text: str, frames: list[rtc.AudioFrame]) -> Path | None:
        if not frames:
            return None
        p = self.path(text)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with wave.open(str(tmp), "wb") as w:
            w.setnchannels(frames[0].num_channels)
            w.setsampwidth(2)
            w.setframerate(frames[0].sample_rate)
            for f in frames:
                w.writeframes(bytes(f.data))
        os.replace(tmp, p)  # never leave a half-written recording behind
        return p


async def play(frames: list[rtc.AudioFrame]) -> AsyncIterator[rtc.AudioFrame]:
    for f in frames:
        yield f


async def synthesize(tts, text: str) -> tuple[float | None, list[rtc.AudioFrame]]:
    """Synthesise on the streaming path the live agent uses. Returns (first-audio s, frames)."""
    frames: list[rtc.AudioFrame] = []
    first = None
    t0 = time.perf_counter()
    if tts.capabilities.streaming:
        async with tts.stream() as s:
            s.push_text(text)
            s.end_input()
            async for ev in s:
                if first is None:
                    first = time.perf_counter() - t0
                frames.append(ev.frame)
    else:
        async with tts.synthesize(text) as s:
            async for ev in s:
                if first is None:
                    first = time.perf_counter() - t0
                frames.append(ev.frame)
    return first, frames
