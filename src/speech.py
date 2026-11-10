"""Speak a fixed line (greeting, silence prompt, hang-up line...) word for word.

The bot has a TTS, so fixed lines are spoken exactly as written with `session.say`.
Lines registered with `use_recordings` play from a recording in the caller-facing voice
(no TTS characters billed, first audio at once); a missing recording is spoken live and
recorded after it has played, for the next caller.
"""

from __future__ import annotations

from typing import Any

_RECORDINGS: Any = None  # the worker's GreetingCache (one voice per worker process)
_FIXED: frozenset[str] = frozenset()


def use_recordings(cache: Any, fixed_texts: list[str] | set[str]) -> None:
    """Play these exact texts from `cache` (a GreetingCache). None turns it off."""
    global _RECORDINGS, _FIXED
    _RECORDINGS = cache
    _FIXED = frozenset(fixed_texts) if cache is not None else frozenset()


def speak(
    session: Any,
    text: str,
    *,
    allow_interruptions: bool | None = None,
    add_to_chat_ctx: bool = True,
    audio: Any = None,
) -> Any:
    """Returns a SpeechHandle (wait_for_playout() works)."""
    cache, record = _RECORDINGS, False
    if audio is None and cache is not None and text in _FIXED:
        from greeting_cache import play  # lazy: livekit.rtc only when recordings are on

        frames = cache.load(text)
        if frames:
            audio = play(frames)
        else:
            record = True
    kw: dict = {}
    if allow_interruptions is not None:
        kw["allow_interruptions"] = allow_interruptions
    if audio is not None:
        kw["audio"] = audio
    handle = session.say(text, add_to_chat_ctx=add_to_chat_ctx, **kw)
    if record:
        cache.record_after(handle, text)
    return handle
