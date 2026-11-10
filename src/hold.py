"""Hold SOP, applied automatically around any slow backend call.

- Fast lookups (< first_notice_after_s) never put the caller on hold.
- Slow ones: announce the hold once, play soft hold music, check in every
  checkin_every_s, and give up at max_hold_s (the tool then tells the LLM to
  offer a callback, exactly as the script says).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable
from typing import TypeVar

from livekit.agents import AudioConfig, BackgroundAudioPlayer, BuiltinAudioClip, RunContext

from call_state import CallState, Lang, detect_language
from config import HoldCfg
from lines import line
from speech import speak

logger = logging.getLogger("frontdesk.hold")
T = TypeVar("T")


class HoldTimeout(Exception):
    def __init__(self, reason: str, pending: asyncio.Future | None = None) -> None:
        super().__init__(reason)
        self.pending = pending  # still-running backend call, when not cancelled


class HoldOutcome:
    def __init__(self) -> None:
        self.held = False
        self.held_s = 0.0

    def note(self) -> str:
        if not self.held:
            return ""
        return f" (The caller was on hold for about {round(self.held_s)} seconds: thank them for their patience first.)"


def _caller_language(ctx: RunContext[CallState]) -> Lang:
    """Language of the caller's latest utterance (works in voice and text mode)."""
    state = ctx.userdata
    try:
        for msg in reversed(ctx.session.history.messages()):
            if msg.role == "user" and msg.text_content:
                lang = detect_language(msg.text_content)
                if lang is not None:
                    state.language = lang
                break
    except Exception:
        pass
    return state.language


class HoldController:
    def __init__(self, cfg: HoldCfg, background: BackgroundAudioPlayer | None) -> None:
        self._cfg = cfg
        self._bg = background

    async def run(
        self,
        ctx: RunContext[CallState],
        work: Awaitable[T],
        *,
        reason: str,
        cancel_on_timeout: bool = True,
    ) -> tuple[T, HoldOutcome]:
        """Await `work`, holding the caller per SOP if it is slow.

        Raises HoldTimeout after max_hold_s. Exceptions from `work` propagate.
        """
        state = ctx.userdata
        outcome = HoldOutcome()
        music = None
        started = time.monotonic()

        def filler(step: int):
            text = _filler_text(step)
            if text is None:
                return None
            # spoken but NOT added to the LLM context: a trailing assistant message makes
            # some providers (Claude) reject the next request; the tool result reports the hold
            return speak(ctx.session, text, add_to_chat_ctx=False)

        def _filler_text(step: int) -> str | None:
            nonlocal music
            lang = _caller_language(ctx)
            if step == 0:
                outcome.held = True
                state.on_hold = True
                if self._bg is not None and self._cfg.music_volume > 0:
                    try:
                        music = self._bg.play(
                            AudioConfig(BuiltinAudioClip.HOLD_MUSIC, volume=self._cfg.music_volume),
                            loop=True,
                        )
                    except Exception:  # music is nice-to-have, never fatal
                        logger.exception("could not start hold music")
                return line("hold_start", lang)
            if step == 1:
                return line("hold_checkin_1", lang)
            return line("hold_checkin_2", lang)

        task = asyncio.ensure_future(work)
        try:
            async with ctx.with_filler(
                filler,
                delay=self._cfg.first_notice_after_s,
                interval=self._cfg.checkin_every_s,
                max_steps=3,
            ):
                try:
                    result = await asyncio.wait_for(asyncio.shield(task), self._cfg.max_hold_s)
                except asyncio.TimeoutError:
                    if cancel_on_timeout:
                        task.cancel()
                        raise HoldTimeout(reason) from None
                    raise HoldTimeout(reason, pending=task) from None
            return result, outcome
        finally:
            if music is not None:
                try:
                    music.stop()
                except Exception:
                    logger.exception("could not stop hold music")
            state.on_hold = False
            outcome.held_s = time.monotonic() - started if outcome.held else 0.0
            if outcome.held:
                state.holds.append({"reason": reason, "seconds": round(outcome.held_s, 1)})
