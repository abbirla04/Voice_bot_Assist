"""Call safety nets: caller silence, maximum call length, clean hang-up."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from livekit.agents import (
    AgentSession,
    ConversationItemAddedEvent,
    ErrorEvent,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
)

from call_state import CallState
from config import ClinicConfig
from lines import LINES, line
from notify import notify_bg
from speech import speak

_FIXED_LINES = {text for variants in LINES.values() for text in variants.values()}

logger = logging.getLogger("frontdesk.guards")


class CallGuards:
    def __init__(
        self,
        session: AgentSession[CallState],
        cfg: ClinicConfig,
        *,
        disconnect: Callable[[], Awaitable[object]],
        reprompt_s: float = 6.0,
    ) -> None:
        self.session = session
        self.state: CallState = session.userdata
        self.cfg = cfg
        self._disconnect = disconnect
        self._reprompt_s = reprompt_s
        self._silence_task: asyncio.Task | None = None
        self._duration_task: asyncio.Task | None = None
        self.hangup_reason: str | None = None
        self._llm_failures = 0
        self._failsafe_task: asyncio.Task | None = None
        self._hangup_task: asyncio.Future | None = None  # strong ref: asyncio keeps weak ones
        # the caller made speech sounds the recogniser never turned into text: they ARE there,
        # we just didn't catch it. Ask them to repeat instead of "are you there?"
        self.speech_without_text = False
        self.not_caught_count = 0
        session.on("user_state_changed", self._on_user_state)
        session.on("user_input_transcribed", self._on_transcript)
        session.on("error", self._on_error)
        session.on("conversation_item_added", self._on_item)

    # -- hang up ---------------------------------------------------------------------
    async def hang_up(self, text: str | None, reason: str) -> None:
        """Say the line (if any) and disconnect. Runs in its own task and is shielded, so a
        caller talking over the goodbye cannot cancel it half-way and leave the call open."""
        if self.state.ending:
            return
        self.state.ending = True
        self.hangup_reason = reason
        logger.info("hanging up: %s", reason)
        task = asyncio.ensure_future(self._hang_up(text))
        self._hangup_task = task
        await asyncio.shield(task)

    async def _hang_up(self, text: str | None) -> None:
        try:
            if text:
                handle = speak(self.session, text, allow_interruptions=False)
                await asyncio.wait_for(handle.wait_for_playout(), timeout=20)
        except Exception:
            logger.exception("could not play hang-up line")
        finally:
            self.session.shutdown()
            try:
                await self._disconnect()
            except Exception:
                logger.exception("could not disconnect the call")

    # -- silence -----------------------------------------------------------------------
    def _paused(self) -> bool:
        s = self.state
        return s.on_hold or s.transferred or s.ending or s.awaiting_answer

    def _on_transcript(self, ev: UserInputTranscribedEvent) -> None:
        if ev.is_final and ev.transcript.strip():
            self.speech_without_text = False
            self.not_caught_count = 0
            self.state.silence_prompts = 0  # they really said something: start over

    def _on_user_state(self, ev: UserStateChangedEvent) -> None:
        if ev.new_state == "speaking":
            self.speech_without_text = True
        if ev.new_state == "away":
            if self._paused():
                # LiveKit only re-arms the away timer from "listening": without this, a caller
                # who went quiet during a hold would never get a silence prompt afterwards
                self._rearm_away()
                return
            if self._silence_task is None or self._silence_task.done():
                self._silence_task = asyncio.create_task(self._silence_flow())
        elif (
            ev.new_state == "speaking"
            and not self.state.ending
            and self._silence_task
            and not self._silence_task.done()
        ):
            # a sound is not an answer: the count resets only when words come through
            # (_on_transcript), so the same "am I audible?" is never said twice in a row
            self._silence_task.cancel()

    def _rearm_away(self) -> None:
        reset = getattr(self.session, "reset_away_timer", None)
        if reset is not None:
            reset()

    async def _silence_flow(self) -> None:
        if self.speech_without_text and not self._paused():
            # not silence: speech we failed to transcribe. Never counts towards hanging up.
            self.speech_without_text = False
            self.not_caught_count += 1
            key = ("not_caught", "not_caught_2", "not_caught_3")[min(self.not_caught_count, 3) - 1]
            logger.info(
                "caller spoke but nothing was transcribed (%d): %s", self.not_caught_count, key
            )
            speak(self.session, line(key, self.state.language))
            self._rearm_away()  # if they now stay quiet, the normal silence prompts follow
            return
        for n in range(self.state.silence_prompts, self.cfg.limits.silence_prompts_before_hangup):
            if self._paused():
                return
            self.state.silence_prompts = n + 1
            speak(self.session, line("silence_1" if n == 0 else "silence_2", self.state.language))
            await asyncio.sleep(self._reprompt_s)
        if not self._paused():
            await self.hang_up(line("silence_hangup", self.state.language), "caller silent")

    # -- provider failures (after the fallback chain is exhausted) ---------------------------
    def _on_item(self, ev: ConversationItemAddedEvent) -> None:
        # a real model reply means the LLM recovered; our own fixed lines do not count
        if (
            getattr(ev.item, "role", None) == "assistant"
            and ev.item.text_content not in _FIXED_LINES
        ):
            self._llm_failures = 0

    def _on_error(self, ev: ErrorEvent) -> None:
        err = ev.error
        if getattr(err, "recoverable", True) or self.state.ending or self.state.transferred:
            return
        kind = type(err).__name__
        self.state.errors.append(f"{kind}: {str(getattr(err, 'error', err))[:200]}")
        if kind == "LLMError":
            self._llm_failures += 1
            if self._llm_failures == 1:
                # every model in the chain failed this turn: never leave the caller in silence
                speak(self.session, line("llm_retry", self.state.language))
                return
            self._start_failsafe(f"{kind} x{self._llm_failures}", speak=True)
        elif kind == "TTSError":
            self._start_failsafe(kind, speak=False)  # cannot speak; make sure a human calls back
        else:  # STT or anything else unrecoverable
            self._start_failsafe(kind, speak=True)

    def _start_failsafe(self, reason: str, *, speak: bool) -> None:
        if self._failsafe_task is None or self._failsafe_task.done():
            self._failsafe_task = asyncio.create_task(self._failsafe(reason, speak))

    async def _failsafe(self, reason: str, speak: bool) -> None:
        notify_bg(
            "callback_request",
            {
                "reason": f"technical failure during call ({reason})",
                "phone": self.state.caller_number,
                "patient": self.state.patient.for_log(),
                "room": self.state.room,
            },
        )
        await self.hang_up(
            line("technical_issue", self.state.language) if speak else None, f"failsafe: {reason}"
        )

    # -- max duration ------------------------------------------------------------------
    def start_duration_limit(self, seconds: float | None = None) -> None:
        limit = seconds if seconds is not None else self.cfg.limits.max_call_minutes * 60
        self._duration_task = asyncio.create_task(self._duration_flow(limit))

    async def _duration_flow(self, limit: float) -> None:
        await asyncio.sleep(limit)
        # never cut a caller off mid-hold or mid-handover; give it up to two more minutes
        for _ in range(120):
            if not self.state.on_hold:
                break
            await asyncio.sleep(1)
        if self.state.transferred or self.state.ending:
            return
        notify_bg(
            "callback_request",
            {
                "reason": "call hit the maximum duration",
                "phone": self.state.caller_number,
                "patient": self.state.patient.for_log(),
                "room": self.state.room,
            },
        )
        await self.hang_up(line("max_duration", self.state.language), "max call duration")

    async def aclose(self) -> None:
        for t in (self._silence_task, self._duration_task):
            if t and not t.done():
                t.cancel()
        self.session.off("user_state_changed", self._on_user_state)
        self.session.off("user_input_transcribed", self._on_transcript)
        self.session.off("error", self._on_error)
        self.session.off("conversation_item_added", self._on_item)
