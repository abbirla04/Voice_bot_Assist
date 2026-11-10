"""A phone call's live events for the dashboard: what the caller and the bot said, tool
calls, reply speed, barge-ins, the bot's state, and the end of the call with its cost.

Only listens to AgentSession events and hands them to dashboard_feed (non-blocking), so it
cannot change or slow what the caller hears. The bridge adds the caller's voice levels."""

from __future__ import annotations

import json
import logging
from typing import Any

from call_state import CallState, mask_phone
from dashboard_feed import Feed, feed
from front_desk import strip_turn_note
from telemetry import scrub

logger = logging.getLogger("frontdesk.tap")


def _args(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        out = json.loads(raw or "{}")
        return out if isinstance(out, dict) else {}
    except (TypeError, ValueError):
        return {}


def tool_event(name: str, arguments: Any, out: Any) -> dict:
    """One tool call as the hub's "tool" event, phone numbers masked (calls and WhatsApp)."""
    return {
        "name": name,
        "args": scrub(_args(arguments)),
        "output": scrub(str(getattr(out, "output", "") or ""))[:300],
        "error": bool(getattr(out, "is_error", False)),
    }


class DashboardTap:
    def __init__(
        self, session, state: CallState, *, channel: str = "phone", sink: Feed | None = None
    ):
        self.session = session
        self.state = state
        self.sid = state.room or "call"
        self.feed = sink or feed()
        self._last_user: dict = {}
        self.feed.send(
            self.sid,
            "session_start",
            channel=channel,
            direction=state.direction,
            persona=state.persona,
            caller=mask_phone(state.caller_number) if state.caller_number else "",
            bot=state.bot,
        )
        session.on("user_input_transcribed", self._on_transcript)
        session.on("conversation_item_added", self._on_item)
        session.on("function_tools_executed", self._on_tools)
        session.on("agent_state_changed", self._on_agent_state)

    def caller_known(self) -> None:
        """The caller ID arrives a moment after the call starts (inbound)."""
        if self.state.caller_number:
            self.feed.send(self.sid, "session_start", caller=mask_phone(self.state.caller_number))

    def _on_transcript(self, ev) -> None:
        if not ev.is_final and ev.transcript.strip():
            self.feed.send(self.sid, "caption", text=scrub(ev.transcript)[-200:])

    def _on_item(self, ev) -> None:
        item = ev.item
        role = getattr(item, "role", None)
        text = (getattr(item, "text_content", None) or "").strip()
        metrics = getattr(item, "metrics", None) or {}
        if role == "user" and text:
            self._last_user = dict(metrics)
            # the bridge's utterance + the transcript meet in the hub; strip our language note
            self.feed.send(
                self.sid,
                "user_text",
                id=item.id,
                text=scrub(strip_turn_note(text)),
                language=self.state.language,
            )
        elif role == "assistant" and text:
            latency = None
            if metrics.get("e2e_latency") is not None:
                latency = {
                    "ttfa": metrics.get("e2e_latency"),
                    "llm": metrics.get("llm_node_ttft"),
                    "tts": metrics.get("tts_node_ttfb"),
                    "eou": self._last_user.get("end_of_turn_delay"),
                }
            self.feed.send(
                self.sid,
                "bot_text",
                id=item.id,
                text=scrub(text),
                interrupted=bool(getattr(item, "interrupted", False)),
                latency=latency,
            )
            if getattr(item, "interrupted", False):
                self.feed.send(self.sid, "barge_in")

    def _on_tools(self, ev) -> None:
        for call, out in zip(ev.function_calls, ev.function_call_outputs, strict=False):
            self.feed.send(self.sid, "tool", **tool_event(call.name, call.arguments, out))

    def _on_agent_state(self, ev) -> None:
        self.feed.send(self.sid, "bot_state", state=ev.new_state)

    async def end(self, reason: str, cost_usd: float | None = None) -> None:
        self.feed.send(
            self.sid,
            "session_end",
            reason=reason,
            cost_usd=cost_usd,
            booking=(self.state.booking or {}).get("status"),
        )
        await self.feed.drain()
        await self.feed.aclose()
