"""Agent-side half of the Exotel bridge: tell the bridge when the caller cut the bot off,
so it can send Exotel a `clear` and the phone stops playing buffered audio at once.

Kept separate from exotel_bridge.py so the agent does not import the bridge server.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from typing import Any

logger = logging.getLogger("frontdesk.exotel")

CLEAR_TOPIC = "exotel.clear"


def is_interruption(old_state: str, new_state: str, user_state: str) -> bool:
    """The bot stopped speaking while the caller was speaking: that's a barge-in.
    (When the bot simply finishes a sentence the caller is not speaking, and clearing
    then would chop the last few hundred ms of the sentence.)"""
    return old_state == "speaking" and new_state != "speaking" and user_state == "speaking"


def wire_clear_signal(session: Any, room: Any, spawn: Callable[[Coroutine], Any]) -> None:
    def send(why: str) -> None:
        logger.debug("barge-in (%s): asking the bridge to clear Exotel's buffer", why)
        spawn(room.local_participant.publish_data(b"clear", reliable=True, topic=CLEAR_TOPIC))

    def on_state(ev: Any) -> None:
        if is_interruption(ev.old_state, ev.new_state, session.user_state):
            send("state")

    def on_item(ev: Any) -> None:
        item = ev.item
        if getattr(item, "role", None) == "assistant" and getattr(item, "interrupted", False):
            send("interrupted reply")  # backstop: the reply was cut off, whatever the timing

    session.on("agent_state_changed", on_state)
    session.on("conversation_item_added", on_item)
