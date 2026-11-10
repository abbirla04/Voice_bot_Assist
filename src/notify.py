"""Outbound notifications: SMS confirmations, callback requests, emergency alerts.

If NOTIFY_WEBHOOK_URL is set, each event is POSTed there as JSON (point it at
n8n / your backend, which sends the DLT-approved SMS via Exotel). Every event is
also appended to logs/notifications.jsonl. A notification failure never breaks
the call: it is logged and reported back as "not sent".
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path

import aiohttp

logger = logging.getLogger("frontdesk.notify")
LOG_PATH = Path(os.getenv("LOG_DIR", "logs")) / "notifications.jsonl"


_PHONE_KEYS = ("to", "phone")
_BACKGROUND: set[asyncio.Task] = set()


def _masked(event: dict) -> dict:
    """Local log copy: phone numbers masked (the webhook still gets the full number)."""
    from call_state import mask_phone

    return {
        k: (mask_phone(v) if k in _PHONE_KEYS and isinstance(v, str) else v)
        for k, v in event.items()
    }


def notify_bg(kind: str, payload: dict) -> None:
    """Fire-and-forget: never make the caller wait on a webhook."""
    try:
        task = asyncio.get_running_loop().create_task(notify(kind, payload))
    except RuntimeError:  # no loop (sync context): drop to a blocking local write only
        _write_local({"kind": kind, "at": time.time(), **payload})
        return
    _BACKGROUND.add(task)
    task.add_done_callback(_BACKGROUND.discard)


def _write_local(event: dict) -> None:
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_masked(event), ensure_ascii=False, default=str) + "\n")
    except OSError:
        logger.exception("could not write notification log")


async def notify(kind: str, payload: dict, *, timeout: float = 5.0) -> bool:
    event = {"kind": kind, "at": time.time(), **payload}
    _write_local(event)
    url = os.getenv("NOTIFY_WEBHOOK_URL")
    if not url:
        logger.info("notification %s recorded locally (no NOTIFY_WEBHOOK_URL)", kind)
        return False
    headers = {}
    if token := os.getenv("NOTIFY_WEBHOOK_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    try:
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with (
            aiohttp.ClientSession(timeout=client_timeout) as s,
            s.post(url, json=event, headers=headers) as r,
        ):
            ok = 200 <= r.status < 300
            if not ok:
                logger.warning("notification %s got HTTP %s", kind, r.status)
            return ok
    except (aiohttp.ClientError, asyncio.TimeoutError):
        logger.exception("notification %s failed", kind)
        return False
