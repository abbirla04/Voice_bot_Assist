"""Agent assist: a person at the clinic takes the call, and Claude suggests what to say.

The Exotel flow "Agent_assist" rings the agent (Connect applet) and, once they pick up,
sends a copy of the call audio here (Stream applet, one direction only: nothing we do can
be heard on the call). The audio is caller and agent mixed in one channel, so the speech
recogniser also tells the voices apart (Deepgram diarization). Each finished sentence goes
to the dashboard as an `assist_segment`; the hub asks Claude for the next suggestion and
shows it on the agent's screen (dashboard_hub.py, assist_brain.py).

    wss://<bridge>/exotel/assist      (the Stream applet's URL)
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
from collections.abc import Callable
from typing import Any

import aiohttp
from aiohttp import WSMsgType, web

from call_state import mask_phone
from dashboard_feed import feed as dash_feed
from telemetry import scrub

logger = logging.getLogger("frontdesk.assist")

ASSIST_STT_MODEL = os.getenv("ASSIST_STT_MODEL", "deepgram/nova-3")
ASSIST_STT_LANGUAGE = os.getenv("ASSIST_STT_LANGUAGE", "multi")  # Hindi, English and both mixed
# how long a pause ends a sentence: long enough that one question arrives as one line
ASSIST_ENDPOINTING_MS = int(os.getenv("ASSIST_ENDPOINTING_MS", "500"))


def make_assist_stt(http: aiohttp.ClientSession, diarize: bool = True) -> Any:
    """Deepgram through LiveKit Inference (the LiveKit keys in .env.local), with speakers."""
    from livekit.agents import inference

    extra: dict = {"endpointing": ASSIST_ENDPOINTING_MS}
    if diarize:
        extra["diarize"] = True
    return inference.STT(
        model=ASSIST_STT_MODEL, language=ASSIST_STT_LANGUAGE, extra_kwargs=extra, http_session=http
    )


def split_by_speaker(alt: Any) -> list[tuple[str | None, str]]:
    """One final transcript can hold both voices: cut it where the speaker changes.
    Returns [(speaker id or None, text)]."""
    words = list(getattr(alt, "words", None) or [])
    if not words or all(getattr(w, "speaker_id", None) is None for w in words):
        text = (getattr(alt, "text", "") or "").strip()
        return [(getattr(alt, "speaker_id", None), text)] if text else []
    out: list[tuple[str | None, list[str]]] = []
    for w in words:
        spk = getattr(w, "speaker_id", None)
        if out and out[-1][0] == spk:
            out[-1][1].append(str(w))
        else:
            out.append((spk, [str(w)]))
    return [(spk, " ".join(ws).strip()) for spk, ws in out if " ".join(ws).strip()]


class AssistCall:
    """One Exotel stream: audio in, sentences with speakers out to the dashboard."""

    def __init__(
        self,
        ws: web.WebSocketResponse,
        parse_start: Callable[[dict], dict],
        *,
        make_stt: Callable[..., Any] = make_assist_stt,
        sink: Any = None,
    ) -> None:
        self.ws = ws
        self.parse_start = parse_start  # the bridge's: the same start event as the voicebot
        self.make_stt = make_stt
        self.sink = sink
        self.sid = ""
        self.sample_rate = 8000
        self.stream: Any = None
        self.reader: asyncio.Task | None = None
        self.http: aiohttp.ClientSession | None = None
        self.n = 0
        self.frames = 0
        self.diarize = True
        self._closing: asyncio.Task | None = None

    def _send(self, kind: str, **data: Any) -> None:
        (self.sink or dash_feed()).send(self.sid, kind, **data)

    async def run(self) -> str:
        try:
            async for msg in self.ws:
                if msg.type != WSMsgType.TEXT:
                    if msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                        break
                    continue
                try:
                    data = json.loads(msg.data)
                except ValueError:
                    continue
                event = data.get("event")
                if event == "start":
                    await self._start(data)
                elif event == "media":
                    self._media(data)
                elif event == "stop":
                    return str((data.get("stop") or {}).get("reason") or "call ended")
            return "Exotel closed the stream"
        finally:
            await self._finish()

    async def _start(self, msg: dict) -> None:
        info = self.parse_start(msg)
        self.sid = f"assist-{info['call_sid']}"
        self.sample_rate = info["sample_rate"]
        logger.info(
            "agent assist: call %s from %s (%d Hz)",
            info["call_sid"],
            mask_phone(info["caller"]) or "unknown",
            self.sample_rate,
        )
        self._send(
            "session_start",
            channel="assist",
            direction="inbound",
            persona="Agent",
            caller=mask_phone(info["caller"]) if info["caller"] else "",
        )
        try:
            self.http = aiohttp.ClientSession()
            self.stream = self.make_stt(self.http).stream()
            self.reader = asyncio.create_task(self._read())
        except Exception as e:
            logger.exception("agent assist: speech recognition could not start")
            self._send(
                "assist_error", text=f"Speech recognition could not start ({type(e).__name__})"
            )

    def _media(self, msg: dict) -> None:
        if self.stream is None:
            return
        payload = (msg.get("media") or {}).get("payload")
        if not payload:
            return
        pcm = base64.b64decode(payload)
        pcm = pcm[: len(pcm) - len(pcm) % 2]
        if not pcm:
            return
        from livekit import rtc

        self.frames += 1
        # a stream that just failed: the retry takes over, the call goes on
        with contextlib.suppress(Exception):
            self.stream.push_frame(rtc.AudioFrame(pcm, self.sample_rate, 1, len(pcm) // 2))

    async def _read(self) -> None:
        from livekit.agents.stt import SpeechEventType

        try:
            async for ev in self.stream:
                if not ev.alternatives:
                    continue
                alt = ev.alternatives[0]
                if ev.type == SpeechEventType.INTERIM_TRANSCRIPT:
                    if (alt.text or "").strip():
                        self._send("caption", text=scrub(alt.text)[-200:])
                elif ev.type == SpeechEventType.FINAL_TRANSCRIPT:
                    for speaker, text in split_by_speaker(alt):
                        self.n += 1
                        self._send(
                            "assist_segment",
                            id=f"s{self.n}",
                            text=scrub(text),
                            speaker=None if speaker is None else str(speaker),
                            language=str(getattr(alt, "language", "") or ""),
                        )
        except Exception as e:
            if self.diarize and self.n == 0 and self.http is not None:
                # refused at the start: try again without speakers (Claude labels the lines)
                logger.warning("agent assist: no speaker labels (%s); retrying without", e)
                self.diarize = False
                failed, self.stream = self.stream, self.make_stt(self.http, diarize=False).stream()
                self._closing = asyncio.create_task(failed.aclose())
                self.reader = asyncio.create_task(self._read())
                return
            logger.exception("agent assist: speech recognition stopped")
            self._send("assist_error", text=f"Speech recognition stopped ({type(e).__name__})")

    async def _finish(self) -> None:
        if self.stream is not None:
            with contextlib.suppress(Exception):
                self.stream.end_input()
            if self.reader is not None:
                with contextlib.suppress(Exception, asyncio.TimeoutError):
                    await asyncio.wait_for(asyncio.shield(self.reader), 5.0)  # last words
            with contextlib.suppress(Exception):
                await self.stream.aclose()
        if self.reader is not None and not self.reader.done():
            self.reader.cancel()
        if self.http is not None:
            await self.http.close()
        if self.sid:
            logger.info("agent assist: call ended (%d audio chunks, %d lines)", self.frames, self.n)


async def serve(request: web.Request, parse_start: Callable[[dict], dict]) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=20, max_msg_size=4 * 1024 * 1024)
    await ws.prepare(request)
    call = AssistCall(ws, parse_start)
    reason = "error"
    try:
        reason = await call.run()
    except Exception:
        logger.exception("agent assist stream crashed")
    finally:
        if call.sid:
            call._send("session_end", reason=reason)
            with contextlib.suppress(Exception):
                await dash_feed().drain()
    return ws
