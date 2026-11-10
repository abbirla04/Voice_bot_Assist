"""Live transcript of a call a person answers, taken inside LiveKit (works in the cloud).

Vobiz brings the call into a LiveKit room over SIP; the clinic's staff member joins the same
room. This worker joins it too, only listens, and runs speech recognition on each person's
own audio track. Who is the caller and who is staff is known from the track (a SIP
participant is the caller), so no speaker guessing is needed. Each finished sentence goes to
the dashboard hub as an `assist_segment`, the same event the Exotel route sends
(agent_assist.py), so the hub, Claude's suggestions and the scorecard work unchanged.

    python src/room_transcriber.py dev       # or `start` on the server
    dispatch: agent name "eye-transcriber" (a SIP dispatch rule, or an explicit dispatch)

Nothing here publishes audio, so nothing we do can be heard on the call.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env.local")
load_dotenv(ROOT / ".env")

import aiohttp  # noqa: E402

from call_state import mask_phone  # noqa: E402
from dashboard_feed import feed as dash_feed  # noqa: E402
from telemetry import scrub  # noqa: E402

logger = logging.getLogger("frontdesk.transcriber")

TRANSCRIBER_NAME = os.getenv("TRANSCRIBER_AGENT_NAME", "eye-transcriber")
STT_MODEL = os.getenv("ASSIST_STT_MODEL", "deepgram/nova-3")
# "multi": Hindi, English and both mixed. Set e.g. ASSIST_STT_LANGUAGE=en for one language.
STT_LANGUAGE = os.getenv("ASSIST_STT_LANGUAGE", "multi")
ENDPOINTING_MS = int(os.getenv("ASSIST_ENDPOINTING_MS", "500"))
SAMPLE_RATE = 16000
# words the recogniser was unsure of are marked [?word] below this confidence (0 turns it off)
LOW_CONFIDENCE = float(os.getenv("TRANSCRIPT_LOW_CONFIDENCE", "0.6"))


def role_of(kind: Any, identity: str = "") -> str | None:
    """'patient' for the phone caller, 'agent' for a staff member, None for a bot (Clara,
    this worker) which is never transcribed here."""
    from livekit import rtc

    if kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
        return "patient"
    if kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT or identity.startswith("agent-"):
        return None
    return "agent"


def mark_doubtful(alt: Any, floor: float = LOW_CONFIDENCE) -> str:
    """The sentence, with words the recogniser was unsure of as [?word], so the person
    reading an accented caller's transcript knows what to ask them to repeat."""
    text = (getattr(alt, "text", "") or "").strip()
    words = list(getattr(alt, "words", None) or [])
    if not floor or not words:
        return text
    parts = []
    for w in words:
        conf = getattr(w, "confidence", None)
        parts.append(f"[?{w}]" if isinstance(conf, (int, float)) and 0 < conf < floor else str(w))
    marked = " ".join(parts).strip()
    return marked if "[?" in marked else text


def make_stt(http: aiohttp.ClientSession) -> Any:
    """Deepgram through LiveKit Inference (the LiveKit keys in .env.local)."""
    from livekit.agents import inference

    return inference.STT(
        model=STT_MODEL,
        language=STT_LANGUAGE,
        extra_kwargs={"endpointing": ENDPOINTING_MS},
        http_session=http,
    )


class RoomTranscript:
    """One call's room: a speech recogniser per audio track, sentences out to the dashboard."""

    def __init__(
        self, room_name: str, *, make_stt: Callable[..., Any] = make_stt, sink: Any = None
    ) -> None:
        self.sid = f"assist-{room_name}"
        self.make_stt = make_stt
        self.sink = sink
        self.http: aiohttp.ClientSession | None = None
        self.tasks: dict[str, asyncio.Task] = {}
        self.n = 0
        self.started = False
        self.ended = False

    def _send(self, kind: str, **data: Any) -> None:
        (self.sink or dash_feed()).send(self.sid, kind, **data)

    def start(self, caller: str = "") -> None:
        if self.started:
            return
        self.started = True
        self._send(
            "session_start",
            channel="assist",
            direction="inbound",
            persona="Agent",
            caller=mask_phone(caller) if caller else "",
        )

    def add_track(self, key: str, role: str, frames: Any) -> None:
        """Start listening to one participant's audio (`frames`: an async iterator of
        rtc.AudioFrame). A track that was already being transcribed is left alone."""
        if self.ended or key in self.tasks:
            return
        self.tasks[key] = asyncio.create_task(self._listen(role, frames))

    async def _listen(self, role: str, frames: Any) -> None:
        if self.http is None:
            self.http = aiohttp.ClientSession()
        try:
            stream = self.make_stt(self.http).stream()
        except Exception as e:
            logger.exception("transcriber: speech recognition could not start")
            self._send(
                "assist_error", text=f"Speech recognition could not start ({type(e).__name__})"
            )
            return
        reader = asyncio.create_task(self._read(stream, role))
        try:
            async for ev in frames:
                # a stream that just failed: the retry takes over, the call goes on
                with contextlib.suppress(Exception):
                    stream.push_frame(getattr(ev, "frame", ev))
        except Exception:
            logger.exception("transcriber: audio stopped")
        finally:
            with contextlib.suppress(Exception):
                stream.end_input()
            with contextlib.suppress(Exception, asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(reader), 5.0)  # the last words
            if not reader.done():
                reader.cancel()
            with contextlib.suppress(Exception):
                await stream.aclose()

    async def _read(self, stream: Any, role: str) -> None:
        from livekit.agents.stt import SpeechEventType

        try:
            async for ev in stream:
                if not ev.alternatives:
                    continue
                alt = ev.alternatives[0]
                text = (getattr(alt, "text", "") or "").strip()
                if not text:
                    continue
                if ev.type == SpeechEventType.INTERIM_TRANSCRIPT:
                    self._send("caption", text=scrub(text)[-200:])
                elif ev.type == SpeechEventType.FINAL_TRANSCRIPT:
                    self.n += 1
                    self._send(
                        "assist_segment",
                        id=f"s{self.n}",
                        text=scrub(mark_doubtful(alt)),
                        speaker=role,
                        role=role,
                        language=str(getattr(alt, "language", "") or ""),
                    )
        except Exception as e:
            logger.exception("transcriber: speech recognition stopped")
            self._send("assist_error", text=f"Speech recognition stopped ({type(e).__name__})")

    async def finish(self, reason: str = "call ended") -> None:
        if self.ended:
            return
        self.ended = True
        for t in list(self.tasks.values()):
            with contextlib.suppress(Exception, asyncio.TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(t, 8.0)
        if self.http is not None:
            await self.http.close()
        if self.started:
            logger.info("transcriber: %s ended (%d lines)", self.sid, self.n)
            self._send("session_end", reason=reason)
            with contextlib.suppress(Exception):
                await dash_feed().drain()


# -- the LiveKit worker ---------------------------------------------------------------------------
def build_server() -> Any:
    from livekit import rtc
    from livekit.agents import AgentServer, AutoSubscribe, JobContext

    server = AgentServer()

    @server.rtc_session(agent_name=TRANSCRIBER_NAME)
    async def entrypoint(ctx: JobContext) -> None:
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        call = RoomTranscript(ctx.room.name)
        done = asyncio.Event()

        def listen(track: rtc.Track, participant: rtc.RemoteParticipant) -> None:
            role = role_of(participant.kind, participant.identity)
            if role is None or track.kind != rtc.TrackKind.KIND_AUDIO:
                return
            if role == "patient":
                call.start(participant.attributes.get("sip.phoneNumber", ""))
            else:
                call.start()
            frames = rtc.AudioStream(track, sample_rate=SAMPLE_RATE, num_channels=1)
            call.add_track(f"{participant.identity}/{track.sid}", role, frames)

        def left(participant: rtc.RemoteParticipant) -> None:
            if role_of(participant.kind, participant.identity) == "patient":
                done.set()  # the caller hung up: the call is over

        ctx.room.on("track_subscribed", lambda track, pub, p: listen(track, p))
        ctx.room.on("participant_disconnected", left)
        ctx.room.on("disconnected", lambda *_: done.set())
        for p in ctx.room.remote_participants.values():  # already in the room before us
            for pub in p.track_publications.values():
                if pub.track is not None:
                    listen(pub.track, p)
        await done.wait()
        await call.finish()

    return server


if __name__ == "__main__":
    from livekit.agents import cli

    if len(sys.argv) > 1 and sys.argv[1] in ("dev", "start", "connect"):
        missing = [
            k for k in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET") if not os.getenv(k)
        ]
        if missing:
            raise SystemExit(f"LiveKit keys missing in .env.local: {', '.join(missing)}")
    cli.run_app(build_server())
