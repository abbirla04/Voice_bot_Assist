"""Exotel Voicebot applet <-> LiveKit bridge.

Exotel trial accounts have no SIP trunking, so calls reach the bot through Exotel's
Voicebot applet instead: a WebSocket carrying 16-bit PCM (8 kHz by default) as base64
`media` events. For every call this bridge

  1. joins a new LiveKit room `exotel-<CallSid>` as the caller (attribute sip.phoneNumber),
     and the room dispatches the front-desk agent the moment it is created,
  2. publishes the caller's audio into the room,
  3. mixes everything the agent plays (voice + hold music) and streams it back to Exotel,
  4. on barge-in sends Exotel `clear`, so the phone stops the bot's buffered audio at once,
  5. when the agent hangs up, lets the goodbye finish playing (Exotel `mark`) and closes
     the socket, which ends the applet and the call; when the caller hangs up, deletes
     the room.

Run on the machine that runs the agent:

    uv run python src/agent.py dev                  # terminal 1: the agent
    uv run python src/exotel_bridge.py              # terminal 2: this bridge, port 8765
    ngrok http --url=<your-static-domain> 8765      # terminal 3: public wss for Exotel

Exotel flow -> Voicebot applet URL:  wss://<your-static-domain>/exotel/stream
Optional, better speech recognition and ~100 ms faster replies: append ?sample-rate=16000
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import sys
import time
import uuid
import wave
from collections.abc import Awaitable, Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from aiohttp import WSMsgType, web  # noqa: E402
from livekit import api, rtc  # noqa: E402

from call_state import mask_phone, normalise_indian_mobile  # noqa: E402
from dashboard_feed import dashboard_enabled  # noqa: E402
from dashboard_feed import feed as dash_feed  # noqa: E402
from exotel_signals import CLEAR_TOPIC  # noqa: E402

logger = logging.getLogger("frontdesk.exotel")

MAX_INPUT_BACKLOG_S = 0.5  # never let caller audio queue up: it would delay every turn
GOODBYE_MARK = "bot-hangup"
AGENT_KIND = rtc.ParticipantKind.PARTICIPANT_KIND_AGENT


def agent_name() -> str:
    return os.getenv("AGENT_NAME", "eye-front-desk")


def exotel_chunk_bytes(sample_rate: int) -> int:
    """Exotel accepts 3,200..100,000-byte media chunks in multiples of 320 bytes. Use the
    smallest one, for the lowest latency: 200 ms at 8 kHz, 100 ms at 16 kHz."""
    size = max(3200, sample_rate // 10 * 2)  # at least 100 ms
    return size - size % 320


def caller_e164(raw: str) -> str:
    """Exotel sends numbers like 09876543210; the agent expects +919876543210."""
    return normalise_indian_mobile(raw or "") or (raw or "")


def parse_start(msg: dict) -> dict:
    """Fields of Exotel's `start` event (stream_sid may sit at the top level or inside)."""
    start = msg.get("start") or {}
    fmt = start.get("media_format") or {}
    return {
        "stream_sid": msg.get("stream_sid") or start.get("stream_sid") or "",
        "call_sid": start.get("call_sid") or uuid.uuid4().hex[:16],
        "caller": caller_e164(start.get("from", "")),
        "called": start.get("to", ""),
        "sample_rate": int(fmt.get("sample_rate") or 8000),
        "custom": start.get("custom_parameters") or {},
    }


def _digits10(number: str) -> str:
    return "".join(ch for ch in number or "" if ch.isdigit())[-10:]


def caller_token(info: dict, key: str, secret: str) -> str:
    """Access token for the caller side. Its room configuration dispatches the agent as
    soon as the room is created, so there is no separate API round trip."""
    metadata = {
        "caller_number": info["caller"],
        "channel": "exotel",
        "exotel_call_sid": info["call_sid"],
        "exotel_to": info["called"],
        "custom": info["custom"],
    }
    return (
        api.AccessToken(key, secret)
        .with_identity(f"exotel-{info['call_sid']}")
        .with_name("Caller")
        .with_attributes({"sip.phoneNumber": info["caller"], "exotel.callSid": info["call_sid"]})
        .with_grants(api.VideoGrants(room_join=True, room=f"exotel-{info['call_sid']}"))
        .with_room_config(
            api.RoomConfiguration(
                empty_timeout=30,
                departure_timeout=5,
                agents=[
                    api.RoomAgentDispatch(agent_name=agent_name(), metadata=json.dumps(metadata))
                ],
            )
        )
        .to_jwt()
    )


def media_event(stream_sid: str, pcm: bytes) -> str:
    return json.dumps(
        {
            "event": "media",
            "stream_sid": stream_sid,
            "media": {"payload": base64.b64encode(pcm).decode()},
        }
    )


def _lk_env() -> tuple[str, str, str]:
    return (
        os.environ["LIVEKIT_URL"],
        os.environ["LIVEKIT_API_KEY"],
        os.environ["LIVEKIT_API_SECRET"],
    )


async def delete_livekit_room(name: str) -> None:
    url, key, secret = _lk_env()
    async with api.LiveKitAPI(url, key, secret) as lk:
        await lk.room.delete_room(api.DeleteRoomRequest(room=name))


class OutputLevel:
    """The bot's loudness on the phone line.

    The TTS delivers speech at about -15 dBFS with peaks right at full scale (measured on
    the first calls, 18 Oct), around 10 dB hotter than a normal phone voice; on a mobile
    that sounds like shouting. Everything sent to Exotel (voice and hold music) is turned
    down by BOT_OUTPUT_GAIN_DB and a peak limiter keeps it under BOT_OUTPUT_CEILING_DB,
    so no single word can jump out louder than the rest. Gain 0 and ceiling 0 = untouched.
    """

    BLOCK_S = 0.005  # gain is updated every 5 ms
    RELEASE_S = 0.08  # after a peak the gain recovers over ~80 ms

    def __init__(self, rate: int, gain_db: float | None = None, ceiling_db: float | None = None):
        if gain_db is None:
            gain_db = float(os.getenv("BOT_OUTPUT_GAIN_DB", "-6"))
        if ceiling_db is None:
            ceiling_db = float(os.getenv("BOT_OUTPUT_CEILING_DB", "-4"))
        self.gain = 10 ** (min(gain_db, 6.0) / 20)
        self.ceiling = 32767 * 10 ** (min(ceiling_db, 0.0) / 20)
        self.bypass = gain_db == 0 and ceiling_db >= 0
        self.block = max(1, int(rate * self.BLOCK_S))
        self.release = 1 - np.exp(-self.BLOCK_S / self.RELEASE_S)  # per block
        self.env = 1.0  # the limiter's current gain (1 = not limiting)

    def process(self, pcm: bytes) -> bytes:
        if self.bypass or not pcm:
            return pcm
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) * self.gain
        out = np.empty_like(x)
        for i in range(0, len(x), self.block):
            seg = x[i : i + self.block]
            peak = float(np.abs(seg).max()) if seg.size else 0.0
            need = min(1.0, self.ceiling / peak) if peak > 0 else 1.0
            if need < self.env:  # attack at once: the loud block itself is turned down
                self.env = need
                out[i : i + self.block] = seg * need
            else:  # release smoothly back towards full level
                new = min(need, self.env + (1.0 - self.env) * self.release)
                out[i : i + self.block] = seg * np.linspace(self.env, new, seg.size)
                self.env = new
        np.clip(out, -self.ceiling, self.ceiling, out=out)
        return out.astype(np.int16).tobytes()


def _dbfs(amplitude: float) -> float:
    """16-bit sample amplitude (RMS or peak) as dBFS; silence floors at about -90."""
    return float(20 * np.log10(max(amplitude, 1.0) / 32768))


class VoiceMeter:
    """How loud the caller is, for the live dashboard: a level 5 times a second (the iris
    reacts to it) and, after each stretch of speech, its length and loudness. The hub
    compares every utterance with the caller's own normal level: a raised voice is a mood
    signal the words alone don't give."""

    EVERY_S = 0.2
    END_S = 0.7  # this much quiet ends an utterance

    def __init__(self, timing: SpeechTiming, send, clock: Callable[[], float] = time.monotonic):
        self.timing = timing  # shares its noise floor: speech = 4x the line's floor
        self.send = send
        self.clock = clock
        self._window_peak = -90.0
        self._window_speech = False
        self._sent_speech = False
        self._next = clock() + self.EVERY_S
        self._utt_start: float | None = None
        self._utt_last: float = 0.0
        self._utt_energy: list[float] = []
        self._utt_peak = 0.0

    def audio(self, pcm: bytes, rms: float | None = None) -> None:
        """rms: the chunk's loudness when SpeechTiming already measured it."""
        if len(pcm) < 2:
            return
        if rms is None:
            a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
            rms = float(np.sqrt(np.mean(a * a)))
        now = self.clock()
        db = _dbfs(rms)
        speaking = self.timing.is_speech(rms)
        self._window_peak = max(self._window_peak, db)
        self._window_speech = self._window_speech or speaking
        if speaking:
            if self._utt_start is None:
                self._utt_start, self._utt_energy, self._utt_peak = now, [], 0.0
            self._utt_last = now
            self._utt_energy.append(rms * rms)
            peak = float(np.abs(np.frombuffer(pcm, dtype=np.int16).astype(np.int32)).max())
            self._utt_peak = max(self._utt_peak, peak)
        elif self._utt_start is not None and now - self._utt_last > self.END_S:
            self._end_utterance()
        if now >= self._next:
            # while the caller is quiet, one "stopped" event is enough: no stream of silence
            if self._window_speech or self._sent_speech:
                self.send("voice", level=round(self._window_peak, 1), speaking=self._window_speech)
            self._sent_speech = self._window_speech
            self._window_peak, self._window_speech = -90.0, False
            self._next = now + self.EVERY_S

    def _end_utterance(self) -> None:
        if self._utt_start is None:
            return
        dur = self._utt_last - self._utt_start
        if dur >= 0.25 and self._utt_energy:
            self.send(
                "utterance",
                dur_s=round(dur, 2),
                mean_db=round(_dbfs(float(np.sqrt(np.mean(self._utt_energy)))), 1),
                peak_db=round(_dbfs(self._utt_peak), 1),
            )
        self._utt_start = None


class SpeechTiming:
    """TTFA as the phone line sees it, measured where the bridge hands audio to Exotel:
    caller's last loud audio -> first loud audio of the bot's reply. Includes LiveKit's
    network and Exotel chunking; the phone network adds ~0.2-0.3 s on top."""

    CALLER_RMS = 250.0  # minimum; the real threshold is 4x the line's noise floor
    BOT_PEAK = 300
    BOT_GAP_S = 0.6  # this much bot silence ends a bot reply

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.connected = clock()
        self.caller_last_loud: float | None = None
        self.caller_spoke = False  # since the bot's last reply started
        self.noise_floor = 100.0
        self.last_rms = 0.0
        self._chunks = 0
        self.bot_last_loud: float | None = None
        self.replies: list[float] = []
        self.greeting_s: float | None = None

    def is_speech(self, rms: float) -> bool:
        """Caller speech, not line noise: loud enough and well above this line's floor."""
        return rms > max(self.CALLER_RMS, 4 * self.noise_floor)

    def caller_audio(self, pcm: bytes, rate: int) -> None:
        a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        if not a.size:
            return
        rms = float(np.sqrt(np.mean(a * a)))
        self.last_rms = rms  # the voice meter reuses it
        # the line's noise floor: the quietest of the first chunks, then it falls fast and
        # rises slowly; speech barely moves it
        self._chunks += 1
        if self._chunks <= 25:  # first ~0.5 s: the quietest chunk is the floor
            self.noise_floor = rms if self._chunks == 1 else min(self.noise_floor, rms)
            k = 0.0
        elif rms < self.noise_floor:
            k = 0.1
        elif rms < 3 * self.noise_floor:
            k = 0.01
        else:
            k = 0.0005
        self.noise_floor += k * (rms - self.noise_floor)
        if self.is_speech(rms):
            self.caller_last_loud = self.clock()
            self.caller_spoke = True

    def bot_audio(self, pcm: bytes, rate: int) -> None:
        a = np.abs(np.frombuffer(pcm, dtype=np.int16))
        loud = np.nonzero(a > self.BOT_PEAK)[0]
        if not len(loud):
            return
        now = self.clock()
        heard = now + loud[0] / rate  # the chunk starts playing when Exotel gets it
        starts_reply = self.bot_last_loud is None or now - self.bot_last_loud > self.BOT_GAP_S
        self.bot_last_loud = now + len(a) / rate
        if not starts_reply:
            return
        if self.greeting_s is None:
            self.greeting_s = heard - self.connected
            logger.info(
                "TIMING greeting: first bot audio %.0f ms after the call connected",
                self.greeting_s * 1000,
            )
        elif self.caller_spoke and self.caller_last_loud is not None:
            ttfa = heard - self.caller_last_loud
            self.replies.append(ttfa)
            logger.info(
                "TIMING reply %d: TTFA %.0f ms at the phone line (caller stopped -> bot audio)%s",
                len(self.replies),
                ttfa * 1000,
                "" if ttfa <= 1.15 else "  SLOW (target 900 + ~250 phone network)",
            )
        self.caller_spoke = False

    def summary(self) -> str:
        if not self.replies:
            return f"no caller turns measured (line noise floor {self.noise_floor:.0f} RMS)"
        r = sorted(self.replies)
        return (
            f"TTFA at the line: median {r[len(r) // 2] * 1000:.0f} ms, worst {r[-1] * 1000:.0f} ms "
            f"over {len(r)} replies (noise floor {self.noise_floor:.0f} RMS)"
        )


class CallRecorder:
    """Test calls only (EXOTEL_RECORD_DIR set): what the caller said and what the bot said,
    as two WAVs, so missed or misheard speech can be listened to and re-run through every
    speech recogniser (bench/stt_compare.py). Real patients' calls are not recorded unless
    the variable is set."""

    def __init__(self, folder: str, call_sid: str, rate: int) -> None:
        base = Path(folder)
        base.mkdir(parents=True, exist_ok=True)
        stem = f"{time.strftime('%Y%m%d-%H%M%S')}-{call_sid}"
        self.paths = [base / f"{stem}-caller.wav", base / f"{stem}-bot.wav"]
        self.files = []
        for path in self.paths:
            w = wave.open(str(path), "wb")  # noqa: SIM115 - stays open for the whole call
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            self.files.append(w)

    def caller(self, pcm: bytes) -> None:
        self.files[0].writeframes(pcm)

    def bot(self, pcm: bytes) -> None:
        self.files[1].writeframes(pcm)

    def close(self) -> None:
        for w in self.files:
            with contextlib.suppress(Exception):
                w.close()


class _QuietMixerTimeouts(logging.Filter):
    """AudioMixer logs a warning whenever a track is briefly silent; that's normal here."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith("AudioMixer: stream")


class CallBridge:
    """One Exotel call <-> one LiveKit room."""

    def __init__(
        self,
        ws: web.WebSocketResponse,
        *,
        room: rtc.Room | None = None,
        delete_room: Callable[[str], Awaitable[None]] | None = None,
        agent_join_timeout_s: float | None = None,
    ) -> None:
        self.ws = ws
        self.room = room or rtc.Room()
        self.delete_room = delete_room or (lambda name: delete_livekit_room(name))
        self.agent_join_timeout_s = agent_join_timeout_s or float(
            os.getenv("EXOTEL_AGENT_JOIN_TIMEOUT_S", "15")
        )
        self.info: dict = {}
        self.stream_sid = ""
        self.sample_rate = 8000
        self.room_name = ""
        self.source: rtc.AudioSource | None = None
        self.mixer: rtc.AudioMixer | None = None
        self.pump_task: asyncio.Task | None = None
        self.feeds: list[FrameFeed] = []
        self.tasks: set[asyncio.Task] = set()
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.agent_ended = asyncio.Event()
        self.agent_joined = asyncio.Event()
        self.failed: str | None = None
        self._closing = False
        self.stats = {"in": 0, "out": 0, "dropped": 0, "clears": 0, "started": time.time()}
        self.timing = SpeechTiming()
        self.level = OutputLevel(self.sample_rate)
        self.voice: VoiceMeter | None = None
        self.recorder: CallRecorder | None = None

    def _spawn(self, coro) -> asyncio.Task:
        t = asyncio.create_task(coro)
        self.tasks.add(t)
        t.add_done_callback(self.tasks.discard)
        return t

    # -- call start ---------------------------------------------------------------------
    async def start(self, msg: dict) -> None:
        self.info = parse_start(msg)
        self.stream_sid = self.info["stream_sid"]
        self.sample_rate = self.info["sample_rate"]
        self.level = OutputLevel(self.sample_rate)
        self.room_name = f"exotel-{self.info['call_sid']}"
        room = self.room_name
        if dashboard_enabled():  # caller voice levels are for the dashboard only
            self.voice = VoiceMeter(
                self.timing, lambda kind, **d: dash_feed().send(room, kind, **d)
            )
        if os.getenv("EXOTEL_RECORD_DIR"):
            self.recorder = CallRecorder(
                os.environ["EXOTEL_RECORD_DIR"], self.info["call_sid"], self.sample_rate
            )
        logger.info(
            "call %s from %s to %s -> room %s (%d Hz)",
            self.info["call_sid"],
            mask_phone(self.info["caller"]) or "unknown",
            self.info["called"] or "?",
            self.room_name,
            self.sample_rate,
        )
        url, key, secret = _lk_env()
        self.room.on("track_subscribed", self._on_track)
        self.room.on("participant_connected", self._on_joined)
        self.room.on("participant_disconnected", self._on_left)
        self.room.on("data_received", self._on_data)
        self.room.on("disconnected", self._on_room_closed)
        await self.room.connect(url, caller_token(self.info, key, secret))

        self.source = rtc.AudioSource(self.sample_rate, 1)
        track = rtc.LocalAudioTrack.create_audio_track("caller", self.source)
        await self.room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        )
        self.mixer = make_mixer(self.sample_rate)
        self.pump_task = self._spawn(self._pump_to_exotel())
        for p in list(self.room.remote_participants.values()):
            self._on_joined(p)
        self._spawn(self._watch_agent_join())

    async def _watch_agent_join(self) -> None:
        try:
            await asyncio.wait_for(self.agent_joined.wait(), self.agent_join_timeout_s)
        except asyncio.TimeoutError:
            self.failed = (
                f"no agent joined within {self.agent_join_timeout_s:.0f}s: is "
                f"`python src/agent.py dev` running with AGENT_NAME={agent_name()}?"
            )
            logger.error(self.failed)
            self.agent_ended.set()

    # -- Exotel -> LiveKit ------------------------------------------------------------------
    async def on_media(self, msg: dict) -> None:
        if self.source is None:
            self.stats["dropped"] += 1  # still connecting: never queue stale audio
            return
        payload = (msg.get("media") or {}).get("payload")
        if not payload:
            return
        pcm = base64.b64decode(payload)
        pcm = pcm[: len(pcm) - len(pcm) % 2]
        if not pcm:
            return
        if self.source.queued_duration > MAX_INPUT_BACKLOG_S:
            self.stats["dropped"] += 1
            return
        self.stats["in"] += 1
        self.timing.caller_audio(pcm, self.sample_rate)
        if self.voice:
            self.voice.audio(pcm, rms=self.timing.last_rms)
        if self.recorder:
            self.recorder.caller(pcm)
        await self.source.capture_frame(rtc.AudioFrame(pcm, self.sample_rate, 1, len(pcm) // 2))

    # -- LiveKit -> Exotel ------------------------------------------------------------------
    def _on_joined(self, participant: rtc.RemoteParticipant) -> None:
        if participant.kind == AGENT_KIND:
            logger.info("agent %s joined %s", participant.identity, self.room_name)
            self.agent_joined.set()

    def _on_track(
        self, track: rtc.Track, pub: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ) -> None:
        if track.kind != rtc.TrackKind.KIND_AUDIO or self.mixer is None:
            return
        # every audio track goes to the caller: the agent's voice and its hold music
        self.add_agent_audio(rtc.AudioStream(track, sample_rate=self.sample_rate, num_channels=1))
        logger.info("playing %s from %s to the caller", pub.name or track.sid, participant.identity)

    def add_agent_audio(self, frames) -> None:
        assert self.mixer is not None
        feed = FrameFeed(frames)
        self.feeds.append(feed)
        self.mixer.add_stream(feed)

    async def _pump_to_exotel(self) -> None:
        assert self.mixer is not None
        async for frame in self.mixer:
            if self.ws.closed:
                break
            data = self.level.process(bytes(frame.data))
            await self._send_raw(media_event(self.stream_sid, data))
            self.stats["out"] += 1
            self.timing.bot_audio(data, self.sample_rate)
            if self.recorder:
                self.recorder.bot(data)

    def _on_data(self, packet: rtc.DataPacket) -> None:
        if packet.topic == CLEAR_TOPIC and self.stream_sid:
            self.stats["clears"] += 1
            self._spawn(self._send({"event": "clear", "stream_sid": self.stream_sid}))

    def _on_left(self, participant: rtc.RemoteParticipant) -> None:
        if participant.kind != AGENT_KIND:
            return
        others = [
            p
            for p in self.room.remote_participants.values()
            if p.kind == AGENT_KIND and p.identity != participant.identity
        ]
        if not others:
            logger.info("agent left %s", self.room_name)
            self.agent_ended.set()

    def _on_room_closed(self, *_: object) -> None:
        if not self._closing:  # the agent deleted the room (hang-up)
            self.agent_ended.set()

    async def _send(self, msg: dict) -> None:
        await self._send_raw(json.dumps(msg))

    async def _send_raw(self, text: str) -> None:
        if not self.ws.closed:
            with contextlib.suppress(ConnectionResetError, RuntimeError):
                await self.ws.send_str(text)

    # -- the Exotel socket -------------------------------------------------------------------
    async def _reader(self) -> None:
        try:
            async for msg in self.ws:
                await self.inbox.put(msg)
        finally:
            await self.inbox.put(None)

    async def run(self) -> str:
        """Read Exotel events until the call ends. Returns why it ended."""
        self._spawn(self._reader())
        ended = asyncio.create_task(self.agent_ended.wait())
        try:
            while True:
                get = asyncio.create_task(self.inbox.get())
                done, _ = await asyncio.wait({get, ended}, return_when=asyncio.FIRST_COMPLETED)
                if get not in done:
                    get.cancel()
                    if self.failed:
                        return self.failed
                    await self.flush_goodbye()
                    return "agent hung up"
                msg = get.result()
                if msg is None or msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
                    return "Exotel closed the socket"
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(msg.data)
                except ValueError:
                    continue
                event = data.get("event")
                if event == "media":
                    await self.on_media(data)
                elif event == "start":
                    # connect in the background: keep reading (and dropping) audio meanwhile
                    self._spawn(self._start_or_fail(data))
                elif event == "stop":
                    return f"caller hung up ({(data.get('stop') or {}).get('reason', 'stop')})"
                elif event == "dtmf":
                    logger.info("keypad digit %s ignored", (data.get("dtmf") or {}).get("digit"))
        finally:
            ended.cancel()

    async def _start_or_fail(self, msg: dict) -> None:
        try:
            await self.start(msg)
        except Exception as e:
            logger.exception("could not connect the call to LiveKit")
            self.failed = f"LiveKit connect failed: {e}"
            self.agent_ended.set()

    # -- call end --------------------------------------------------------------------------
    async def flush_goodbye(self, timeout: float = 3.0) -> None:
        """The agent hung up: send what is left of its goodbye, then wait until Exotel says
        it has played it (mark echo) before the socket closes and the call drops."""
        if self.mixer is not None:
            self.mixer.end_input()
        if self.pump_task is not None:
            with contextlib.suppress(Exception):  # incl. timeout: never hold the call open
                await asyncio.wait_for(asyncio.shield(self.pump_task), 1.0)
        if self.ws.closed or not self.stream_sid:
            return
        await self._send(
            {"event": "mark", "stream_sid": self.stream_sid, "mark": {"name": GOODBYE_MARK}}
        )
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._wait_for_mark(GOODBYE_MARK), timeout)

    async def _wait_for_mark(self, name: str) -> None:
        while True:
            msg = await self.inbox.get()
            if msg is None or msg.type != WSMsgType.TEXT:
                return
            with contextlib.suppress(ValueError):
                data = json.loads(msg.data)
                if data.get("event") == "stop":
                    return
                if data.get("event") == "mark" and (data.get("mark") or {}).get("name") == name:
                    return

    def _append_call_log(self, reason: str) -> None:
        """One JSON line per call: what the phone line experienced."""
        if not self.info:
            return
        row = {
            "call_sid": self.info.get("call_sid"),
            "room": self.room_name,
            "bot": "standard",
            "called": self.info.get("called"),
            "started_at": self.stats["started"],
            "duration_s": round(time.time() - self.stats["started"], 1),
            "ended": reason,
            "greeting_ms": round(self.timing.greeting_s * 1000) if self.timing.greeting_s else None,
            "replies_ms": [round(r * 1000) for r in self.timing.replies],
            "barge_in_clears": self.stats["clears"],
            "sim": str(self.info.get("call_sid", "")).startswith("sim-"),
        }
        path = Path(os.getenv("LOG_DIR", ROOT / "logs")) / "bridge-calls.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except OSError:
            logger.warning("could not write %s", path)

    async def close(self, reason: str) -> None:
        if self._closing:
            return
        self._closing = True
        s = self.stats
        logger.info(
            "call %s ended (%s): %.0fs, audio in=%d out=%d dropped=%d, barge-in clears=%d; %s",
            self.info.get("call_sid", "?"),
            reason,
            time.time() - s["started"],
            s["in"],
            s["out"],
            s["dropped"],
            s["clears"],
            self.timing.summary(),
        )
        self._append_call_log(reason)
        for t in list(self.tasks):
            t.cancel()
        if self.mixer is not None:
            with contextlib.suppress(Exception):
                await self.mixer.aclose()
        for feed in self.feeds:
            await feed.aclose()
        with contextlib.suppress(Exception):
            await self.room.disconnect()
        if self.room_name:
            try:
                await asyncio.wait_for(self.delete_room(self.room_name), 5)
            except Exception:
                logger.debug("room %s already gone", self.room_name)
        if self.recorder:
            self.recorder.close()
            logger.info("recorded %s", ", ".join(p.name for p in self.recorder.paths))
        if not self.ws.closed:
            await self.ws.close()


def make_mixer(sample_rate: int) -> rtc.AudioMixer:
    """Mixes the agent's tracks (voice + hold music) into Exotel-sized blocks. A track that
    sends nothing for half a block is skipped for that block, so a silent track can never
    slow the output below real time (the voice would then lag further and further)."""
    block = exotel_chunk_bytes(sample_rate) // 2
    return rtc.AudioMixer(
        sample_rate, 1, blocksize=block, stream_timeout_ms=max(20, block * 500 // sample_rate)
    )


class FrameFeed:
    """Cancel-safe frame source for AudioMixer. The mixer's per-frame timeout cancels the
    wait; on a plain async generator that would end the generator, and with it the agent's
    voice for the rest of the call. Here a timeout only cancels a queue read."""

    MAX_FRAMES = 100  # ~1 s of 10 ms frames; beyond that drop the oldest (never build delay)

    def __init__(self, source) -> None:
        self._q: asyncio.Queue = asyncio.Queue()
        self.dropped = 0
        self._task = asyncio.create_task(self._pull(source))

    async def _pull(self, source) -> None:
        try:
            async for item in source:
                if self._q.qsize() >= self.MAX_FRAMES:
                    self._q.get_nowait()
                    self.dropped += 1
                self._q.put_nowait(getattr(item, "frame", item))  # AudioFrameEvent or frame
        except Exception:
            logger.exception("agent audio track failed")
        finally:
            self._q.put_nowait(None)
            if hasattr(source, "aclose"):
                with contextlib.suppress(Exception):
                    await source.aclose()

    def __aiter__(self) -> FrameFeed:
        return self

    async def __anext__(self) -> rtc.AudioFrame:
        frame = await self._q.get()
        if frame is None:
            self._q.put_nowait(None)  # stay ended
            raise StopAsyncIteration
        return frame

    async def aclose(self) -> None:
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self._task


async def handle_stream(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=20, max_msg_size=4 * 1024 * 1024)
    await ws.prepare(request)
    bridge = CallBridge(ws)
    reason = "bridge error"
    try:
        reason = await bridge.run()
    except Exception:
        logger.exception("bridge crashed")
    finally:
        await bridge.close(reason)
    return ws


async def handle_assist(request: web.Request) -> web.WebSocketResponse:
    """Agent assist (Exotel Stream applet): a copy of a call a person answers."""
    import agent_assist

    return await agent_assist.serve(request, parse_start)


async def health(_: web.Request) -> web.Response:
    return web.json_response({"ok": True, "agent": agent_name(), "stream": "/exotel/stream"})


WHATSAPP_BOT_URL = os.getenv("WHATSAPP_BOT_URL", "http://127.0.0.1:8766")


async def whatsapp_proxy(request: web.Request) -> web.Response:
    """Meta's WhatsApp webhook arrives on the same ngrok link as Exotel: hand it to the
    WhatsApp bot (its own process, so chats never compete with live call audio)."""
    import aiohttp

    keep = ("content-type", "x-hub-signature-256")
    headers = {k: v for k, v in request.headers.items() if k.lower() in keep}
    try:
        async with (
            aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as http,
            http.request(
                request.method,
                WHATSAPP_BOT_URL + request.rel_url.path_qs,
                data=await request.read(),
                headers=headers,
            ) as r,
        ):
            return web.Response(
                status=r.status, body=await r.read(), content_type=r.content_type or "text/plain"
            )
    except (aiohttp.ClientError, asyncio.TimeoutError):
        logger.warning("WhatsApp webhook call, but the WhatsApp bot is not running")
        return web.Response(status=503, text="whatsapp bot not running")


# Meta publishes an app only with a privacy policy link, and delivers real WhatsApp messages
# only to a published app. Served from the bot's own public link.
PRIVACY_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Privacy notice - Demo Eye and Retina Clinic Bot (test service)</title>
<style>body{font:16px/1.55 system-ui,sans-serif;max-width:46rem;margin:2rem auto;padding:0 1rem;
color:#1d2433;background:#fff}h1{font-size:1.5rem}h2{font-size:1.1rem;margin-top:1.6rem}</style>
</head><body>
<h1>Privacy notice: Demo Eye and Retina Clinic Bot (test service)</h1>
<p>This WhatsApp and phone assistant is a test service operated by iKites (ikites.ai) for
evaluation with a small number of invited testers. Last updated 18 October 2026.</p>
<h2>What we collect</h2>
<p>Your phone or WhatsApp number, the messages you send or the words you say to the assistant,
and the details you choose to share, such as a patient's name, age, preferred centre and
preferred day or time.</p>
<h2>How we use it</h2>
<p>Only to reply to you, to note an appointment request in the test, and to check and improve
the quality of the test service. We do not sell your data or use it for advertising.</p>
<h2>Who processes it</h2>
<p>Messages pass through the providers that run the service: Meta (WhatsApp), Anthropic
(Claude, which writes the replies), and for calls LiveKit, Deepgram, ElevenLabs and Exotel.</p>
<h2>How long we keep it</h2>
<p>A WhatsApp conversation is kept for 24 hours so the assistant can follow it. Test logs
(with phone numbers masked) are kept only for the testing period and then deleted.</p>
<h2>Not medical advice</h2>
<p>The assistant does not give medical advice or diagnoses. In an emergency, go to the nearest
eye emergency or hospital emergency.</p>
<h2>Your choices</h2>
<p>To have your data deleted, or for any question, write to ashutosh@ikites.ai.</p>
</body></html>"""


async def privacy(_: web.Request) -> web.Response:
    return web.Response(text=PRIVACY_HTML, content_type="text/html", charset="utf-8")


def make_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/privacy", privacy)
    app.router.add_get("/exotel/stream", handle_stream)
    app.router.add_get("/exotel/assist", handle_assist)
    app.router.add_route("*", "/whatsapp/{tail:.*}", whatsapp_proxy)
    app.router.add_get("/health", health)
    app.router.add_get("/", health)
    return app


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env.local")
    load_dotenv(ROOT / ".env")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("livekit").addFilter(_QuietMixerTimeouts())
    import log_masking

    log_masking.install()  # no verify token in the access log
    missing = [
        k for k in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET") if not os.getenv(k)
    ]
    if missing:
        raise SystemExit(f"missing {', '.join(missing)} in .env.local")
    port = int(os.getenv("EXOTEL_BRIDGE_PORT", "8765"))
    host = os.getenv(
        "EXOTEL_BRIDGE_HOST", "127.0.0.1"
    )  # ngrok connects locally: no firewall prompt
    logger.info("Exotel bridge on http://%s:%d/exotel/stream -> agent %s", host, port, agent_name())
    web.run_app(make_app(), host=host, port=port, print=None)


if __name__ == "__main__":
    main()
