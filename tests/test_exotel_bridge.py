"""Exotel Voicebot applet <-> LiveKit bridge: protocol, audio framing, barge-in, hang-up.

Everything here runs offline: the LiveKit room is faked, the Exotel side is a real
aiohttp WebSocket (end-to-end test) or a fake socket (unit tests).
"""

from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace

import numpy as np
import pytest
from aiohttp import WSMsgType
from aiohttp.test_utils import TestClient, TestServer
from livekit import api, rtc

import exotel_bridge as eb
from exotel_signals import CLEAR_TOPIC, is_interruption, wire_clear_signal

START = {
    "event": "start",
    "sequence_number": 1,
    "stream_sid": "s-123",
    "start": {
        "stream_sid": "s-123",
        "call_sid": "c-abc",
        "account_sid": "demo-account",
        "from": "09876543210",
        "to": "01140000000",
        "custom_parameters": {},
        "media_format": {"encoding": "audio/x-raw", "sample_rate": "8000", "bit_rate": "16"},
    },
}


def media(pcm: bytes) -> dict:
    return {
        "event": "media",
        "stream_sid": "s-123",
        "media": {"payload": base64.b64encode(pcm).decode()},
    }


class FakeWS:
    def __init__(self, echo_marks: bool = False) -> None:
        self._in: asyncio.Queue = asyncio.Queue()
        self.sent: list[dict] = []
        self.closed = False
        self.echo_marks = echo_marks

    def feed(self, data: dict) -> None:
        self._in.put_nowait(SimpleNamespace(type=WSMsgType.TEXT, data=json.dumps(data)))

    def __aiter__(self):
        return self

    async def __anext__(self):
        m = await self._in.get()
        if m is None:
            raise StopAsyncIteration
        return m

    async def send_str(self, s: str) -> None:
        msg = json.loads(s)
        self.sent.append(msg)
        if self.echo_marks and msg["event"] == "mark":
            self.feed({"event": "mark", "stream_sid": msg["stream_sid"], "mark": msg["mark"]})

    async def close(self) -> None:
        self.closed = True
        self._in.put_nowait(None)


class FakeSource:
    def __init__(self, backlog: float = 0.0) -> None:
        self.queued_duration = backlog
        self.frames: list[rtc.AudioFrame] = []

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        self.frames.append(frame)


class FakeRoom:
    def __init__(self) -> None:
        self.handlers: dict = {}
        self.remote_participants: dict = {}
        self.disconnected = False

    def on(self, event, cb=None):
        self.handlers[event] = cb

    async def disconnect(self) -> None:
        self.disconnected = True


def agent_participant(identity: str = "agent-1"):
    return SimpleNamespace(identity=identity, kind=rtc.ParticipantKind.PARTICIPANT_KIND_AGENT)


def make_bridge(ws, **kw):
    deleted: list[str] = []

    async def delete(name: str) -> None:
        deleted.append(name)

    b = eb.CallBridge(ws, room=FakeRoom(), delete_room=delete, **kw)
    return b, deleted


async def fake_start(self, msg):
    """Stands in for the LiveKit connect: same state the real start() leaves behind."""
    self.info = eb.parse_start(msg)
    self.stream_sid = self.info["stream_sid"]
    self.sample_rate = self.info["sample_rate"]
    self.room_name = f"exotel-{self.info['call_sid']}"
    self.source = FakeSource()
    self.mixer = eb.make_mixer(self.sample_rate)
    self.pump_task = self._spawn(self._pump_to_exotel())
    self.agent_joined.set()


async def agent_voice(sample_rate: int, ms: int):
    """10 ms frames, as a LiveKit AudioStream delivers them."""
    n = sample_rate // 100
    for i in range(ms // 10):
        yield rtc.AudioFrame(np.full(n, 1000 + i, dtype=np.int16).tobytes(), sample_rate, 1, n)
        await asyncio.sleep(0)


# -- protocol details ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rate,expected", [(8000, 3200), (16000, 3200), (24000, 4800), (48000, 9600)]
)
def test_exotel_chunk_size_is_legal_and_smallest(rate, expected):
    size = eb.exotel_chunk_bytes(rate)
    assert size == expected
    assert size % 320 == 0 and 3200 <= size <= 100_000


def test_parse_start_event():
    info = eb.parse_start(START)
    assert info["stream_sid"] == "s-123"
    assert info["call_sid"] == "c-abc"
    assert info["caller"] == "+919876543210"  # Exotel's 0-prefixed number, normalised
    assert info["sample_rate"] == 8000  # Exotel sends it as a string
    nested_only = {
        "event": "start",
        "start": {**START["start"], "media_format": {"sample_rate": "16000"}},
    }
    assert eb.parse_start(nested_only)["stream_sid"] == "s-123"
    assert eb.parse_start(nested_only)["sample_rate"] == 16000


def test_unknown_caller_id_is_kept_as_is():
    assert eb.caller_e164("") == ""
    assert eb.caller_e164("anonymous") == "anonymous"
    assert eb.caller_e164("+919876543210") == "+919876543210"


def test_caller_token_dispatches_the_agent_with_caller_id(monkeypatch):
    monkeypatch.setenv("AGENT_NAME", "eye-front-desk")
    info = eb.parse_start(START)
    token = eb.caller_token(info, "APIkey", "s" * 40)
    claims = api.TokenVerifier("APIkey", "s" * 40).verify(token)
    assert claims.identity == "exotel-c-abc"
    assert claims.video.room == "exotel-c-abc" and claims.video.room_join
    assert claims.attributes["sip.phoneNumber"] == "+919876543210"
    (dispatch,) = claims.room_config.agents
    assert dispatch.agent_name == "eye-front-desk"
    meta = json.loads(dispatch.metadata)
    assert meta["caller_number"] == "+919876543210" and meta["channel"] == "exotel"
    assert "phone_number" not in meta  # that key means OUTBOUND to the agent


# -- caller audio -> LiveKit -------------------------------------------------------------------


async def test_caller_audio_before_room_is_ready_is_dropped_not_queued():
    b, _ = make_bridge(FakeWS())
    await b.on_media(media(b"\x01\x00" * 160))
    assert b.stats["dropped"] == 1 and b.source is None


async def test_caller_audio_is_framed_and_backlog_is_bounded():
    b, _ = make_bridge(FakeWS())
    b.source = FakeSource()
    await b.on_media(media(b"\x01\x00" * 160 + b"\x07"))  # odd trailing byte
    (frame,) = b.source.frames
    assert frame.samples_per_channel == 160 and frame.sample_rate == 8000
    b.source.queued_duration = 0.8  # agent side stalled: drop rather than add delay
    await b.on_media(media(b"\x01\x00" * 160))
    assert len(b.source.frames) == 1 and b.stats["dropped"] == 1


# -- agent audio -> Exotel ---------------------------------------------------------------------


async def test_agent_audio_reaches_exotel_in_legal_chunks(monkeypatch):
    monkeypatch.setenv("BOT_OUTPUT_GAIN_DB", "0")  # level untouched: check the bytes exactly
    monkeypatch.setenv("BOT_OUTPUT_CEILING_DB", "0")
    ws = FakeWS()
    b, _ = make_bridge(ws)
    await fake_start(b, START)
    b.add_agent_audio(agent_voice(8000, 400))  # 400 ms of speech
    b.mixer.end_input()
    await asyncio.wait_for(b.pump_task, 5)
    chunks = [base64.b64decode(m["media"]["payload"]) for m in ws.sent if m["event"] == "media"]
    assert len(chunks) == 2
    assert all(len(c) == 3200 for c in chunks)  # 200 ms at 8 kHz, multiple of 320
    assert all(m["stream_sid"] == "s-123" for m in ws.sent)
    first = np.frombuffer(chunks[0], dtype=np.int16)
    assert first[0] == 1000 and first[-1] == 1019  # order and content preserved


def _dbfs(a: np.ndarray) -> float:
    a = a.astype(np.float64)
    return 20 * np.log10(np.sqrt(np.mean(a * a)) / 32768)


def _speechlike(seconds: float, rate: int = 8000) -> np.ndarray:
    """Voice-like test audio at the level the TTS really sent (~-15 dBFS, peaks at full
    scale on stressed syllables)."""
    t = np.arange(int(seconds * rate)) / rate
    syllables = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t) ** 2  # ~4 syllables a second
    x = np.sin(2 * np.pi * 210 * t) * syllables * 7000
    x[int(0.3 * rate) : int(0.33 * rate)] *= 4.6  # one shouted syllable, ~-0.3 dBFS
    return np.clip(x, -32767, 32767).astype(np.int16)


def test_bot_voice_is_turned_down_and_peaks_are_limited(monkeypatch):
    for k in ("BOT_OUTPUT_GAIN_DB", "BOT_OUTPUT_CEILING_DB"):
        monkeypatch.delenv(k, raising=False)
    x = _speechlike(1.0)
    level = eb.OutputLevel(8000)
    out = np.concatenate(  # in 20 ms frames, like the mixer sends
        [
            np.frombuffer(level.process(x[i : i + 160].tobytes()), dtype=np.int16)
            for i in range(0, len(x), 160)
        ]
    )
    assert len(out) == len(x)
    assert 20 * np.log10(np.abs(x).max() / 32768) > -1  # the input really peaks at full scale
    assert 20 * np.log10(np.abs(out).max() / 32768) <= -4.0 + 0.01  # never above the ceiling
    assert -7.5 < _dbfs(out) - _dbfs(x) < -5.5  # about 6 dB quieter overall
    assert np.corrcoef(out[:2000], x[:2000])[0, 1] > 0.99  # same sound, just quieter


def test_output_level_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("BOT_OUTPUT_GAIN_DB", "0")
    monkeypatch.setenv("BOT_OUTPUT_CEILING_DB", "0")
    pcm = _speechlike(0.1).tobytes()
    assert eb.OutputLevel(8000).process(pcm) == pcm


async def test_agent_audio_that_starts_late_still_reaches_the_caller():
    """Regression: the mixer's per-frame timeout used to end a quiet track's generator
    for good, so an agent that took a moment to speak would never be heard."""
    ws = FakeWS()
    b, _ = make_bridge(ws)
    await fake_start(b, START)

    async def late():
        await asyncio.sleep(0.4)  # several mixer timeouts go by first
        async for f in agent_voice(8000, 200):
            yield f

    b.add_agent_audio(late())
    await asyncio.sleep(0.9)
    assert [m for m in ws.sent if m["event"] == "media"]


async def test_a_silent_track_does_not_slow_the_voice():
    """Voice in real time + a hold-music track that sends nothing: output must keep up with
    real time, or the bot would lag further behind the caller with every sentence."""
    ws = FakeWS()
    b, _ = make_bridge(ws)
    await fake_start(
        b, {**START, "start": {**START["start"], "media_format": {"sample_rate": "16000"}}}
    )

    async def silent():
        await asyncio.Event().wait()
        yield  # pragma: no cover

    async def realtime_voice():
        async for f in agent_voice(16000, 600):
            yield f
            await asyncio.sleep(0.01)

    b.add_agent_audio(silent())
    b.add_agent_audio(realtime_voice())
    await asyncio.sleep(1.0)
    chunks = [m for m in ws.sent if m["event"] == "media"]
    assert len(chunks) >= 5  # 600 ms of voice = 6 chunks of 100 ms at 16 kHz
    for f in b.feeds:
        await f.aclose()


# -- barge-in ----------------------------------------------------------------------------------


async def test_barge_in_signal_sends_exotel_clear():
    ws = FakeWS()
    b, _ = make_bridge(ws)
    b.stream_sid = "s-123"
    b._on_data(SimpleNamespace(topic="lk.chat", data=b"x"))
    b._on_data(SimpleNamespace(topic=CLEAR_TOPIC, data=b"clear"))
    await asyncio.sleep(0.01)
    assert ws.sent == [{"event": "clear", "stream_sid": "s-123"}]


def test_interruption_detection():
    assert is_interruption("speaking", "listening", "speaking")
    assert is_interruption("speaking", "thinking", "speaking")
    assert not is_interruption("speaking", "listening", "listening")  # finished its sentence
    assert not is_interruption("thinking", "listening", "speaking")
    assert not is_interruption("speaking", "speaking", "speaking")


async def test_agent_sends_clear_signal_on_barge_in_only():
    handlers: dict = {}
    published: list = []
    session = SimpleNamespace(user_state="listening", on=lambda ev, cb: handlers.setdefault(ev, cb))

    async def publish_data(payload, *, reliable=True, topic=""):
        published.append(topic)

    room = SimpleNamespace(local_participant=SimpleNamespace(publish_data=publish_data))
    wire_clear_signal(session, room, asyncio.ensure_future)

    handlers["agent_state_changed"](SimpleNamespace(old_state="speaking", new_state="listening"))
    await asyncio.sleep(0)
    assert published == []  # normal end of a sentence: no clear

    session.user_state = "speaking"
    handlers["agent_state_changed"](SimpleNamespace(old_state="speaking", new_state="listening"))
    handlers["conversation_item_added"](
        SimpleNamespace(item=SimpleNamespace(role="assistant", interrupted=True))
    )
    handlers["conversation_item_added"](
        SimpleNamespace(item=SimpleNamespace(role="user", interrupted=False))
    )
    await asyncio.sleep(0)
    assert published == [CLEAR_TOPIC, CLEAR_TOPIC]


# -- call end ----------------------------------------------------------------------------------


def test_agent_leaving_ends_call_only_when_no_agent_is_left():
    b, _ = make_bridge(FakeWS())
    a1, a2 = agent_participant("agent-1"), agent_participant("agent-2")
    b.room.remote_participants = {"agent-1": a1, "agent-2": a2}
    b._on_left(a1)
    assert not b.agent_ended.is_set()  # e.g. hand-over to a second agent
    b.room.remote_participants = {"agent-2": a2}
    b._on_left(
        SimpleNamespace(identity="someone", kind=rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD)
    )
    assert not b.agent_ended.is_set()
    b.room.remote_participants = {}
    b._on_left(a2)
    assert b.agent_ended.is_set()


async def test_caller_hangup_deletes_room(monkeypatch):
    monkeypatch.setattr(eb.CallBridge, "start", fake_start)
    ws = FakeWS()
    b, deleted = make_bridge(ws)
    for ev in ({"event": "connected"}, START):
        ws.feed(ev)
    task = asyncio.create_task(b.run())
    await asyncio.sleep(0.05)
    ws.feed(media(b"\x01\x00" * 1600))
    ws.feed({"event": "dtmf", "dtmf": {"digit": "1"}})
    ws.feed({"event": "stop", "stop": {"reason": "callended"}})
    reason = await asyncio.wait_for(task, 2)
    await b.close(reason)
    assert reason == "caller hung up (callended)"
    assert len(b.source.frames) == 1
    assert deleted == ["exotel-c-abc"] and b.room.disconnected and ws.closed


async def test_agent_hangup_lets_goodbye_play_then_closes(monkeypatch):
    monkeypatch.setattr(eb.CallBridge, "start", fake_start)
    ws = FakeWS(echo_marks=True)
    b, deleted = make_bridge(ws)
    ws.feed(START)
    task = asyncio.create_task(b.run())
    await asyncio.sleep(0.05)
    b.add_agent_audio(agent_voice(8000, 200))  # the last words of the goodbye
    b._on_room_closed()  # agent deleted the room
    reason = await asyncio.wait_for(task, 3)
    await b.close(reason)
    assert reason == "agent hung up"
    kinds = [m["event"] for m in ws.sent]
    assert "media" in kinds and kinds[-1] == "mark"  # goodbye sent, then the mark
    assert kinds.index("media") < kinds.index("mark")
    assert deleted == ["exotel-c-abc"] and ws.closed


async def test_goodbye_wait_is_bounded_without_mark_echo(monkeypatch):
    monkeypatch.setattr(eb.CallBridge, "start", fake_start)
    ws = FakeWS(echo_marks=False)
    b, _ = make_bridge(ws)
    await fake_start(b, START)
    await asyncio.wait_for(b.flush_goodbye(timeout=0.1), 2)
    assert ws.sent[-1]["event"] == "mark"


async def test_no_agent_ends_call_with_a_clear_reason():
    ws = FakeWS()
    b, _ = make_bridge(ws, agent_join_timeout_s=0.05)
    ws.feed(START)
    b.start = lambda msg: b._watch_agent_join()  # LiveKit "connected", agent never comes
    reason = await asyncio.wait_for(b.run(), 2)
    assert "no agent joined" in reason and "agent.py dev" in reason


async def test_livekit_failure_ends_call_instead_of_dead_air(monkeypatch):
    async def boom(self, msg):
        raise ConnectionError("livekit unreachable")

    monkeypatch.setattr(eb.CallBridge, "start", boom)
    ws = FakeWS()
    b, _ = make_bridge(ws)
    ws.feed(START)
    reason = await asyncio.wait_for(b.run(), 2)
    assert reason.startswith("LiveKit connect failed")


# -- the real WebSocket endpoint -----------------------------------------------------------


async def test_websocket_endpoint_end_to_end(monkeypatch):
    """A real aiohttp socket on /exotel/stream, speaking Exotel's protocol."""
    deleted: list[str] = []

    async def delete(name: str) -> None:
        deleted.append(name)

    async def start_and_speak(self, msg):
        await fake_start(self, msg)
        self.add_agent_audio(agent_voice(self.sample_rate, 200))  # agent greets

    monkeypatch.setattr(eb.CallBridge, "start", start_and_speak)
    monkeypatch.setattr(eb, "delete_livekit_room", delete)
    monkeypatch.setattr(eb.rtc, "Room", FakeRoom)

    client = TestClient(TestServer(eb.make_app()))
    await client.start_server()
    try:
        health = await client.get("/health")
        assert (await health.json())["ok"] is True
        ws = await client.ws_connect("/exotel/stream")
        await ws.send_str(json.dumps({"event": "connected"}))
        await ws.send_str(json.dumps(START))
        got = await asyncio.wait_for(ws.receive(), 3)
        msg = json.loads(got.data)
        assert msg["event"] == "media" and msg["stream_sid"] == "s-123"
        assert len(base64.b64decode(msg["media"]["payload"])) == 3200
        await ws.send_str(json.dumps(media(b"\x01\x00" * 1600)))
        await ws.send_str(json.dumps({"event": "stop", "stop": {"reason": "callended"}}))
        end = await asyncio.wait_for(ws.receive(), 3)
        assert end.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.CLOSING)
        await ws.close()
        await asyncio.sleep(0.05)
        assert deleted == ["exotel-c-abc"]
    finally:
        await client.close()


async def test_test_calls_are_recorded_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setattr(eb.CallBridge, "start", fake_start)
    monkeypatch.setenv("EXOTEL_RECORD_DIR", str(tmp_path / "audio"))
    ws = FakeWS()
    b, _ = make_bridge(ws)
    b.recorder = eb.CallRecorder(str(tmp_path / "audio"), "c-abc", 8000)
    await fake_start(b, START)
    await b.on_media(media(b"\x01\x00" * 1600))
    b.add_agent_audio(agent_voice(8000, 200))
    await asyncio.sleep(0.4)
    await b.close("test")
    import wave

    caller, bot = b.recorder.paths
    with wave.open(str(caller)) as w:
        assert w.getframerate() == 8000 and w.getnframes() == 1600
    with wave.open(str(bot)) as w:
        assert w.getnframes() >= 1600


def test_calls_are_not_recorded_by_default(monkeypatch):
    monkeypatch.delenv("EXOTEL_RECORD_DIR", raising=False)
    b, _ = make_bridge(FakeWS())
    assert b.recorder is None


async def test_privacy_notice_is_served_for_meta():
    async with TestClient(TestServer(eb.make_app())) as c:
        r = await c.get("/privacy")
        assert r.status == 200 and r.content_type == "text/html"
        text = await r.text()
        assert "iKites" in text and "ashutosh@ikites.ai" in text and "24 hours" in text
