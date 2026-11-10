"""Latency visibility and the recorded greeting: what the logs say, and that they are kept."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
import wave
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from fake_llm import ScriptedLLM
from livekit import rtc
from livekit.agents import AgentSession, metrics, tts
from livekit.agents.testing import fake_job_context

import providers
from call_state import CallState
from exotel_bridge import SpeechTiming
from front_desk import FrontDeskAgent
from greeting_cache import GreetingCache, voice_identity
from scheduling import MockScheduler
from telemetry import CallTelemetry

ROOT = Path(__file__).resolve().parent.parent
IST = ZoneInfo("Asia/Kolkata")


@pytest.fixture
def frozen(cfg, monkeypatch):
    monkeypatch.setattr(type(cfg), "now", lambda self: datetime(2026, 10, 19, 11, 0, tzinfo=IST))
    return cfg


def frames(seconds: float, rate: int = 24000, level: int = 3000) -> list[rtc.AudioFrame]:
    n = rate // 50
    pcm = (np.sin(np.arange(n) / 5) * level).astype(np.int16).tobytes()
    return [rtc.AudioFrame(pcm, rate, 1, n) for _ in range(int(seconds * 50))]


# -- recorded greeting -------------------------------------------------------------------------


def test_greeting_recording_roundtrip(tmp_path):
    cache = GreetingCache("elevenlabs/x|voice1", folder=tmp_path)
    assert cache.load("Hello") is None
    cache.save("Hello", frames(1.0))
    back = cache.load("Hello")
    assert back and sum(f.samples_per_channel for f in back) == 24000
    assert back[0].sample_rate == 24000
    # another voice or another text never gets this recording
    assert GreetingCache("elevenlabs/x|voice2", folder=tmp_path).load("Hello") is None
    assert cache.load("Hello!") is None


def test_broken_recording_is_ignored(tmp_path):
    cache = GreetingCache("v", folder=tmp_path)
    cache.path("Hi").parent.mkdir(parents=True, exist_ok=True)
    cache.path("Hi").write_bytes(b"not a wav")
    assert cache.load("Hi") is None


def test_voice_identity_follows_the_voice(monkeypatch):
    monkeypatch.setenv("ELEVEN_VOICE_ID", "abc")
    a = voice_identity("elevenlabs/eleven_flash_v2_5")
    monkeypatch.setenv("ELEVEN_VOICE_ID", "xyz")
    assert voice_identity("elevenlabs/eleven_flash_v2_5") != a


def test_voice_identity_follows_other_providers_voices(monkeypatch):
    monkeypatch.delenv("SARVAM_SPEAKER", raising=False)
    a = voice_identity("sarvam/bulbul:v2")
    monkeypatch.setenv("SARVAM_SPEAKER", "anushka")
    assert voice_identity("sarvam/bulbul:v2") != a


def test_voice_identity_follows_the_voice_settings(monkeypatch):
    """Changing how the voice speaks must re-record the greetings."""
    for k in ("ELEVEN_STABILITY", "ELEVEN_SPEED", "ELEVEN_VOICE_SETTINGS"):
        monkeypatch.delenv(k, raising=False)
    calm = voice_identity("elevenlabs/eleven_flash_v2_5")
    assert "st0.75" in calm and "sp0.95" in calm and "b0" in calm
    monkeypatch.setenv("ELEVEN_SPEED", "1.0")
    assert voice_identity("elevenlabs/eleven_flash_v2_5") != calm
    monkeypatch.setenv("ELEVEN_VOICE_SETTINGS", "voice")
    assert voice_identity("elevenlabs/eleven_flash_v2_5").endswith("voice-default")


def test_elevenlabs_gets_calm_voice_settings(monkeypatch):
    import providers

    for k in providers.ELEVEN_CALM:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv("ELEVEN_VOICE_SETTINGS", raising=False)
    monkeypatch.setenv("ELEVEN_API_KEY", "test")
    t = providers._make_tts(providers.ChainItem("elevenlabs", "eleven_flash_v2_5"))
    s = t._opts.voice_settings
    assert (s.stability, s.style, s.speed, s.use_speaker_boost) == (0.75, 0.0, 0.95, False)
    monkeypatch.setenv("ELEVEN_SPEED", "3")  # out of range: clamped, never sent as is
    assert providers.eleven_voice_settings().speed == 1.2


async def test_greeting_plays_from_recording(frozen, tmp_path):
    cache = GreetingCache("v", folder=tmp_path)
    calls = []
    with fake_job_context():
        async with AgentSession(llm=ScriptedLLM([]), userdata=CallState()) as session:
            original = session.say

            def say(text, **kw):
                calls.append((text, "audio" in kw))
                return original(text, **kw)

            session.say = say
            agent = FrontDeskAgent(frozen, scheduler=MockScheduler(frozen), greetings=cache)
            # what the agent will say: record it first, as `agent.py warmup` does
            from lines import inbound_greeting

            text = inbound_greeting(
                clinic=frozen.clinic.name,
                persona="Clara",
                now=frozen.now(),
                disclose_ai=frozen.clinic.disclose_ai_in_greeting,
            )
            cache.save(text, frames(0.5))
            await session.start(agent)
            await asyncio.sleep(0.2)
    assert calls and calls[0] == (text, True)  # played from the recording, not the TTS


async def test_missing_recording_is_spoken_live_then_recorded(frozen, tmp_path):
    class FakeTTS(tts.TTS):
        def __init__(self):
            super().__init__(
                capabilities=tts.TTSCapabilities(streaming=False), sample_rate=24000, num_channels=1
            )

        def synthesize(self, text, *, conn_options=None):
            raise NotImplementedError

    recorded = []

    async def fake_synth(t, text):
        recorded.append(text)
        return 0.1, frames(0.3)

    import greeting_cache

    cache = GreetingCache("v", folder=tmp_path, make_tts=FakeTTS)
    orig = greeting_cache.synthesize
    greeting_cache.synthesize = fake_synth
    try:
        with fake_job_context():
            async with AgentSession(llm=ScriptedLLM([]), userdata=CallState()) as session:
                await session.start(
                    FrontDeskAgent(frozen, scheduler=MockScheduler(frozen), greetings=cache)
                )
                await asyncio.sleep(0.5)
    finally:
        greeting_cache.synthesize = orig
    assert recorded and recorded[0].startswith("Namaste, Sample Eye Clinic.")
    assert cache.load(recorded[0])  # the next caller gets the recording


# -- agent latency log lines -------------------------------------------------------------------


def _telemetry():
    handlers = {}
    session = SimpleNamespace(
        on=lambda ev, cb: handlers.setdefault(ev, cb),
        userdata=SimpleNamespace(note_user_text=lambda t: None),
    )
    return CallTelemetry(session), handlers


def test_every_reply_logs_ttfa_breakdown(caplog):
    tel, h = _telemetry()
    caplog.set_level(logging.INFO, logger="frontdesk.latency")
    user = SimpleNamespace(
        role="user",
        text_content="book an appointment",
        metrics={"end_of_turn_delay": 0.41, "transcription_delay": 0.22},
        interrupted=False,
    )
    bot = SimpleNamespace(
        role="assistant",
        text_content="Sure, which centre?",
        interrupted=False,
        metrics={"e2e_latency": 1.62, "llm_node_ttft": 0.74, "tts_node_ttfb": 0.39},
    )
    greeting = SimpleNamespace(
        role="assistant",
        text_content="Demo Eye and Retina Clinic...",
        interrupted=False,
        metrics={"tts_node_ttfb": 3.27},
    )
    for item in (greeting, user, bot):
        h["conversation_item_added"](SimpleNamespace(item=item))
    lines = [r.getMessage() for r in caplog.records]
    assert "reply 1 (scripted): TTS first audio 3270 ms" in lines[0]
    assert lines[1].startswith("reply 2: TTFA 1620 ms SLOW (target 900 ms)")
    assert "end-of-turn 410 ms + LLM first token 740 ms + TTS first audio 390 ms" in lines[1]
    assert tel.samples["e2e_latency"] == [1.62]


def test_provider_metrics_are_logged(caplog):
    _, h = _telemetry()
    caplog.set_level(logging.INFO, logger="frontdesk.latency")
    meta = metrics.base.Metadata(model_name="eleven_flash_v2_5", model_provider="elevenlabs")
    m = metrics.TTSMetrics(
        label="tts",
        request_id="r",
        timestamp=0,
        ttfb=3.1,
        duration=4.0,
        audio_duration=6.0,
        cancelled=False,
        characters_count=120,
        streamed=True,
        connection_reused=False,
        acquire_time=2.4,
        metadata=meta,
    )
    h["metrics_collected"](SimpleNamespace(metrics=m))
    msg = caplog.records[-1].getMessage()
    assert msg.startswith("TTS elevenlabs/eleven_flash_v2_5: first audio 3100 ms")
    assert "connection NEW (acquire 2400 ms)" in msg


def test_fallback_switch_is_logged(caplog):
    class Dummy:
        def __init__(self):
            self._tts_instances = [object(), object()]
            self.cb = None

        def on(self, ev, cb):
            self.cb = cb

    d = Dummy()
    items = [
        SimpleNamespace(spec="elevenlabs/eleven_flash_v2_5"),
        SimpleNamespace(spec="livekit/cartesia/sonic-3"),
    ]
    providers._log_switches(d, "tts", items)
    caplog.set_level(logging.INFO)
    d.cb(SimpleNamespace(tts=d._tts_instances[0], available=False))
    assert "TTS elevenlabs/eleven_flash_v2_5 FAILED: switching" in caplog.records[-1].getMessage()


# -- TTFA as the phone line sees it (bridge) -------------------------------------------------


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def pcm(level: int, ms: int, rate: int = 8000) -> bytes:
    n = rate * ms // 1000
    return (np.sin(np.arange(n) / 3) * level).astype(np.int16).tobytes()


def test_bridge_measures_greeting_and_reply_ttfa(caplog):
    caplog.set_level(logging.INFO, logger="frontdesk.exotel")
    clock = Clock()
    t = SpeechTiming(clock=clock)
    for _ in range(10):  # line noise only
        clock.t += 0.02
        t.caller_audio(pcm(80, 20), 8000)
    clock.t = 1.2
    t.bot_audio(pcm(4000, 200), 8000)  # greeting
    clock.t = 1.4
    t.bot_audio(pcm(4000, 200), 8000)  # still the greeting
    assert t.greeting_s == pytest.approx(1.2, abs=0.01)
    clock.t = 3.0
    for _ in range(50):  # caller speaks for 1 s
        clock.t += 0.02
        t.caller_audio(pcm(3000, 20), 8000)
    stopped = clock.t
    for _ in range(25):  # then silence on the line
        clock.t += 0.02
        t.caller_audio(pcm(80, 20), 8000)
    clock.t = stopped + 1.3
    t.bot_audio(pcm(0, 100, 8000) + pcm(4000, 100, 8000), 8000)  # reply starts mid-chunk
    assert t.replies == [pytest.approx(1.4, abs=0.01)]
    msgs = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("TIMING greeting: first bot audio 1200 ms") for m in msgs)
    assert any("TIMING reply 1: TTFA 1400 ms" in m and "SLOW" in m for m in msgs)
    assert "median 1400 ms" in t.summary()


def test_bridge_ignores_bot_pauses_without_caller_speech():
    clock = Clock()
    t = SpeechTiming(clock=clock)
    t.bot_audio(pcm(4000, 200), 8000)
    clock.t = 2.0  # a pause inside the greeting, nobody spoke
    t.bot_audio(pcm(4000, 200), 8000)
    assert t.replies == []


# -- logs are kept --------------------------------------------------------------------------------


def test_logtee_keeps_a_clean_timestamped_copy(tmp_path):
    log = tmp_path / "run" / "x.log"
    code = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "logtee.py"),
            str(log),
            "--",
            "python",
            "-c",
            "import sys; print('\\x1b[32mhello\\x1b[0m'); print('नमस्ते'); sys.exit(3)",
        ],
        capture_output=True,
    ).returncode
    assert code == 3
    text = log.read_text(encoding="utf-8")
    assert "| hello" in text and "\x1b" not in text and "नमस्ते" in text
    assert "exited with code 3" in text


def test_caller_sample_wav_is_phone_rate(tmp_path, monkeypatch):
    import agent

    out = tmp_path / "caller.wav"
    agent._save_wav_8k(out, frames(1.0, rate=24000))
    with wave.open(str(out)) as w:
        assert w.getframerate() == 8000 and abs(w.getnframes() - 8000) < 200


def test_gemini_deadline_is_at_least_10s(monkeypatch):
    """Regression: Gemini 400'd every request ("Manually set deadline 4s is too short")."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    gem = providers._make_llm(providers.ChainItem("google", "gemini-3.5-flash"))
    assert gem._opts.http_options.timeout >= 10_000


def test_stt_compare_splits_caller_sentences():
    sys.path.insert(0, str(ROOT / "bench"))
    import stt_compare

    rate = 8000
    noise = (np.random.default_rng(0).normal(0, 30, rate)).astype(np.int16)  # 1 s line noise
    speech = (np.sin(np.arange(int(1.2 * rate)) / 3) * 3000).astype(np.int16)
    pcm = np.concatenate([noise, speech, noise, speech, noise])  # 1 s gap: two sentences
    segs = stt_compare.utterances(pcm, rate)
    assert len(segs) == 2
    assert abs(segs[0][0] - 0.8) < 0.1  # starts ~0.2 s before the speech at 1.0 s
    frames = stt_compare.to_frames_16k(segs[0][1], rate)
    assert all(f.sample_rate == 16000 and f.samples_per_channel == 320 for f in frames)
