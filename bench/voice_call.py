"""End-to-end VOICE benchmark over a real LiveKit room (the closest thing to a phone call).

A synthetic caller joins the room, speaks pre-generated utterances in real time, and
listens to the agent's audio. It measures what callers feel:

  TTFA      caller stops speaking -> first agent audio (per turn)
  greeting  caller joins -> first agent audio
  barge-in  caller talks over the agent -> agent audio stops (stop latency, missed rate)
  backchannel  caller says "hmm"/"haan ji" over the agent -> agent keeps going (false-stop rate)

Run (agent must be reachable on LiveKit Cloud):
    python src/agent.py dev            # terminal 1 (or deploy with `lk agent deploy`)
    python bench/voice_call.py         # terminal 2
Saves bench/results/voice-<ts>.json and a stereo WAV (caller L / agent R) you can listen to
or feed to hotato:  uvx hotato run --stereo bench/results/voice-<ts>.wav --expect yield
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import itertools
import json
import os
import sys
import time
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _stream in (sys.stdout, sys.stderr):  # Hindi text on a cp1252 Windows console
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "bench")]

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env.local")
load_dotenv(ROOT / ".env")

import numpy as np  # noqa: E402

SR = 16000
FRAME = 320  # 20 ms
OUT = ROOT / "bench" / "results"

TARGETS = {
    "ttfa_p50": 0.90,
    "ttfa_p95": 1.40,
    "greeting": 1.50,
    "bargein_stop_p95": 0.40,
    "bargein_missed_rate": 0.05,
    "backchannel_false_stop_rate": 0.10,
}

TURNS = [
    ("en", "Hello, I want to book an eye check-up for my father."),
    ("en", "He is a new patient. His name is Ramesh Kumar and he is sixty two years old."),
    ("en", "The Green Park clinic is fine for us."),
    ("hi", "क्या कल सुबह का कोई समय मिल सकता है?"),
]
LONG_PROMPT = "Can you please tell me about all your retina doctors and at which centres they sit?"
INTERRUPTS = ["Wait, wait, one second please.", "Sorry, stop, I have a question.", "रुकिए, एक मिनट।"]
BACKCHANNELS = ["Hmm.", "Haan ji.", "Okay."]


# -- agent-audio activity tracking (pure logic, unit tested) -------------------------------
@dataclass
class SpeechTracker:
    threshold: float = 350.0  # int16 RMS
    start_frames: int = 2
    end_frames: int = 15  # 300 ms of quiet ends a segment
    speaking: bool = False
    segments: list[list[float]] = field(default_factory=list)  # [start, end|None]
    _loud: int = 0
    _quiet: int = 0
    _first_quiet_at: float | None = None
    _changed: asyncio.Event | None = None

    def changed(self) -> asyncio.Event:
        if self._changed is None:
            self._changed = asyncio.Event()
        return self._changed

    def push(self, t: float, rms: float) -> None:
        if rms >= self.threshold:
            self._loud += 1
            self._quiet = 0
            self._first_quiet_at = None
            if not self.speaking and self._loud >= self.start_frames:
                self.speaking = True
                self.segments.append([t - (self.start_frames - 1) * FRAME / SR, None])
                self.changed().set()
        else:
            self._loud = 0
            if self.speaking:
                self._quiet += 1
                if self._first_quiet_at is None:
                    self._first_quiet_at = t
                if self._quiet >= self.end_frames:
                    self.speaking = False
                    self.segments[-1][1] = self._first_quiet_at
                    self.changed().set()

    def first_start_after(self, t: float) -> float | None:
        return next((s for s, _ in self.segments if s >= t), None)

    async def wait(self, pred, timeout: float) -> bool:
        deadline = time.perf_counter() + timeout
        while not pred():
            left = deadline - time.perf_counter()
            if left <= 0:
                return False
            ev = self.changed()
            ev.clear()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(ev.wait(), min(left, 0.1))
        return True


def gaps(segments: list[list[float]], until: float) -> list[tuple[float, float]]:
    """Quiet intervals between agent speech segments: (start, end)."""
    out = []
    for (_, e1), (s2, _) in itertools.pairwise(segments):
        if e1 is not None:
            out.append((e1, s2))
    if segments and segments[-1][1] is not None:
        out.append((segments[-1][1], until))
    return out


def stop_latency(
    segments,
    b_start: float,
    b_end: float,
    until: float,
    min_gap: float = 0.8,
    max_wait: float = 2.0,
) -> float | None:
    """Time from caller barge-in start to the agent going quiet for real.

    A short pause between sentences is not a stop: the quiet must last min_gap, or last
    until the caller finishes the interruption (the agent then answers it). Going quiet
    later than max_wait is the agent simply finishing its answer: that is a MISS.
    """
    for g0, g1 in gaps(segments, until):
        if g0 < b_start - 0.05:
            continue
        if g0 > b_start + max_wait:
            return None
        if g1 - g0 >= min_gap or g1 >= b_end + 0.2:
            return g0 - b_start
    return None


def false_stop(
    segments, b_start: float, until: float, window: float = 1.5, min_gap: float = 1.0
) -> bool:
    """Backchannel handling failed if the agent goes quiet for >= min_gap shortly after it."""
    return any(
        b_start - 0.05 <= g0 <= b_start + window and g1 - g0 >= min_gap
        for g0, g1 in gaps(segments, until)
    )


def trim_silence(pcm: np.ndarray, threshold: float = 300.0) -> np.ndarray:
    """Cut leading/trailing near-silence from a TTS clip so 'caller stopped' is exact."""
    if pcm.size == 0:
        return pcm
    frames = [pcm[i : i + FRAME] for i in range(0, len(pcm), FRAME)]
    loud = [i for i, f in enumerate(frames) if rms(f) >= threshold]
    if not loud:
        return pcm
    return np.concatenate(frames[loud[0] : loud[-1] + 1])


def rms(pcm: np.ndarray) -> float:
    return float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2))) if pcm.size else 0.0


# -- caller audio ----------------------------------------------------------------------
async def synth_pcm(texts: list[str]) -> dict[str, np.ndarray]:
    """Generate caller utterances once (cached as WAV), 16 kHz mono int16."""
    from livekit import rtc
    from livekit.agents import utils

    from providers import _chain_env, _make_tts, usable_items

    cache = OUT / "caller_audio"
    cache.mkdir(parents=True, exist_ok=True)
    out: dict[str, np.ndarray] = {}
    todo = []
    for t in texts:
        f = cache / f"{uuid.uuid5(uuid.NAMESPACE_URL, t).hex[:12]}.wav"
        if f.exists():
            with wave.open(str(f)) as w:
                out[t] = trim_silence(np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16))
        else:
            todo.append((t, f))
    if not todo:
        return out
    spec = os.getenv("BENCH_CALLER_TTS")
    items, _ = usable_items("tts", spec or _chain_env("tts"))
    if not items:
        raise SystemExit("need a TTS key to generate caller audio (or BENCH_CALLER_TTS)")
    tts_obj = _make_tts(items[-1])  # a different voice from the bot's primary, when possible
    async with utils.http_context.open():
        for t, f in todo:
            frames = []
            async with tts_obj.synthesize(t) as s:
                async for ev in s:
                    frames.append(ev.frame)
            res = rtc.AudioResampler(frames[0].sample_rate, SR, num_channels=1)
            pcm = b""
            for fr in frames:
                for o in res.push(fr):
                    pcm += bytes(o.data)
            for o in res.flush():
                pcm += bytes(o.data)
            arr = trim_silence(np.frombuffer(pcm, dtype=np.int16))
            with wave.open(str(f), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SR)
                w.writeframes(arr.tobytes())
            out[t] = arr
    return out


class Caller:
    def __init__(self) -> None:
        from livekit import rtc

        self.rtc = rtc
        self.room = rtc.Room()
        self.source = rtc.AudioSource(SR, 1, queue_size_ms=60)
        self.tracker = SpeechTracker()
        self.queue: asyncio.Queue[tuple[np.ndarray, asyncio.Future]] = asyncio.Queue()
        self.rec_caller: list[np.ndarray] = []
        self.rec_agent: list[np.ndarray] = []
        self._tasks: list[asyncio.Task] = []
        self.agent_audio_ready = asyncio.Event()

    async def connect(self, url: str, token: str) -> None:
        rtc = self.rtc

        @self.room.on("track_subscribed")
        def _sub(track, pub, participant):
            # the agent publishes two audio tracks: its voice and "background_audio" (hold
            # music, thinking sounds). Measure the voice only, or timings get mixed up.
            if track.kind == rtc.TrackKind.KIND_AUDIO and pub.name != "background_audio":
                self._tasks.append(asyncio.create_task(self._listen(track)))

        await self.room.connect(url, token)
        track = rtc.LocalAudioTrack.create_audio_track("caller-mic", self.source)
        opts = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        await self.room.local_participant.publish_track(track, opts)
        self._tasks.append(asyncio.create_task(self._mic()))

    async def _listen(self, track) -> None:
        stream = self.rtc.AudioStream(track, sample_rate=SR, num_channels=1, frame_size_ms=20)
        self.agent_audio_ready.set()
        async for ev in stream:
            pcm = np.frombuffer(bytes(ev.frame.data), dtype=np.int16)
            self.rec_agent.append(pcm)
            self.tracker.push(time.perf_counter(), rms(pcm))

    async def _mic(self) -> None:
        silence = np.zeros(FRAME, dtype=np.int16)
        current: np.ndarray | None = None
        done: asyncio.Future | None = None
        pos = 0
        while True:
            if current is None and not self.queue.empty():
                current, done = self.queue.get_nowait()
                pos = 0
            if current is not None:
                chunk = current[pos : pos + FRAME]
                if len(chunk) < FRAME:
                    chunk = np.pad(chunk, (0, FRAME - len(chunk)))
                pos += FRAME
            else:
                chunk = silence
            frame = self.rtc.AudioFrame(chunk.tobytes(), SR, 1, FRAME)
            await self.source.capture_frame(frame)
            self.rec_caller.append(chunk)
            if current is not None and pos >= len(current):
                # last frame captured; it reaches the wire after what is still queued
                end_t = time.perf_counter() + self.source.queued_duration
                if done and not done.done():
                    done.set_result(end_t)
                current, done = None, None

    async def say(self, pcm: np.ndarray) -> tuple[float, float]:
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        start = time.perf_counter() + self.source.queued_duration
        await self.queue.put((pcm, fut))
        end = await fut
        return start, end

    def save_wav(self, path: Path) -> None:
        n = min(len(self.rec_caller), len(self.rec_agent))
        if not n:
            return
        left = np.concatenate(self.rec_caller[-n:])
        right = np.concatenate(self.rec_agent[-n:])
        m = min(len(left), len(right))
        stereo = np.stack([left[:m], right[:m]], axis=1)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(stereo.astype(np.int16).tobytes())

    async def close(self) -> None:
        for t in self._tasks:
            t.cancel()
        await self.room.disconnect()


def pct(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    vs = sorted(vals)
    k = (len(vs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(vs) - 1)
    if vs[hi] == float("inf"):  # unanswered turns: don't interpolate into NaN
        return vs[hi] if (k > lo or vs[lo] == float("inf")) else vs[lo]
    return vs[lo] + (vs[hi] - vs[lo]) * (k - lo)


async def run(args) -> dict:
    from livekit import api

    url, key, secret = (
        os.getenv(k) for k in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
    )
    if not (url and key and secret):
        raise SystemExit("needs LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET")
    texts = [t for _, t in TURNS] + [LONG_PROMPT] + INTERRUPTS + BACKCHANNELS
    audio = await synth_pcm(texts)

    room_name = f"bench-{uuid.uuid4().hex[:8]}"
    lk = api.LiveKitAPI(url, key, secret)
    agent_name = os.getenv("AGENT_NAME", "eye-front-desk")
    token = (
        api.AccessToken(key, secret)
        .with_identity("bench-caller")
        .with_grants(api.VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )
    caller = Caller()
    result: dict = {"room": room_name, "turns": [], "bargein": [], "backchannel": []}
    try:
        await caller.connect(url, token)
        tr = caller.tracker
        # like a phone call: the caller is already in the room when the agent is dispatched
        dispatched_at = time.perf_counter()
        await lk.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(agent_name=agent_name, room=room_name)
        )
        ok = await tr.wait(lambda: bool(tr.segments), timeout=20)
        result["greeting_s"] = (tr.segments[0][0] - dispatched_at) if ok else None
        await tr.wait(lambda: not tr.speaking, timeout=30)
        await asyncio.sleep(0.5)

        # conversational turns -> TTFA
        for lang, text in TURNS:
            _, end = await caller.say(audio[text])
            got = await tr.wait(lambda end=end: tr.first_start_after(end) is not None, timeout=15)
            ttfa = (tr.first_start_after(end) - end) if got else None
            result["turns"].append({"lang": lang, "text": text, "ttfa_s": ttfa})
            print(f"turn [{lang}] TTFA {'TIMEOUT' if ttfa is None else f'{ttfa * 1000:.0f} ms'}")
            await tr.wait(lambda: not tr.speaking, timeout=40)
            await asyncio.sleep(0.6)

        # barge-in and backchannel trials
        trials = [("bargein", INTERRUPTS[i % len(INTERRUPTS)]) for i in range(args.trials)] + [
            ("backchannel", BACKCHANNELS[i % len(BACKCHANNELS)]) for i in range(args.trials)
        ]
        for kind, text in trials:
            _, end = await caller.say(audio[LONG_PROMPT])
            if not await tr.wait(lambda end=end: tr.first_start_after(end) is not None, timeout=15):
                result[kind].append({"text": text, "error": "agent did not answer the long prompt"})
                continue
            started = tr.first_start_after(end)
            await asyncio.sleep(max(0.0, started + 1.2 - time.perf_counter()))
            if not tr.speaking:
                result[kind].append({"text": text, "error": "agent answer too short to interrupt"})
                continue
            b_start, b_end = await caller.say(audio[text])
            await asyncio.sleep(4.0)  # let the agent react, then analyse the audio timeline
            now = time.perf_counter()
            if kind == "bargein":
                stop = stop_latency(tr.segments, b_start, b_end, now)
                result[kind].append({"text": text, "stop_latency_s": stop, "missed": stop is None})
                print(
                    f"barge-in '{text}': {'MISSED' if stop is None else f'stopped in {stop * 1000:.0f} ms'}"
                )
            else:
                fs = false_stop(tr.segments, b_start, now)
                result[kind].append({"text": text, "false_stop": fs})
                print(f"backchannel '{text}': {'FALSE STOP' if fs else 'kept talking'}")
            await tr.wait(lambda: not tr.speaking, timeout=40)
            await asyncio.sleep(0.8)
    finally:
        ts = int(time.time())
        OUT.mkdir(parents=True, exist_ok=True)
        caller.save_wav(OUT / f"voice-{ts}.wav")
        result["wav"] = str(OUT / f"voice-{ts}.wav")
        await caller.close()
        try:
            await lk.room.delete_room(api.DeleteRoomRequest(room=room_name))
        finally:
            await lk.aclose()
    return result


def score(r: dict) -> list[tuple[str, str, str, bool]]:
    ttfa = [t["ttfa_s"] for t in r["turns"] if t.get("ttfa_s") is not None]
    timeouts = sum(1 for t in r["turns"] if t.get("ttfa_s") is None)
    stops = [b["stop_latency_s"] for b in r["bargein"] if b.get("stop_latency_s") is not None]
    b_trials = [b for b in r["bargein"] if "missed" in b]
    c_trials = [b for b in r["backchannel"] if "false_stop" in b]
    missed = sum(b["missed"] for b in b_trials) / len(b_trials) if b_trials else None
    false_stop = sum(b["false_stop"] for b in c_trials) / len(c_trials) if c_trials else None
    rows = []

    def row(name, val, target, ok, fmt="ms"):
        if val is None:
            rows.append((name, "n/a", target, False))
        else:
            if val == float("inf"):
                shown = "never"
            else:
                shown = f"{val * 1000:.0f} ms" if fmt == "ms" else f"{val:.0%}"
            rows.append((name, shown, target, ok))

    # a turn the bot never answered counts as infinitely slow
    ttfa_all = ttfa + [float("inf")] * timeouts
    p50, p95 = pct(ttfa_all, 0.5), pct(ttfa_all, 0.95)
    row(
        "TTFA p50 (caller stops -> bot audio)",
        p50,
        "<= 900 ms",
        p50 is not None and p50 <= TARGETS["ttfa_p50"],
    )
    row(
        f"TTFA p95 ({timeouts} unanswered)",
        p95,
        "<= 1400 ms",
        p95 is not None and p95 <= TARGETS["ttfa_p95"],
    )
    g = r.get("greeting_s")
    row("Greeting first audio", g, "<= 1500 ms", g is not None and g <= TARGETS["greeting"])
    sp95 = pct(stops, 0.95)
    row(
        "Barge-in stop latency p95",
        sp95,
        "<= 400 ms",
        sp95 is not None and sp95 <= TARGETS["bargein_stop_p95"],
    )
    row(
        "Barge-in missed",
        missed,
        "<= 5%",
        missed is not None and missed <= TARGETS["bargein_missed_rate"],
        fmt="%",
    )
    row(
        "Backchannel false stops",
        false_stop,
        "<= 10%",
        false_stop is not None and false_stop <= TARGETS["backchannel_false_stop_rate"],
        fmt="%",
    )
    return rows


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=6, help="barge-in and backchannel trials each")
    args = ap.parse_args()
    r = await run(args)
    rows = score(r)
    r["score"] = [dict(zip(("metric", "value", "target", "pass"), x, strict=True)) for x in rows]
    print("\n| metric | value | target | verdict |\n|---|---|---|---|")
    for name, val, target, ok in rows:
        print(f"| {name} | {val} | {target} | {'PASS' if ok else 'FAIL'} |")
    out = OUT / f"voice-{int(time.time())}.json"
    out.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten {out}\naudio   {r['wav']}")
    print(
        "note: measured over WebRTC; a PSTN call through Exotel adds roughly 150-300 ms each way."
    )
    return 0 if all(x[3] for x in rows) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
