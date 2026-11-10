"""Pretend to be Exotel: call the bridge over its WebSocket, exactly as the Voicebot applet does.

Use it to check the bridge + agent path on your PC BEFORE dialling the real number, and to
check the ngrok tunnel (pass the public wss:// URL). It sends Exotel's `connected`/`start`
events, streams caller audio in real time (silence, or a WAV you give it), records what the
bot says, and checks the replies follow Exotel's rules.

    uv run python bench/exotel_sim.py                                   # local bridge
    uv run python bench/exotel_sim.py --url wss://<domain>/exotel/stream # through ngrok
    uv run python bench/exotel_sim.py --wav bench/results/caller-sample.wav  # + one caller turn

Prints the greeting latency and, with --wav, the TTFA of the bot's reply as the phone line
would see it (the caller's sentence ends -> the bot's first audio arrives). Saves the bot's
audio to bench/results/exotel-sim-<ts>.wav and a result line to logs/sim-results.jsonl.
`python src/agent.py warmup` records bench/results/caller-sample.wav.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
import uuid
import wave
from pathlib import Path

import aiohttp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")


def load_wav(path: Path, rate: int) -> bytes:
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise SystemExit("WAV must be 16-bit PCM")
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if w.getnchannels() > 1:
            pcm = pcm.reshape(-1, w.getnchannels()).mean(axis=1).astype(np.int16)
        src = w.getframerate()
    if src != rate:  # linear resample is fine for a smoke test
        n = int(len(pcm) * rate / src)
        pcm = np.interp(np.linspace(0, len(pcm) - 1, n), np.arange(len(pcm)), pcm).astype(np.int16)
    return pcm.tobytes()


LOUD = 300  # int16 peak that counts as speech (bot or caller)


def speech_length_s(pcm: bytes, rate: int) -> float:
    """Where the caller's speech really ends inside the WAV (TTS adds trailing silence)."""
    a = np.abs(np.frombuffer(pcm, dtype=np.int16))
    loud = np.nonzero(a > LOUD)[0]
    return (loud[-1] + 1) / rate if len(loud) else 0.0


async def run(
    url: str, seconds: float, rate: int, wav: Path | None, caller: str, to: str = "01140000000"
) -> int:
    call_sid = "sim-" + uuid.uuid4().hex[:10]
    stream_sid = "st-" + call_sid
    frame_bytes = rate // 50 * 2  # Exotel sends 20 ms packets
    speech = load_wav(wav, rate) if wav else b""
    speech_len = speech_length_s(speech, rate) if speech else 0.0
    will_speak = bool(speech)
    received = bytearray()
    sizes: list[int] = []
    events: dict[str, int] = {}
    first_audio: float | None = None  # greeting latency
    last_loud: float | None = None
    speech_start: float | None = None
    speech_end: float | None = None  # caller stops speaking
    reply_at: float | None = None  # first bot audio after that = TTFA
    t0 = time.monotonic()

    async with aiohttp.ClientSession() as http, http.ws_connect(url, heartbeat=20) as ws:
        await ws.send_str(json.dumps({"event": "connected"}))
        await ws.send_str(
            json.dumps(
                {
                    "event": "start",
                    "sequence_number": 1,
                    "stream_sid": stream_sid,
                    "start": {
                        "stream_sid": stream_sid,
                        "call_sid": call_sid,
                        "account_sid": "simulator",
                        "from": caller,
                        "to": to,
                        "custom_parameters": {},
                        "media_format": {
                            "encoding": "audio/x-raw",
                            "sample_rate": str(rate),
                            "bit_rate": "16",
                        },
                    },
                }
            )
        )
        t0 = time.monotonic()

        async def reader() -> None:
            nonlocal first_audio, last_loud, reply_at
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    break
                data = json.loads(msg.data)
                ev = data.get("event", "?")
                events[ev] = events.get(ev, 0) + 1
                if ev == "media":
                    pcm = base64.b64decode(data["media"]["payload"])
                    sizes.append(len(pcm))
                    received.extend(pcm)
                    loud = np.nonzero(np.abs(np.frombuffer(pcm, dtype=np.int16)) > LOUD)[0]
                    if len(loud):
                        # a chunk starts playing when it arrives: the caller hears its first
                        # loud sample that much later
                        now = time.monotonic() - t0 + loud[0] / rate
                        first_audio = first_audio if first_audio is not None else now
                        if speech_end is not None and reply_at is None and now > speech_end:
                            reply_at = now
                        last_loud = time.monotonic() - t0
                elif ev == "mark":  # echo it, as Exotel does once audio before it played
                    await ws.send_str(json.dumps(data))

        def done() -> bool:
            now = time.monotonic() - t0
            quiet = last_loud is not None and now - last_loud > 2.0
            if not will_speak:
                return quiet  # greeting finished
            return reply_at is not None and quiet  # reply finished

        read_task = asyncio.create_task(reader())
        sent = 0
        seq = 2
        start = time.monotonic()
        while time.monotonic() - start < seconds and not ws.closed and not done():
            chunk = b"\x00" * frame_bytes
            now = time.monotonic() - t0
            # speak 1.5 s after the greeting has gone quiet
            if speech and last_loud is not None and now - last_loud > 1.5 and speech_end is None:
                if speech_start is None:
                    speech_start = now
                chunk = speech[:frame_bytes].ljust(frame_bytes, b"\x00")
                speech = speech[frame_bytes:]
                if not speech:
                    speech_end = speech_start + speech_len
            await ws.send_str(
                json.dumps(
                    {
                        "event": "media",
                        "sequence_number": seq,
                        "stream_sid": stream_sid,
                        "media": {
                            "chunk": seq,
                            "timestamp": str(sent * 20),
                            "payload": base64.b64encode(chunk).decode(),
                        },
                    }
                )
            )
            seq += 1
            sent += 1
            await asyncio.sleep(max(0.0, start + sent * 0.02 - time.monotonic()))
        if not ws.closed:
            await ws.send_str(
                json.dumps(
                    {"event": "stop", "stream_sid": stream_sid, "stop": {"reason": "callended"}}
                )
            )
        await asyncio.wait_for(read_task, 5)

    out_dir = ROOT / "bench" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"exotel-sim-{time.strftime('%Y%m%d-%H%M%S')}.wav"
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(received))

    bad = [s for s in sizes if s % 320 or not 3200 <= s <= 100_000]
    greeting_ms = round(first_audio * 1000) if first_audio is not None else None
    ttfa_ms = round((reply_at - speech_end) * 1000) if reply_at and speech_end else None
    print(f"events from bridge: {events}")
    print(f"bot audio: {len(received) / (2 * rate):.1f} s in {len(sizes)} chunks -> {out}")
    print(
        f"greeting: first bot audio {greeting_ms} ms after the call connected (target 1500)"
        if greeting_ms is not None
        else "NO BOT AUDIO: is the agent running? check the bridge log"
    )
    if speech_end is not None:
        print(
            f"TTFA: caller stopped -> bot audio at the phone line: {ttfa_ms} ms "
            f"(target 900 + ~250 for the phone network)"
            if ttfa_ms is not None
            else "TTFA: NO REPLY to the caller's sentence"
        )
    print(
        "chunk sizes: OK (multiples of 320, 3200..100000 bytes)"
        if not bad
        else f"BAD chunk sizes: {sorted(set(bad))}"
    )
    log = ROOT / "logs" / "sim-results.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "url": url,
                    "rate": rate,
                    "greeting_ms": greeting_ms,
                    "ttfa_ms": ttfa_ms,
                    "spoke": speech_end is not None,
                    "chunks_ok": not bad,
                    "wav": out.name,
                }
            )
            + "\n"
        )
    ok = greeting_ms is not None and not bad and (speech_end is None or ttfa_ms is not None)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", default="ws://127.0.0.1:8765/exotel/stream")
    ap.add_argument("--seconds", type=float, default=40, help="upper limit; ends earlier once done")
    ap.add_argument("--rate", type=int, default=8000, choices=[8000, 16000, 24000])
    ap.add_argument("--wav", type=Path, help="caller speech, spoken after the greeting")
    ap.add_argument("--caller", default="09999999999", help="caller ID the bridge receives")
    ap.add_argument(
        "--to",
        default="01140000000",
        help="number dialled",
    )
    a = ap.parse_args()
    return asyncio.run(run(a.url, a.seconds, a.rate, a.wav, a.caller, a.to))


if __name__ == "__main__":
    raise SystemExit(main())
