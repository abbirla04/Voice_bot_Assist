"""Component latency + accuracy benchmark.

    python bench/latency.py                 # everything the current keys allow
    python bench/latency.py --only local    # no keys needed: turn detector + VAD speed
    python bench/latency.py --only llm tts stt --n 10

Measures, per provider in your chains:
  local  turn-detector (v1-mini, on-device) + VAD inference time
  llm    TTFT (time to first token) with the real system prompt
  tts    TTFB (time to first audio) + real-time factor, EN and HI; saves WAVs to listen to
  stt    final-transcript delay after speech ends + WER/CER, on 8 kHz telephone-band audio
Writes bench/results/latency-<ts>.json and prints a pass/fail table against BENCHMARK.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
import wave
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
from livekit import rtc  # noqa: E402
from livekit.agents import llm, stt, utils  # noqa: E402

from config import load_config  # noqa: E402
from prompts import build_instructions  # noqa: E402
from providers import _chain_env, _make_llm, _make_stt, _make_tts, usable_items  # noqa: E402

OUT = ROOT / "bench" / "results"

# targets (seconds) - keep in sync with BENCHMARK.md
TARGETS = {
    "llm_ttft": (0.45, 0.90),
    "tts_ttfb": (0.25, 0.45),
    "stt_final_delay": (0.30, 0.60),
    "eot_inference": (0.05, 0.10),
    "vad_inference": (0.002, 0.005),
}
WER_TARGET = {"en": 0.15, "hi": 0.20}

SENTENCES = {
    "en": [
        "I want to book an appointment with a retina specialist at the Central Delhi centre.",
        "My father has trouble reading the newspaper and his eyes water a lot.",
        "Can I come on Monday morning with my daughter?",
    ],
    "hi": [
        "मुझे कैलाश कॉलोनी सेंटर में आँखों की जाँच के लिए अपॉइंटमेंट चाहिए।",
        "मेरी माँ की उम्र सत्तर साल है और उन्हें धुंधला दिखाई देता है।",
        "क्या कल सुबह ग्यारह बजे का समय मिल सकता है?",
    ],
}
BOT_LINES = {
    "en": "I have registered an appointment request with Doctor Arjun Mehta at our Green Park clinic for Monday morning.",
    "hi": "आपकी अपॉइंटमेंट रिक्वेस्ट सोमवार सुबह के लिए ग्रीन पार्क क्लिनिक में डॉक्टर अर्जुन मेहता के साथ दर्ज हो गई है।",
}


def p(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    vs = sorted(vals)
    k = (len(vs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(vs) - 1)
    return vs[lo] + (vs[hi] - vs[lo]) * (k - lo)


def stats(vals: list[float]) -> dict:
    return {
        "n": len(vals),
        "p50": p(vals, 0.5),
        "p95": p(vals, 0.95),
        "max": max(vals) if vals else None,
    }


def verdict(metric: str, s: dict) -> str:
    if not s.get("n"):
        return "NO DATA"
    t50, t95 = TARGETS[metric]
    ok = s["p50"] <= t50 and s["p95"] <= t95
    return "PASS" if ok else "FAIL"


def edit_distance(a: list[str], b: list[str]) -> int:
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return d[len(b)]


def norm_text(t: str) -> str:
    import re

    t = t.lower().replace("।", " ").replace("॥", " ")  # Hindi danda is punctuation
    t = re.sub(r"[^\w\sऀ-ॿ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def error_rate(ref: str, hyp: str, lang: str) -> float:
    r, h = norm_text(ref), norm_text(hyp)
    if lang == "hi":  # character error rate is fairer for Hindi spelling variants
        return edit_distance(list(r.replace(" ", "")), list(h.replace(" ", ""))) / max(
            1, len(r.replace(" ", ""))
        )
    return edit_distance(r.split(), h.split()) / max(1, len(r.split()))


# -- local ------------------------------------------------------------------------------
def bench_local(n: int) -> dict:
    from livekit import local_inference as li

    li.init_eot()
    li.init_vad()
    eot, vad = li.EOT(), li.VAD()
    rng = np.random.default_rng(0)
    pcm = (rng.standard_normal(li.EOT_MAX_SAMPLES) * 1500).astype(np.int16)
    eot.predict(pcm)  # warm
    eot_t = []
    for _ in range(max(20, n)):
        t = time.perf_counter()
        eot.predict(pcm)
        eot_t.append(time.perf_counter() - t)
    win = (rng.standard_normal(li.VAD_WINDOW_SAMPLES) * 1500).astype(np.int16)
    vad_t = []
    for _ in range(300):
        t = time.perf_counter()
        vad.predict(win)
        vad_t.append(time.perf_counter() - t)
    return {
        "eot_inference": stats(eot_t),
        "vad_inference": stats(vad_t),
        "cpu_count": os.cpu_count(),
    }


# -- llm ----------------------------------------------------------------------------------
async def bench_llm(n: int) -> dict:
    cfg = load_config()
    system = build_instructions(cfg, direction="inbound")
    turns = [
        "Hello, I want to book an appointment for my mother, she has cataract.",
        "मुझे कल का अपॉइंटमेंट चाहिए, गुड़गाँव में।",
        "What is the consultation fee at the South Delhi centre?",
    ]
    from livekit.agents.llm import ToolContext

    from front_desk import FrontDeskAgent
    from scheduling import MockScheduler

    # send the real tool schemas: they are a big part of the prompt the provider must process
    tools = ToolContext(FrontDeskAgent(cfg, scheduler=MockScheduler(cfg)).tools).flatten()
    out = {}
    items, skipped = usable_items("llm", _chain_env("llm"))
    for item in items:
        model = _make_llm(item)
        ttft, total, errors = [], [], 0
        for i in range(n):
            ctx = llm.ChatContext()
            ctx.add_message(role="system", content=system)
            ctx.add_message(
                role="assistant",
                content="Hello, welcome to Demo Eye and Retina Clinic. I'm Clara, your virtual assistant. How may I help you today?",
            )
            ctx.add_message(role="user", content=turns[i % len(turns)])
            t0 = time.perf_counter()
            first = None
            try:
                async with model.chat(chat_ctx=ctx, tools=tools) as s:
                    async for ch in s:
                        if first is None and ch.delta and (ch.delta.content or ch.delta.tool_calls):
                            first = time.perf_counter() - t0
                total.append(time.perf_counter() - t0)
                if first is not None:
                    ttft.append(first)
            except Exception as e:
                errors += 1
                print(f"  llm {item.spec} error: {e}")
        out[item.spec] = {"llm_ttft": stats(ttft), "total": stats(total), "errors": errors}
        print(
            f"llm {item.spec}: ttft p50 {fmt(out[item.spec]['llm_ttft']['p50'])} p95 {fmt(out[item.spec]['llm_ttft']['p95'])} errors {errors}"
        )
    out["_skipped"] = skipped
    return out


# -- tts ----------------------------------------------------------------------------------
def save_wav(path: Path, frames: list[rtc.AudioFrame]) -> float:
    if not frames:
        return 0.0
    sr, ch = frames[0].sample_rate, frames[0].num_channels
    data = b"".join(bytes(f.data) for f in frames)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data)
    return len(data) / 2 / ch / sr


async def synth(tts_obj, text: str) -> tuple[float | None, float, list[rtc.AudioFrame]]:
    frames: list[rtc.AudioFrame] = []
    t0 = time.perf_counter()
    first = None
    if tts_obj.capabilities.streaming:  # the path the live agent uses
        async with tts_obj.stream() as s:
            s.push_text(text)
            s.end_input()
            async for ev in s:
                if first is None:
                    first = time.perf_counter() - t0
                frames.append(ev.frame)
    else:
        async with tts_obj.synthesize(text) as s:
            async for ev in s:
                if first is None:
                    first = time.perf_counter() - t0
                frames.append(ev.frame)
    return first, time.perf_counter() - t0, frames


async def bench_tts(n: int) -> dict:
    out = {}
    items, skipped = usable_items("tts", _chain_env("tts"))
    for item in items:
        tts_obj = _make_tts(item)
        res = {}
        for lang in ("en", "hi"):
            ttfb, rtf, errors = [], [], 0
            for i in range(n):
                try:
                    first, total, frames = await synth(tts_obj, BOT_LINES[lang])
                    dur = (
                        save_wav(OUT / "audio" / f"tts-{item.provider}-{lang}.wav", frames)
                        if i == 0
                        else sum(f.samples_per_channel / f.sample_rate for f in frames)
                    )
                    if first is not None:
                        ttfb.append(first)
                    if dur:
                        rtf.append(total / dur)
                except Exception as e:
                    errors += 1
                    print(f"  tts {item.spec} {lang} error: {e}")
            res[lang] = {"tts_ttfb": stats(ttfb), "real_time_factor": stats(rtf), "errors": errors}
            print(
                f"tts {item.spec} [{lang}]: ttfb p50 {fmt(res[lang]['tts_ttfb']['p50'])} p95 {fmt(res[lang]['tts_ttfb']['p95'])} errors {errors}"
            )
        out[item.spec] = res
    out["_skipped"] = skipped
    return out


# -- stt ----------------------------------------------------------------------------------
def _mulaw_roundtrip(pcm: np.ndarray) -> np.ndarray:
    """G.711 mu-law encode+decode (the codec on Indian PSTN/Exotel legs)."""
    x = pcm.astype(np.float64) / 32768.0
    mu = 255.0
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    y = np.round((y + 1) / 2 * 255) / 255 * 2 - 1  # 8-bit quantisation
    x2 = np.sign(y) * (np.power(1 + mu, np.abs(y)) - 1) / mu
    return (x2 * 32767).astype(np.int16)


def _trim(pcm: np.ndarray, threshold: float = 300.0, frame: int = 320) -> np.ndarray:
    frames = [pcm[i : i + frame] for i in range(0, len(pcm), frame)]
    loud = [
        i for i, f in enumerate(frames) if np.sqrt(np.mean(f.astype(np.float64) ** 2)) >= threshold
    ]
    return np.concatenate(frames[loud[0] : loud[-1] + 1]) if loud else pcm


def to_phone_band(frames: list[rtc.AudioFrame]) -> list[rtc.AudioFrame]:
    """Simulate a phone call: 8 kHz band, G.711 mu-law, line noise; back to 16 kHz."""
    down = rtc.AudioResampler(frames[0].sample_rate, 8000, num_channels=1)
    mid: list[rtc.AudioFrame] = []
    for f in frames:
        mid += down.push(f)
    mid += down.flush()
    up = rtc.AudioResampler(8000, 16000, num_channels=1)
    out: list[rtc.AudioFrame] = []
    for f in mid:
        out += up.push(f)
    out += up.flush()
    pcm16 = np.frombuffer(b"".join(bytes(f.data) for f in out), dtype=np.int16)
    pcm16 = _trim(_mulaw_roundtrip(pcm16))
    noise = np.random.default_rng(1).normal(0, 60, pcm16.shape)  # faint line hiss
    pcm = np.clip(pcm16 + noise, -32768, 32767).astype(np.int16).tobytes()
    # re-chunk to 20 ms frames
    step = 320 * 2
    return [
        rtc.AudioFrame(pcm[i : i + step], 16000, 1, len(pcm[i : i + step]) // 2)
        for i in range(0, len(pcm) - step + 1, step)
    ]


async def transcribe_stream(
    stt_obj: stt.STT, frames: list[rtc.AudioFrame], lang: str
) -> tuple[str, float | None]:
    stream = stt_obj.stream()
    finals: list[str] = []
    end_at: float | None = None
    first_final_after_end: float | None = None

    async def feed():
        nonlocal end_at
        silence = rtc.AudioFrame(b"\x00\x00" * 320, 16000, 1, 320)
        for f in frames:
            stream.push_frame(f)
            await asyncio.sleep(0.02)
        end_at = time.perf_counter()
        for _ in range(75):  # 1.5 s of trailing silence, like a caller pausing
            stream.push_frame(silence)
            await asyncio.sleep(0.02)
        stream.end_input()

    async def read():
        nonlocal first_final_after_end
        async for ev in stream:
            if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT and ev.alternatives:
                finals.append(ev.alternatives[0].text)
                if end_at is not None and first_final_after_end is None:
                    first_final_after_end = time.perf_counter() - end_at

    await asyncio.wait_for(asyncio.gather(feed(), read()), timeout=60)
    await stream.aclose()
    return " ".join(finals), first_final_after_end


async def bench_stt(n: int) -> dict:
    out = {}
    tts_items, _ = usable_items("tts", _chain_env("tts"))
    if not tts_items:
        return {"_skipped": "no TTS available to generate test speech"}
    voice = _make_tts(tts_items[0])
    audio: dict[str, list[tuple[str, list[rtc.AudioFrame]]]] = {}
    for lang, sents in SENTENCES.items():
        audio[lang] = []
        for s in sents:
            _, _, frames = await synth(voice, s)
            audio[lang].append((s, to_phone_band(frames)))
    items, skipped = usable_items("stt", _chain_env("stt"))
    for item in items:
        stt_obj = _make_stt(item, [])
        res = {}
        for lang, clips in audio.items():
            delays, errs, failures = [], [], 0
            for i in range(n):
                ref, frames = clips[i % len(clips)]
                try:
                    hyp, delay = await transcribe_stream(stt_obj, frames, lang)
                    if delay is not None:
                        delays.append(delay)
                    errs.append(error_rate(ref, hyp, lang))
                    if i < len(clips):
                        print(f"  stt {item.spec} [{lang}] '{hyp[:80]}'")
                except Exception as e:
                    failures += 1
                    print(f"  stt {item.spec} {lang} error: {e}")
            res[lang] = {
                "stt_final_delay": stats(delays),
                "error_rate_mean": statistics.fmean(errs) if errs else None,
                "metric": "CER" if lang == "hi" else "WER",
                "errors": failures,
            }
            print(
                f"stt {item.spec} [{lang}]: final delay p50 {fmt(res[lang]['stt_final_delay']['p50'])} {res[lang]['metric']} {res[lang]['error_rate_mean']}"
            )
        out[item.spec] = res
    out["_skipped"] = skipped
    return out


def fmt(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x * 1000:.2f}ms" if x < 0.001 else f"{x * 1000:.0f}ms"


def table(results: dict) -> list[str]:
    rows = []
    if "local" in results:
        for m in ("eot_inference", "vad_inference"):
            s = results["local"][m]
            rows.append(f"| local {m} | {fmt(s['p50'])} | {fmt(s['p95'])} | {verdict(m, s)} |")
    for spec, r in results.get("llm", {}).items():
        if not spec.startswith("_"):
            rows.append(
                f"| LLM TTFT {spec} | {fmt(r['llm_ttft']['p50'])} | {fmt(r['llm_ttft']['p95'])} | {verdict('llm_ttft', r['llm_ttft'])} |"
            )
    for spec, r in results.get("tts", {}).items():
        if not spec.startswith("_"):
            for lang, v in r.items():
                rows.append(
                    f"| TTS TTFB {spec} [{lang}] | {fmt(v['tts_ttfb']['p50'])} | {fmt(v['tts_ttfb']['p95'])} | {verdict('tts_ttfb', v['tts_ttfb'])} |"
                )
    for spec, r in results.get("stt", {}).items():
        if not spec.startswith("_"):
            for lang, v in r.items():
                er = v["error_rate_mean"]
                ok = er is not None and er <= WER_TARGET[lang]
                rows.append(
                    f"| STT final delay {spec} [{lang}] | {fmt(v['stt_final_delay']['p50'])} | {fmt(v['stt_final_delay']['p95'])} | {verdict('stt_final_delay', v['stt_final_delay'])} |"
                )
                rows.append(
                    f"| STT {v['metric']} {spec} [{lang}] | {er if er is None else f'{er:.1%}'} | target <= {WER_TARGET[lang]:.0%} | {'PASS' if ok else 'FAIL'} |"
                )
    for kind in ("llm", "tts", "stt"):
        r = results.get(kind)
        if r is None:
            continue
        usable = [k for k in r if not k.startswith("_")]
        skipped = r.get("_skipped") or {}
        if not usable:
            keys = (
                sorted({k for ks in skipped.values() for k in ks})
                if isinstance(skipped, dict)
                else []
            )
            why = f"needs {', '.join(keys)}" if keys else str(skipped)
            rows.append(f"| {kind.upper()} | - | - | BLOCKED: {why} |")
    return ["| metric | p50 | p95 | verdict |", "|---|---|---|---|", *rows]


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=["local", "llm", "tts", "stt"])
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    which = args.only or ["local", "llm", "tts", "stt"]
    results: dict = {"at": time.time()}
    async with utils.http_context.open():
        if "local" in which:
            results["local"] = bench_local(args.n)
            print("local:", json.dumps({k: v for k, v in results["local"].items()}, default=str))
        if "llm" in which:
            results["llm"] = await bench_llm(args.n)
        if "tts" in which:
            results["tts"] = await bench_tts(args.n)
        if "stt" in which:
            results["stt"] = await bench_stt(args.n)
    lines = table(results)
    print("\n".join(lines))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"latency-{int(time.time())}.json"
    path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nwritten {path}")
    body = lines[2:]
    if any("FAIL" in row or "NO DATA" in row for row in body):
        return 1
    return 2 if any("BLOCKED" in row for row in body) else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
