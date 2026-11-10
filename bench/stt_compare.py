"""Which speech recogniser hears your callers best? Re-runs recorded test calls through every
STT in STT_CHAIN and prints, per caller sentence, what each one heard and how fast.

    uv run python bench/stt_compare.py                 # the 3 latest recorded calls
    uv run python bench/stt_compare.py --calls 10
    uv run python bench/stt_compare.py logs/audio/20261018-161344-xxx-caller.wav

Recordings come from the Exotel bridge when EXOTEL_RECORD_DIR is set (the phone-test starter
sets it to logs/audio). Writes bench/results/stt-compare-<ts>.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "bench")]

import numpy as np  # noqa: E402
from latency import transcribe_stream  # noqa: E402  (also loads .env.local)
from livekit import rtc  # noqa: E402
from livekit.agents import utils  # noqa: E402

from providers import _chain_env, _make_stt, usable_items  # noqa: E402

FRAME_S = 0.02


def utterances(pcm: np.ndarray, rate: int) -> list[tuple[float, np.ndarray]]:
    """Caller sentences: energy above the line's noise floor, gaps under 0.7 s merged."""
    n = int(rate * FRAME_S)
    frames = [pcm[i : i + n] for i in range(0, len(pcm) - n + 1, n)]
    if not frames:
        return []
    rms = np.array([np.sqrt(np.mean(f.astype(np.float64) ** 2)) for f in frames])
    floor = float(np.percentile(rms, 20))
    loud = rms > max(250.0, 4 * floor)
    segs: list[list[int]] = []
    for i, is_loud in enumerate(loud):
        if not is_loud:
            continue
        if segs and i - segs[-1][1] <= int(0.7 / FRAME_S):
            segs[-1][1] = i
        else:
            segs.append([i, i])
    out = []
    pad = int(0.2 / FRAME_S)
    for a, b in segs:
        if (b - a + 1) * FRAME_S < 0.4:
            continue  # a click or a cough
        a, b = max(0, a - pad), min(len(frames) - 1, b + pad)
        out.append((a * FRAME_S, np.concatenate(frames[a : b + 1])))
    return out


def to_frames_16k(pcm: np.ndarray, rate: int) -> list[rtc.AudioFrame]:
    rs = rtc.AudioResampler(rate, 16000, num_channels=1)
    out = []
    step = int(rate * FRAME_S)
    for i in range(0, len(pcm), step):
        chunk = pcm[i : i + step]
        out += rs.push(rtc.AudioFrame(chunk.tobytes(), rate, 1, len(chunk)))
    out += rs.flush()
    # re-cut into 20 ms frames: transcribe_stream paces one frame per 20 ms
    data = b"".join(bytes(f.data) for f in out)
    size = 320 * 2
    return [
        rtc.AudioFrame(data[i : i + size].ljust(size, b"\0"), 16000, 1, 320)
        for i in range(0, len(data), size)
    ]


def load(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16), w.getframerate()


async def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--calls", type=int, default=3, help="how many of the latest recorded calls")
    ap.add_argument("--lang", default="hi", help="language hint for the delay timer only")
    a = ap.parse_args()
    files = a.files or sorted((ROOT / "logs" / "audio").glob("*-caller.wav"))[-a.calls :]
    if not files:
        print("no recordings in logs/audio: make a test call with the phone-test starter first")
        return 1
    items, skipped = usable_items("stt", _chain_env("stt"))
    if skipped:
        print(f"skipped (missing keys): {skipped}")
    report: dict = {"at": time.time(), "calls": []}
    totals = {i.spec: {"missed": 0, "delays": []} for i in items}
    count = 0
    async with utils.http_context.open():
        stts = {i.spec: _make_stt(i, []) for i in items}
        for path in files:
            pcm, rate = load(path)
            segs = utterances(pcm, rate)
            print(f"\n=== {path.name}: {len(segs)} caller sentences")
            call = {"file": path.name, "utterances": []}
            for start, seg in segs:
                frames = to_frames_16k(seg, rate)
                row = {"at_s": round(start, 1), "length_s": round(len(seg) / rate, 1)}
                print(f"\n  at {start:5.1f}s ({len(seg) / rate:.1f}s of speech)")
                for spec, stt_obj in stts.items():
                    try:
                        text, delay = await transcribe_stream(stt_obj, frames, a.lang)
                    except Exception as e:
                        text, delay = f"<error: {e}>", None
                    if not text.strip():
                        totals[spec]["missed"] += 1
                    if delay is not None:
                        totals[spec]["delays"].append(delay)
                    row[spec] = {"text": text, "final_after_s": delay}
                    shown = text if text.strip() else "(NOTHING HEARD)"
                    d = f"{delay * 1000:5.0f} ms" if delay is not None else "    -   "
                    print(f"    {spec:34s} {d}  {shown[:110]}")
                count += 1
                call["utterances"].append(row)
            report["calls"].append(call)
        for s in stts.values():
            await s.aclose()
    print(f"\n=== summary over {count} caller sentences")
    for spec, t in totals.items():
        med = statistics.median(t["delays"]) * 1000 if t["delays"] else float("nan")
        print(
            f"  {spec:34s} heard nothing: {t['missed']:3d}   final transcript after speech end: median {med:.0f} ms"
        )
    report["summary"] = {
        k: {
            "missed": v["missed"],
            "median_final_s": statistics.median(v["delays"]) if v["delays"] else None,
        }
        for k, v in totals.items()
    }
    out = ROOT / "bench" / "results" / f"stt-compare-{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
