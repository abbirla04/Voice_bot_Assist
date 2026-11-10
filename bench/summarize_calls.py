"""Scorecard from real calls: aggregates logs/calls/*.json written by the agent.

python bench/summarize_calls.py            # logs/calls
python bench/summarize_calls.py path/to/calls
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _stream in (sys.stdout, sys.stderr):  # Hindi text on a cp1252 Windows console
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")
sys.path.insert(0, str(ROOT / "src"))

from telemetry import pct  # noqa: E402

TARGETS_MS = {
    "e2e_latency": (900, 1400, "TTFA: caller stops -> bot audio"),
    "llm_node_ttft": (450, 900, "LLM TTFT"),
    "tts_node_ttfb": (250, 450, "TTS TTFB"),
    "end_of_turn_delay": (350, 700, "end-of-turn decision"),
    "transcription_delay": (300, 600, "final transcript after speech end"),
}


def main() -> int:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "logs" / "calls"
    files = sorted(folder.glob("*.json"))
    if not files:
        print(f"no call reports in {folder}")
        return 1
    samples: dict[str, list[float]] = {k: [] for k in TARGETS_MS}
    outcomes = Counter()
    errors = Counter()
    interruptions = false_int = 0
    for f in files:
        r = json.loads(f.read_text(encoding="utf-8"))
        for k in TARGETS_MS:
            samples[k] += r.get("latency_samples", {}).get(k, [])
        c = r["call"]
        outcomes[
            "booked"
            if c.get("booking")
            else (
                "transferred"
                if c.get("transferred")
                else ("callback" if c.get("callbacks") else "other")
            )
        ] += 1
        if c.get("emergency_flagged"):
            outcomes["emergency"] += 1
        for e in r.get("provider_errors", []):
            errors[e["source"]] += 1
        interruptions += r.get("interruptions", {}).get("interruptions", 0)
        false_int += r.get("interruptions", {}).get("false_interruptions", 0)

    print(f"{len(files)} calls in {folder}\n")
    print("| metric | n | p50 | p95 | target p50/p95 | verdict |\n|---|---|---|---|---|---|")
    for k, (t50, t95, label) in TARGETS_MS.items():
        vs = samples[k]
        p50, p95 = pct(vs, 0.5), pct(vs, 0.95)
        if p50 is None:
            print(f"| {label} | 0 | - | - | {t50}/{t95} ms | NO DATA |")
            continue
        ok = p50 * 1000 <= t50 and p95 * 1000 <= t95
        print(
            f"| {label} | {len(vs)} | {p50 * 1000:.0f} ms | {p95 * 1000:.0f} ms | {t50}/{t95} ms | {'PASS' if ok else 'FAIL'} |"
        )
    print(f"\noutcomes: {dict(outcomes)}")
    print(
        f"caller interruptions: {interruptions}, false interruptions (cough/hmm, resumed): {false_int}"
    )
    print(f"provider errors by source: {dict(errors) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
