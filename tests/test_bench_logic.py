"""The benchmark's own measuring logic must be right, or its numbers mean nothing."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))

from latency import error_rate
from voice_call import FRAME, SR, SpeechTracker, false_stop, stop_latency

DT = FRAME / SR  # 20 ms


def feed(tracker: SpeechTracker, pattern: list[tuple[float, bool]], t0: float = 0.0) -> float:
    """pattern: (seconds, loud?) runs."""
    t = t0
    for dur, loud in pattern:
        for _ in range(round(dur / DT)):
            tracker.push(t, 2000.0 if loud else 10.0)
            t += DT
    return t


def test_tracker_segments_and_pause_tolerance():
    tr = SpeechTracker()
    # 1 s speech, 0.2 s pause (sentence gap), 1 s speech, 1 s silence
    end = feed(tr, [(1.0, True), (0.2, False), (1.0, True), (1.0, False)])
    assert len(tr.segments) == 1, "a 200 ms gap is not the end of speech"
    s, e = tr.segments[0]
    assert abs(s - 0.0) < 0.03 and abs(e - 2.2) < 0.03
    assert not tr.speaking and end > e


def test_tracker_two_segments_after_long_gap():
    tr = SpeechTracker()
    feed(tr, [(0.5, True), (0.6, False), (0.5, True), (0.5, False)])
    assert len(tr.segments) == 2
    assert abs(tr.first_start_after(0.6) - 1.1) < 0.03


def test_tracker_ignores_single_click():
    tr = SpeechTracker()
    feed(tr, [(0.02, True), (0.5, False)])
    assert tr.segments == []


def test_stop_latency_true_interruption():
    # agent talks 0-3 s, caller barges in at 1.5-2.5 s, agent stops at 1.75 and answers at 3.4
    segs = [[0.0, 1.75], [3.4, 5.0]]
    assert abs(stop_latency(segs, 1.5, 2.5, until=6.0) - 0.25) < 1e-9


def test_stop_latency_ignores_sentence_pause_and_detects_miss():
    # agent keeps talking through the barge-in with only a 0.3 s sentence pause
    segs = [[0.0, 1.8], [2.1, 6.0]]
    assert stop_latency(segs, 1.5, 2.5, until=7.0) is None


def test_backchannel_false_stop():
    kept_talking = [[0.0, 2.0], [2.3, 5.0]]  # 0.3 s pause = normal
    assert not false_stop(kept_talking, 1.5, until=6.0)
    stopped = [[0.0, 1.7], [3.5, 5.0]]  # went quiet 1.8 s after "hmm"
    assert false_stop(stopped, 1.5, until=6.0)


def test_error_rates():
    assert error_rate("book an appointment", "book an appointment", "en") == 0
    assert (
        abs(error_rate("book an appointment please", "book appointment please", "en") - 0.25) < 1e-9
    )
    assert error_rate("मुझे अपॉइंटमेंट चाहिए", "मुझे अपॉइंटमेंट चाहिए।", "hi") == 0
