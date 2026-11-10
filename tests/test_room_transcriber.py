"""Live transcript inside LiveKit: each participant's audio -> lines labelled caller / staff."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from livekit import rtc
from livekit.agents.stt import SpeechEventType
from livekit.agents.types import TimedString
from test_dashboard import FakeJudge, feed_in

import dashboard_hub as dh
import room_transcriber as rt


class FakeSTTStream:
    def __init__(self, events):
        self.events, self.frames, self.q = events, 0, asyncio.Queue()

    def push_frame(self, frame):
        self.frames += 1
        if self.frames == 1:
            for e in self.events:
                self.q.put_nowait(e)

    def end_input(self):
        self.q.put_nowait(None)

    async def aclose(self):
        pass

    def __aiter__(self):
        return self

    async def __anext__(self):
        e = await self.q.get()
        if e is None:
            raise StopAsyncIteration
        return e


class Sink:
    def __init__(self):
        self.sent = []

    def send(self, sid, kind, **d):
        self.sent.append((sid, kind, d))


def final(text, words=None):
    alt = SimpleNamespace(text=text, words=words, language="en")
    return SimpleNamespace(type=SpeechEventType.FINAL_TRANSCRIPT, alternatives=[alt])


def interim(text):
    alt = SimpleNamespace(text=text, words=None, language="en")
    return SimpleNamespace(type=SpeechEventType.INTERIM_TRANSCRIPT, alternatives=[alt])


async def audio(n=2):
    for _ in range(n):
        yield SimpleNamespace(frame=object())
        await asyncio.sleep(0.01)


def test_who_is_who_comes_from_the_participant():
    assert rt.role_of(rtc.ParticipantKind.PARTICIPANT_KIND_SIP) == "patient"
    assert rt.role_of(rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD) == "agent"
    assert rt.role_of(rtc.ParticipantKind.PARTICIPANT_KIND_AGENT) is None  # Clara is a bot
    assert rt.role_of(rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD, "agent-xyz") is None


def test_doubtful_words_are_marked_for_the_reader():
    words = [
        TimedString("I", confidence=0.95),
        TimedString("want", confidence=0.9),
        TimedString("catarak", confidence=0.31),
        TimedString("surgery", confidence=0.92),
    ]
    alt = SimpleNamespace(text="I want catarak surgery", words=words)
    assert rt.mark_doubtful(alt, 0.6) == "I want [?catarak] surgery"
    assert rt.mark_doubtful(alt, 0) == "I want catarak surgery"  # off
    sure = SimpleNamespace(text="hello there", words=[TimedString("hello", confidence=0.9)])
    assert rt.mark_doubtful(sure, 0.6) == "hello there"  # nothing doubtful: untouched
    assert rt.mark_doubtful(SimpleNamespace(text="hi", words=None), 0.6) == "hi"


async def test_two_tracks_become_labelled_lines_and_the_call_ends():
    sink = Sink()
    streams = {
        "patient": FakeSTTStream(
            [interim("I want to"), final("I want to book a consultation, my number is 9876512345")]
        ),
        "agent": FakeSTTStream([final("Demo Eye and Retina Clinic, how can I help")]),
    }
    order = iter(["patient", "agent"])
    call = rt.RoomTranscript(
        "call-1",
        make_stt=lambda http: SimpleNamespace(stream=lambda: streams[next(order)]),
        sink=sink,
    )
    call.start("+919876512345")
    call.add_track("p/1", "patient", audio())
    await asyncio.sleep(0)  # the patient's recogniser claims its stream first
    call.add_track("a/1", "agent", audio())
    call.add_track("a/1", "agent", audio())  # the same track again: ignored
    await call.finish()
    kinds = [k for _, k, _ in sink.sent]
    assert kinds[0] == "session_start" and kinds[-1] == "session_end"
    assert sink.sent[0][0] == "assist-call-1" and sink.sent[0][2]["channel"] == "assist"
    assert "98765" not in sink.sent[0][2]["caller"] and sink.sent[0][2]["caller"].endswith("2345")
    segs = [d for _, k, d in sink.sent if k == "assist_segment"]
    assert {(d["role"], d["speaker"]) for d in segs} == {("patient", "patient"), ("agent", "agent")}
    assert any("consultation" in d["text"] for d in segs)
    assert all("9876512345" not in d["text"] for d in segs)  # numbers masked first
    assert any(k == "caption" for _, k, _ in sink.sent)  # live partial words too
    assert len({d["id"] for d in segs}) == len(segs)


async def test_hub_trusts_the_labels_and_claude_does_not_guess_speakers(tmp_path):
    h = dh.Hub(judge=FakeJudge(), store=tmp_path / "sessions")
    await feed_in(
        h,
        "assist-r1",
        [
            ("session_start", {"channel": "assist", "direction": "inbound", "persona": "Agent"}),
            (
                "assist_segment",
                {
                    "id": "s1",
                    "text": "Hello, I need a cataract consultation",
                    "speaker": "patient",
                    "role": "patient",
                },
            ),
            (
                "assist_segment",
                {
                    "id": "s2",
                    "text": "Demo Eye and Retina Clinic, namaste",
                    "speaker": "agent",
                    "role": "agent",
                },
            ),
        ],
    )
    s = h.sessions["assist-r1"]
    assert s.roles_fixed
    assert [t["role"] for t in s.turns] == ["user", "bot"]
    assert s.speaker_roles == {"patient": "user", "agent": "bot"}
