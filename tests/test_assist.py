"""Agent assist: Exotel's stream of a call a person answers -> transcript with speakers ->
Claude's suggested reply, facts and booking details on the dashboard."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from types import SimpleNamespace

import pytest
from aiohttp import WSMsgType
from livekit.agents.stt import SpeechEventType
from livekit.agents.types import TimedString
from test_dashboard import FakeJudge, feed_in

import agent_assist as aa
import assist_brain as ab
import dashboard_hub as dh
import exotel_bridge as eb


# -- the stream from Exotel ---------------------------------------------------------------------
def word(text, spk):
    return TimedString(text, speaker_id=spk)


def final(text, words=None, speaker=None):
    alt = SimpleNamespace(text=text, words=words, speaker_id=speaker, language="en")
    return SimpleNamespace(type=SpeechEventType.FINAL_TRANSCRIPT, alternatives=[alt])


def test_one_final_with_two_voices_becomes_two_lines():
    words = [
        word("Demo", "0"),
        word("Eye", "0"),
        word("Clinic", "0"),
        word("Hello", "1"),
        word("ji", "1"),
    ]
    assert aa.split_by_speaker(final("x", words).alternatives[0]) == [
        ("0", "Demo Eye Clinic"),
        ("1", "Hello ji"),
    ]
    # no diarization: one line, no speaker
    assert aa.split_by_speaker(final("Hello there").alternatives[0]) == [(None, "Hello there")]


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


class FakeWS:
    def __init__(self, msgs):
        self.msgs = msgs

    def __aiter__(self):
        async def gen():
            for m in self.msgs:
                await asyncio.sleep(0.01)
                yield SimpleNamespace(type=WSMsgType.TEXT, data=json.dumps(m))

        return gen()


class Sink:
    def __init__(self):
        self.sent = []

    def send(self, sid, kind, **d):
        self.sent.append((sid, kind, d))


async def test_exotel_stream_becomes_lines_with_speakers_on_the_dashboard():
    pcm = base64.b64encode(b"\x00\x01" * 800).decode()
    msgs = [
        {"event": "connected"},
        {
            "event": "start",
            "start": {
                "stream_sid": "st1",
                "call_sid": "CA123",
                "from": "09876512345",
                "to": "09500000000",
                "media_format": {"sample_rate": "8000"},
            },
        },
        {"event": "media", "media": {"payload": pcm}},
        {"event": "stop", "stop": {"reason": "callended"}},
    ]
    words = [
        word("Demo", "0"),
        word("Eye", "0"),
        word("Clinic", "0"),
        word("my", "1"),
        word("number", "1"),
        word("9876512345", "1"),
    ]
    stream = FakeSTTStream([final("x", words)])
    sink = Sink()
    call = aa.AssistCall(
        FakeWS(msgs),
        eb.parse_start,
        make_stt=lambda http, diarize=True: SimpleNamespace(stream=lambda: stream),
        sink=sink,
    )
    reason = await call.run()
    assert reason == "callended"
    kinds = [k for _, k, _ in sink.sent]
    assert kinds == ["session_start", "assist_segment", "assist_segment"]
    assert sink.sent[0][0] == "assist-CA123" and sink.sent[0][2]["channel"] == "assist"
    assert sink.sent[0][2]["caller"].endswith("2345") and "98765" not in sink.sent[0][2]["caller"]
    segs = [d for _, k, d in sink.sent if k == "assist_segment"]
    assert (segs[0]["speaker"], segs[0]["text"]) == ("0", "Demo Eye Clinic")
    assert "9876512345" not in segs[1]["text"]  # numbers masked before the dashboard
    assert stream.frames == 1


def test_bridge_serves_the_assist_stream():
    app = eb.make_app()
    assert any(
        getattr(r.resource, "canonical", "") == "/exotel/assist" for r in app.router.routes()
    )


# -- Claude's part ----------------------------------------------------------------------------
def test_suggested_words_show_before_the_whole_answer_is_written():
    assert ab.early_fields('{"say": "Ji zaroor, aapka naam?", "kind": "ask", "wh') == {
        "say": "Ji zaroor, aapka naam?",
        "kind": "ask",
    }
    assert ab.early_fields('{"say": "Ji zar') == {}


def test_suggestion_is_bounded_and_cleaned():
    out = ab.clean_suggestion(
        {
            "say": "x" * 900,
            "kind": "nonsense",
            "facts": [{"title": "A", "text": "b"}] * 5,
            "form": {"name": "Ramesh", "age": 70, "junk": "y"},
            "speakers": {"0": "agent", "1": "doctor"},
            "alert": {"level": "critical", "text": "Sudden loss of vision"},
        }
    )
    assert len(out["say"]) == 400 and out["kind"] == "answer" and len(out["facts"]) == 3
    assert (
        out["form"]["name"] == "Ramesh" and out["form"]["age"] == "70" and "junk" not in out["form"]
    )
    assert out["speakers"] == {"0": "agent"} and out["alert"]["level"] == "critical"


def test_prompt_carries_the_clinic_facts_and_safety_rules(real_cfg):
    sys_prompt = ab._system(real_cfg)
    assert real_cfg.clinic.name in sys_prompt and "Green Park" in sys_prompt
    assert (
        "Never suggest a diagnosis" in sys_prompt and real_cfg.emergency.advice[:40] in sys_prompt
    )


class FakeClaude:
    """messages.stream / messages.create stand-in that writes `answer` in pieces."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = []
        self.messages = self

    def stream(self, **kw):
        self.calls.append(kw)
        answer = self.answer

        class St:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            @property
            def text_stream(self):
                async def gen():
                    for i in range(0, len(answer), 7):
                        yield answer[i : i + 7]

                return gen()

            async def get_final_message(self):
                return SimpleNamespace(
                    content=[SimpleNamespace(text=answer)],
                    usage=SimpleNamespace(
                        input_tokens=100, output_tokens=50, cache_read_input_tokens=9000
                    ),
                )

        return St()

    async def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(
            content=[SimpleNamespace(text=self.answer)],
            usage=SimpleNamespace(input_tokens=100, output_tokens=50),
        )


SUGGESTION = {
    "say": "Ji zaroor. Kya aapke pitaji pehli baar aa rahe hain?",
    "kind": "ask",
    "why": "Needs an appointment for his father",
    "alert": None,
    "facts": [{"title": "Green Park clinic", "text": "A-14 Green Park"}],
    "form": {"patient_type": "new", "name": "Ramesh Gupta", "age": 70, "centre": None},
    "speakers": {"0": "agent", "1": "patient"},
    "lines": {},
}


async def test_brain_streams_the_words_then_the_rest(real_cfg):
    claude = FakeClaude(json.dumps(SUGGESTION, ensure_ascii=False))
    brain = ab.AssistBrain(cfg=real_cfg, client=claude, model="claude-sonnet-5-5")
    early = []
    out, usd = await brain.suggest(
        [{"id": "s1", "speaker": "1", "text": "mujhe papa ke liye appointment chahiye"}],
        {},
        "Monday",
        on_say=early.append,
    )
    assert early and early[0]["say"] == SUGGESTION["say"]
    assert out["form"]["name"] == "Ramesh Gupta" and usd > 0
    assert claude.calls[0]["extra_body"] == {"output_config": {"effort": "low"}}
    assert claude.calls[0]["system"][0]["cache_control"] == {"type": "ephemeral"}


# -- the hub: the agent's screen -----------------------------------------------------------------
class FakeBrain:
    ready = True

    def __init__(self, answer=None, wrap=None):
        self.answer = answer or ab.clean_suggestion(SUGGESTION)
        self.wrap = wrap
        self.calls = []

    async def suggest(self, lines, form, today, on_say=None):
        self.calls.append([ln["id"] for ln in lines])
        if on_say:
            on_say({"say": self.answer["say"]})
        return self.answer, 0.001

    async def wrap_up(self, lines):
        return self.wrap, 0.0005


def assist_hub(tmp_path, brain):
    return dh.Hub(judge=FakeJudge(mood=lambda t: None), store=tmp_path / "s", brain=brain)


async def test_agent_sees_a_suggestion_the_details_heard_and_the_script(tmp_path):
    brain = FakeBrain(
        wrap={
            "summary": "Son booked for his father.",
            "outcome": "Appointment request",
            "follow_up": "",
        }
    )
    h = assist_hub(tmp_path, brain)
    await feed_in(
        h,
        "assist-CA1",
        [
            ("session_start", {"channel": "assist", "direction": "inbound", "persona": "Agent"}),
            (
                "assist_segment",
                {"id": "s1", "text": "Demo Eye Clinic, how may I help you?", "speaker": "0"},
            ),
            (
                "assist_segment",
                {"id": "s2", "text": "I need an appointment for my father.", "speaker": "1"},
            ),
        ],
    )
    s = h.sessions["assist-CA1"]
    assert s.checklist.name == "Inbound call (Demo Clinic front desk SOP, agent)"
    assert [t["role"] for t in s.turns] == ["bot", "user"]  # first voice = the agent
    assert s.items["greeting"]["status"] == "done" and s.items["how_help"]["status"] == "done"
    assert s.assist["say"].startswith("Ji zaroor") and not s.assist["partial"]
    assert s.assist_form["name"] == "Ramesh Gupta"
    assert s.items["name"]["status"] == "done" and s.items["age"]["status"] == "done"
    j = s.to_json()
    assert (
        j["assist"]["facts"][0]["title"] == "Green Park clinic" and j["assist_form"]["age"] == "70"
    )
    await h.end(s, "call ended")
    assert (
        s.outcome == "Appointment request" and s.summary()["note"] == "Son booked for his father."
    )


async def test_claude_corrects_who_is_who(tmp_path):
    answer = ab.clean_suggestion({**SUGGESTION, "speakers": {"0": "patient", "1": "agent"}})
    h = assist_hub(tmp_path, FakeBrain(answer))
    await feed_in(
        h,
        "assist-CA2",
        [
            ("session_start", {"channel": "assist", "direction": "inbound"}),
            ("assist_segment", {"id": "s1", "text": "Hello, is this Demo Clinic?", "speaker": "0"}),
            (
                "assist_segment",
                {
                    "id": "s2",
                    "text": "Yes, Demo Eye Clinic, how may I help?",
                    "speaker": "1",
                },
            ),
        ],
    )
    s = h.sessions["assist-CA2"]
    assert [t["role"] for t in s.turns] == ["user", "bot"]


async def test_an_emergency_alerts_once_not_on_every_line(tmp_path):
    answer = ab.clean_suggestion(
        {
            **SUGGESTION,
            "kind": "emergency",
            "alert": {"level": "critical", "text": "Sudden vision loss"},
        }
    )
    h = assist_hub(tmp_path, FakeBrain(answer))
    events = [("session_start", {"channel": "assist", "direction": "inbound"})]
    for n in range(3):
        events.append(
            ("assist_segment", {"id": f"s{n}", "text": f"line {n}", "speaker": str(n % 2)})
        )
    await feed_in(h, "assist-CA3", events[:2])
    await feed_in(h, "assist-CA3", events[2:])
    s = h.sessions["assist-CA3"]
    assert s.assist["kind"] == "emergency"
    assert [f["level"] for f in s.flags] == ["critical"]


async def test_no_claude_means_no_suggestions_but_the_call_still_shows(tmp_path):
    h = assist_hub(tmp_path, SimpleNamespace(ready=False))
    await feed_in(
        h,
        "assist-CA4",
        [
            ("session_start", {"channel": "assist", "direction": "inbound"}),
            (
                "assist_segment",
                {"id": "s1", "text": "Demo Eye Clinic, hello", "speaker": "0"},
            ),
        ],
    )
    s = h.sessions["assist-CA4"]
    assert s.assist is None and s.turns[0]["text"] == "Demo Eye Clinic, hello"


@pytest.fixture
def real_cfg():
    from config import load_config

    return load_config()


def test_demo_has_an_assist_call():
    import dashboard_demo

    assert dashboard_demo.ASSIST[0][0] == "0" and time.time()


async def test_without_speaker_labels_the_transcript_still_flows():
    """The recogniser refuses diarization: retry plain, the lines still reach the dashboard."""

    class Refusing(FakeSTTStream):
        async def __anext__(self):
            raise RuntimeError("diarize not supported")

    pcm = base64.b64encode(b"\x00\x01" * 800).decode()
    msgs = [
        {"event": "start", "start": {"call_sid": "CA9", "from": "", "media_format": {}}},
        {"event": "media", "media": {"payload": pcm}},
        {"event": "media", "media": {"payload": pcm}},
        {"event": "stop", "stop": {"reason": "callended"}},
    ]
    plain = FakeSTTStream([final("Hello, Demo Eye Clinic")])
    made = []

    def make(http, diarize=True):
        made.append(diarize)
        return SimpleNamespace(stream=lambda: Refusing([]) if diarize else plain)

    sink = Sink()
    await aa.AssistCall(FakeWS(msgs), eb.parse_start, make_stt=make, sink=sink).run()
    assert made == [True, False]
    segs = [d for _, k, d in sink.sent if k == "assist_segment"]
    assert (
        segs and segs[0]["speaker"] is None and segs[0]["text"] == "Hello, Demo Eye Clinic"
    )


# -- general knowledge, and a bot call the agent watches --------------------------------------------
KNOWLEDGE = {
    "question": "LASIK cost",
    "answer": "LASIK typically costs Rs 20,000 to 60,000 per eye in India.",
    "points": ["Depends on the laser", "x", "y", "z", "dropped"],
    "approximate": True,
    "confirm": "the clinic's exact LASIK price",
}


def test_general_knowledge_is_cleaned_and_bounded():
    out = ab.clean_suggestion({"say": "x", "knowledge": {**KNOWLEDGE, "answer": "a" * 900}})
    k = out["knowledge"]
    assert len(k["answer"]) <= 523 and len(k["points"]) == 4 and k["approximate"] is True
    # a long answer ends after a whole sentence, never in the middle of a word
    long = "LASIK reshapes the cornea. " * 30
    assert ab.clean_suggestion({"knowledge": {**KNOWLEDGE, "answer": long}})["knowledge"][
        "answer"
    ].endswith("cornea.")
    assert ab.clean_suggestion({"say": "x", "knowledge": {"question": "q"}})["knowledge"] is None
    assert ab.clean_suggestion({"say": "x"})["knowledge"] is None


def test_prompt_allows_general_knowledge_but_not_invented_clinic_facts(real_cfg):
    sys_prompt = ab._system(real_cfg)
    assert "General knowledge IS allowed" in sys_prompt and '"knowledge"' in sys_prompt
    assert "Never invent the clinic's own fee" in sys_prompt


async def test_two_calls_run_together_and_the_words_do_not_wait_for_the_knowledge(real_cfg):
    answer = {**SUGGESTION, "knowledge": KNOWLEDGE}
    claude = FakeClaude(json.dumps(answer, ensure_ascii=False))
    brain = ab.AssistBrain(cfg=real_cfg, client=claude, model="claude-haiku-5-5")
    early = []
    out, usd = await brain.suggest(
        [{"id": "s1", "speaker": "1", "text": "How much is LASIK?"}],
        {},
        "Monday",
        on_say=early.append,
    )
    assert len(claude.calls) == 2 and usd > 0
    asks = [c["messages"][0]["content"] for c in claude.calls]
    assert any('ONLY the keys "say"' in a for a in asks) and any('"knowledge"' in a for a in asks)
    assert all(c["system"][0]["cache_control"] == {"type": "ephemeral"} for c in claude.calls)
    # the last early update carries why / kind / alert: the screen has them before the knowledge
    assert set(early[-1]) == {"say", "kind", "why", "alert"}
    assert out["knowledge"]["approximate"] is True and out["form"]["name"] == "Ramesh Gupta"


async def test_a_failed_knowledge_call_still_gives_the_words():
    class Half(FakeClaude):
        def stream(self, **kw):
            if 'ONLY the keys "knowledge"' in kw["messages"][0]["content"]:
                raise RuntimeError("overloaded")
            return super().stream(**kw)

    claude = Half(json.dumps(SUGGESTION, ensure_ascii=False))
    brain = ab.AssistBrain(cfg=None, client=claude, model="claude-haiku-5-5")
    brain.system = "x"
    out, _ = await brain.suggest([{"id": "s1", "speaker": "1", "text": "hello"}], {}, "Monday")
    assert out["say"] == SUGGESTION["say"] and out["knowledge"] is None


async def test_a_bot_call_gets_suggestions_and_keeps_its_own_speakers(tmp_path):
    answer = ab.clean_suggestion(
        {
            **SUGGESTION,
            "knowledge": KNOWLEDGE,
            "speakers": {"0": "patient"},
            "lines": {"u1": "agent"},
        }
    )
    brain = FakeBrain(answer)
    h = assist_hub(tmp_path, brain)
    await feed_in(
        h,
        "exotel-CA9",
        [
            ("session_start", {"channel": "phone", "direction": "inbound", "persona": "Clara"}),
            ("bot_text", {"id": "b1", "text": "Welcome to Demo Eye Clinic."}),
            ("user_text", {"id": "u1", "text": "How much does LASIK cost?"}),
        ],
    )
    s = h.sessions["exotel-CA9"]
    assert brain.calls == [["b1", "u1"]]
    assert s.assist["knowledge"]["answer"].startswith("LASIK") and not s.assist["partial"]
    assert [t["role"] for t in s.turns] == ["bot", "user"]  # Claude's labels are not used here
    # an outbound call is never assisted
    await feed_in(
        h,
        "exotel-OUT1",
        [
            ("session_start", {"channel": "phone", "direction": "outbound", "persona": "Clara"}),
            ("user_text", {"id": "u1", "text": "Yes, tell me."}),
        ],
    )
    assert len(brain.calls) == 1 and h.sessions["exotel-OUT1"].assist is None


async def test_the_last_question_s_knowledge_is_not_left_under_the_next_question(tmp_path):
    shown = []

    class Slow(FakeBrain):
        async def suggest(self, lines, form, today, on_say=None):
            if len(self.calls) == 1:  # the second question: words only, then wait
                on_say({"say": "Next words"})
                shown.append(dict(h.sessions["exotel-CA7"].assist))
            self.calls.append([ln["id"] for ln in lines])
            return self.answer, 0.001

    brain = Slow(ab.clean_suggestion({**SUGGESTION, "knowledge": KNOWLEDGE}))
    brain.calls.append(["first"])  # the first question already answered
    h = assist_hub(tmp_path, brain)
    await feed_in(
        h,
        "exotel-CA7",
        [
            ("session_start", {"channel": "phone", "direction": "inbound", "persona": "Clara"}),
            ("user_text", {"id": "u1", "text": "What types of lenses are there?"}),
        ],
    )
    assert shown and shown[0]["say"] == "Next words" and shown[0]["knowledge"] is None
    assert shown[0]["facts"] == [] and shown[0]["partial"] is True
