"""The live dashboard: hub analysis, scorecards, the feed that must never slow a call."""

from __future__ import annotations

import asyncio
import json
import time

import numpy as np
from aiohttp.test_utils import TestClient, TestServer

import dashboard_hub as dh
from dashboard_feed import Feed


class FakeJudge:
    """Claude stand-in: scripted answers, no network."""

    def __init__(self, mood=None, script=None):
        self._mood, self._script = mood, script
        self.ready = mood is not None or script is not None
        self.script_calls = []

    async def mood(self, context, text):
        return (self._mood(text) if self._mood else None), 0.00005

    async def script(self, context, reply, steps, rules):
        self.script_calls.append(([i.id for i in steps], [i.id for i in rules]))
        return (self._script(reply) if self._script else None), 0.00008


def hub(tmp_path, **judge):
    return dh.Hub(judge=FakeJudge(**judge), store=tmp_path / "sessions")


async def feed_in(h, sid, events):
    for kind, data in events:
        await h.ingest({"session": sid, "type": kind, "at": time.time(), **data})
    for s in h.sessions.values():  # let background analysis finish
        if s.tasks:
            await asyncio.wait(list(s.tasks), timeout=5)


OUTBOUND = [
    (
        "session_start",
        {"channel": "phone", "direction": "outbound", "persona": "Clara", "caller": "xxxxxx2210"},
    ),
    (
        "bot_text",
        {
            "id": "b1",
            "text": "Namaste, good evening. This call is from Demo Eye and Retina Clinic. "
            "This is Clara speaking, a virtual assistant and part of Dr. Arjun Mehta's team. We "
            "received your enquiry through our website, so how may I help you?",
        },
    ),
    ("user_text", {"id": "u1", "text": "Yes, for my father's check-up."}),
    (
        "tool",
        {
            "name": "update_patient_details",
            "args": {"name": "Suresh", "age": 68, "city": "Saket", "concern": "blurred vision"},
            "output": "Saved.",
        },
    ),
    (
        "bot_text",
        {"id": "b2", "text": "Thank you. Can I book an appointment with the doctor for you?"},
    ),
    ("user_text", {"id": "u2", "text": "Yes, Monday morning, on this number."}),
    (
        "tool",
        {
            "name": "request_appointment",
            "args": {"preferred_day": "Monday", "use_calling_number": True},
            "output": "REQUEST REGISTERED SEC-R1",
        },
    ),
    (
        "bot_text",
        {
            "id": "b3",
            "text": "Please bring your previous eye reports and kindly come about fifteen minutes "
            "early. Is there anything else I may assist you with?",
        },
    ),
    ("user_text", {"id": "u3", "text": "No, thank you so much."}),
    (
        "bot_text",
        {
            "id": "b4",
            "text": "Thank you for your time. You were speaking with Clara from Demo Eye and "
            "Retina Clinic. Take care, and have a good day.",
        },
    ),
]


async def test_outbound_script_ticks_from_words_and_tools_and_is_scored(tmp_path):
    h = hub(tmp_path)
    await feed_in(h, "call-1", OUTBOUND)
    s = h.sessions["call-1"]
    st = {k: v["status"] for k, v in s.items.items()}
    steps = (
        "greeting",
        "call_from",
        "intro",
        "source",
        "how_help",
        "name",
        "age",
        "city",
        "complaint",
        "contact",
        "offer_booking",
        "registered",
        "confirmation",
        "anything_else",
        "closing",
    )
    for k in steps:
        assert st[k] == "done", k
    assert st["verify"] == "open"  # needs Claude: still open while the call runs
    await h.end(s, "caller hung up")
    st = {k: v["status"] for k, v in s.items.items()}
    assert st["verify"] == "missed" and st["bad_time"] == "na" and st["no_diagnosis"] == "kept"
    assert st["price_rule"] == "kept"
    summary = h.history[-1]
    assert summary["outcome"] == "Appointment request"
    assert summary["sop"] == {**summary["sop"], "done": len(steps), "total": len(steps) + 1}
    saved = list((tmp_path / "sessions").glob("*.json"))
    assert (
        len(saved) == 1
        and json.loads(saved[0].read_text(encoding="utf-8"))["summary"]["id"] == "call-1"
    )
    # a fresh hub reads the scorecard back for History
    assert hub(tmp_path).history[-1]["id"] == "call-1"


async def test_booking_steps_are_not_needed_when_nobody_asked_to_book(tmp_path):
    h = hub(tmp_path)
    await feed_in(
        h,
        "chat-1",
        [
            ("session_start", {"channel": "whatsapp", "direction": "inbound"}),
            (
                "bot_text",
                {
                    "text": "Namaste, welcome to Demo Eye and Retina Clinic. I'm Clara, a virtual assistant from Dr. Arjun Mehta's team. How may I help you today?"
                },
            ),
            ("user_text", {"text": "Where is the clinic?"}),
            ("bot_text", {"text": "It is at A-14 Green Park. Is there anything else?"}),
        ],
    )
    s = h.sessions["chat-1"]
    await h.end(s, "chat finished")
    assert s.items["registered"]["status"] == "na" and s.items["age"]["status"] == "na"
    # Demo Clinic SOP: always take a name and number, even when nobody books
    assert s.items["name"]["status"] == "missed" and s.items["contact"]["status"] == "missed"
    assert s.items["closing"]["status"] == "missed"
    assert h.history[-1]["outcome"] == "Enquiry"


async def test_claude_ticks_paraphrases_and_flags_a_broken_rule(tmp_path):
    def script(reply):
        if "person who enquired" in reply:
            return {"done": ["verify"], "broken": []}
        if "cataract" in reply:
            return {"done": [], "broken": [{"id": "no_diagnosis", "why": "named a condition"}]}
        return {"done": [], "broken": []}

    h = hub(tmp_path, script=script)
    alerts = []
    h._send_all = _collect(alerts)
    await feed_in(
        h,
        "call-2",
        [
            ("session_start", {"direction": "outbound"}),
            ("bot_text", {"text": "Am I speaking with the person who enquired?"}),
            ("bot_text", {"text": "That sounds like cataract to me."}),
        ],
    )
    s = h.sessions["call-2"]
    assert s.items["verify"]["status"] == "done"
    assert s.items["no_diagnosis"]["status"] == "broken" and s.items["no_diagnosis"]["why"]
    assert any(a.get("type") == "alert" and "Rule broken" in a["alert"]["text"] for a in alerts)
    # Claude is only asked about steps it can judge, and rules not broken yet
    steps, rules = h.judge.script_calls[-1]
    assert "verify" not in steps and "no_diagnosis" in rules


def _collect(sink):
    async def send(msg):
        sink.append(msg)

    return send


async def test_mood_uses_the_voice_against_the_callers_own_normal(tmp_path):
    h = hub(tmp_path)  # no Claude: mood from the words
    alerts = []
    h._send_all = _collect(alerts)
    events = [("session_start", {"direction": "inbound"})]
    for level, text in [
        (-24, "I need an appointment"),
        (-25, "For my mother"),
        (-24, "She is seventy"),
    ]:
        events += [
            ("utterance", {"dur_s": 2.0, "mean_db": level, "peak_db": level + 8}),
            ("user_text", {"text": text}),
        ]
    events += [
        ("utterance", {"dur_s": 2.4, "mean_db": -15, "peak_db": -6}),
        ("user_text", {"text": "Why are you asking so many questions? This is useless!"}),
    ]
    await feed_in(h, "call-3", events)
    s = h.sessions["call-3"]
    last = s.turns[-1]
    assert last["voice"]["raised"] and last["voice"]["rel_db"] >= 6  # ~9 dB above their normal
    assert last["mood"]["score"] < 0 and last["mood"]["by"] == "words"
    assert s.mood_now()["arousal"] > 0.7
    assert any(a.get("type") == "alert" for a in alerts)
    assert s.turns[0]["voice"]["rel_db"] == 0.0  # the first utterance sets the normal


async def test_browser_gets_a_snapshot_then_live_updates(tmp_path):
    app = dh.make_app(hub(tmp_path))
    async with TestClient(TestServer(app)) as c:
        ws = await c.ws_connect("/ws")
        hello = json.loads((await ws.receive()).data)
        assert hello["type"] == "hello" and set(hello["checklists"]) == {"inbound", "outbound"}
        r = await c.post(
            "/events",
            json=[
                {"session": "call-9", "type": "session_start", "direction": "inbound"},
                {"session": "call-9", "type": "voice", "level": -20, "speaking": True},
            ],
        )
        assert r.status == 200
        got = {}
        for _ in range(4):
            m = json.loads((await asyncio.wait_for(ws.receive(), 3)).data)
            got[m["type"]] = m
            if {"voice", "session"} <= set(got):
                break
        assert got["voice"]["level"] == -20 and got["session"]["session"]["id"] == "call-9"
        page = await c.get("/")
        assert page.status == 200 and "Front desk monitor" in await page.text()
        assert (await c.get("/static/three.module.min.js")).status == 200


async def test_feed_never_waits_when_the_dashboard_is_down(monkeypatch):
    monkeypatch.setenv("DASHBOARD", "1")
    f = Feed("http://127.0.0.1:9/events")  # nothing listens there
    t0 = time.perf_counter()
    for i in range(500):
        f.send("call", "voice", level=-30, n=i)
    assert time.perf_counter() - t0 < 0.2  # sending is just a queue put
    await f.drain(timeout=1.0)
    assert f.sent == 0


async def test_feed_delivers_batches_to_the_hub(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHBOARD", "1")
    h = hub(tmp_path)
    async with TestServer(dh.make_app(h)) as server:
        f = Feed(str(server.make_url("/events")))
        f.send("call-5", "session_start", direction="outbound", persona="Clara")
        f.send("call-5", "bot_text", text="This call is from Demo Eye and Retina Clinic.")
        await f.drain()
        for _ in range(20):
            if "call-5" in h.sessions and h.sessions["call-5"].turns:
                break
            await asyncio.sleep(0.05)
    s = h.sessions["call-5"]
    assert s.persona == "Clara" and s.items["call_from"]["status"] == "done"


def test_voice_meter_reports_levels_and_whole_utterances():
    import exotel_bridge as eb

    now = [0.0]
    sent = []
    timing = eb.SpeechTiming(clock=lambda: now[0])
    timing.noise_floor = 40.0
    meter = eb.VoiceMeter(timing, lambda kind, **d: sent.append((kind, d)), clock=lambda: now[0])
    rate, chunk = 8000, 160
    t = np.arange(chunk) / rate
    speech = (np.sin(2 * np.pi * 200 * t) * 4000).astype(np.int16).tobytes()
    quiet = (np.random.default_rng(1).normal(0, 20, chunk)).astype(np.int16).tobytes()
    for _ in range(100):  # 2 s of speech
        meter.audio(speech)
        now[0] += 0.02
    for _ in range(50):  # 1 s of quiet ends it
        meter.audio(quiet)
        now[0] += 0.02
    kinds = [k for k, _ in sent]
    voice = [d for k, d in sent if k == "voice"]
    assert kinds.count("utterance") == 1
    # ~5 levels a second while they speak, then a single "stopped": no stream of silence
    assert 9 <= len(voice) <= 12 and voice[-1]["speaking"] is False
    assert sum(1 for d in voice if not d["speaking"]) == 1
    utt = next(d for k, d in sent if k == "utterance")
    assert 1.8 <= utt["dur_s"] <= 2.1 and -22 < utt["mean_db"] < -15
    assert any(d["speaking"] for k, d in sent if k == "voice")


async def test_phone_tap_sends_what_happened_without_the_language_note():
    from livekit.agents import ConversationItemAddedEvent, FunctionToolsExecutedEvent, llm

    from call_state import CallState
    from dashboard_tap import DashboardTap

    class Sink:
        def __init__(self):
            self.events = []

        def send(self, sid, kind, **d):
            self.events.append((sid, kind, d))

        async def drain(self, timeout=2.0):
            pass

        async def aclose(self):
            pass

    class Session:
        def __init__(self):
            self.handlers = {}

        def on(self, name, fn):
            self.handlers[name] = fn

    sink, sess = Sink(), Session()
    state = CallState(room="exotel-abc", caller_number="+919876543210", persona="Clara")
    tap = DashboardTap(sess, state, sink=sink)
    user = llm.ChatMessage(
        role="user",
        content=[
            "mujhe appointment chahiye",
            "[Reply language for this turn: Hindi. Time now: 11:06 PM.]",
        ],
    )
    sess.handlers["conversation_item_added"](ConversationItemAddedEvent(item=user))
    bot = llm.ChatMessage(role="assistant", content=["Ji zaroor."])
    bot.metrics = {"e2e_latency": 0.82, "llm_node_ttft": 0.5, "tts_node_ttfb": 0.2}
    sess.handlers["conversation_item_added"](ConversationItemAddedEvent(item=bot))
    call = llm.FunctionCall(
        call_id="c1",
        name="update_patient_details",
        arguments='{"name": "Ramesh", "mobile_number": "9876543210"}',
    )
    out = llm.FunctionCallOutput(
        call_id="c1", name="update_patient_details", output="Saved.", is_error=False
    )
    sess.handlers["function_tools_executed"](
        FunctionToolsExecutedEvent(function_calls=[call], function_call_outputs=[out])
    )
    await tap.end("call ended", 0.064)
    kinds = [k for _, k, _ in sink.events]
    assert kinds == ["session_start", "user_text", "bot_text", "tool", "session_end"]
    assert all(sid == "exotel-abc" for sid, _, _ in sink.events)
    assert sink.events[0][2]["caller"].endswith("3210") and "9876543210" not in json.dumps(
        sink.events
    )
    assert sink.events[1][2]["text"] == "mujhe appointment chahiye"
    assert sink.events[2][2]["latency"]["ttfa"] == 0.82
    assert sink.events[3][2]["args"]["name"] == "Ramesh"
    assert sink.events[4][2]["cost_usd"] == 0.064


async def test_tap_on_a_real_agent_session(real_cfg):
    """The event names and fields really come out of LiveKit's AgentSession."""
    from fake_llm import ScriptedLLM
    from livekit.agents import AgentSession
    from livekit.agents.testing import fake_job_context

    from call_state import CallState
    from dashboard_tap import DashboardTap
    from front_desk import FrontDeskAgent
    from scheduling import MockScheduler

    events = []

    class Sink:
        def send(self, sid, kind, **d):
            events.append((kind, d))

        async def drain(self, timeout=2.0):
            pass

        async def aclose(self):
            pass

    steps = [
        (
            "say_call",
            "Thank you. May I know the patient's age, please?",
            "update_patient_details",
            {"patient_type": "new", "name": "Ramesh"},
        ),
    ]
    state = CallState(room="exotel-xyz", caller_number="+919876543210")
    with fake_job_context():
        async with AgentSession(llm=ScriptedLLM(steps), userdata=state, max_tool_steps=5) as s:
            DashboardTap(s, state, sink=Sink())
            await s.start(FrontDeskAgent(real_cfg, scheduler=MockScheduler(real_cfg), greet=False))
            await s.run(user_input="New patient, Ramesh")
    kinds = [k for k, _ in events]
    assert "user_text" in kinds and "bot_text" in kinds and "tool" in kinds and "bot_state" in kinds
    tool = next(d for k, d in events if k == "tool")
    assert tool["name"] == "update_patient_details" and tool["args"]["name"] == "Ramesh"
