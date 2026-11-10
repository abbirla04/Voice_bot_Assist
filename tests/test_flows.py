"""End-to-end conversation flows through a real AgentSession, offline.

The LLM is scripted (tests/fake_llm.py) so these check OUR code: tools, hold
SOP timing, transfer fallbacks, end-of-call, provider failover. What the real
model says is covered by tests/test_evals.py (needs keys).
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from fake_llm import FailingLLM, ScriptedLLM
from livekit.agents import AgentSession, llm
from livekit.agents.testing import fake_job_context

import front_desk
from call_state import CallState
from config import HoldCfg
from front_desk import FAILED_MSG, TIMEOUT_MSG, FrontDeskAgent
from lines import LINES
from scheduling import MockScheduler

IST = ZoneInfo("Asia/Kolkata")
MONDAY_11 = datetime(2026, 10, 19, 11, 0, tzinfo=IST)


@pytest.fixture
def frozen(cfg, monkeypatch):
    monkeypatch.setattr(type(cfg), "now", lambda self: MONDAY_11)
    return cfg


def tool_outputs(result) -> list:
    return [ev.item for ev in result.events if type(ev).__name__ == "FunctionCallOutputEvent"]


def said(session) -> list[str]:
    return [m.text_content for m in session.history.messages() if m.role == "assistant"]


def spoken(session) -> list[str]:
    """Everything sent to the voice, including lines kept out of the LLM context (hold)."""
    return list(getattr(session, "_spoken", []))


def _record_say(session) -> None:
    session._spoken = []
    original = session.say

    def say(text, **kw):
        session._spoken.append(text if isinstance(text, str) else "<stream>")
        return original(text, **kw)

    session.say = say


async def run_session(cfg, steps, user_inputs, *, state=None, scheduler=None, llm_override=None):
    fake = llm_override or ScriptedLLM(steps)
    state = state or CallState(caller_number="+919876543210")
    scheduler = scheduler or MockScheduler(cfg)
    results = []
    with fake_job_context():
        async with AgentSession(llm=fake, userdata=state, max_tool_steps=5) as session:
            _record_say(session)
            await session.start(FrontDeskAgent(cfg, scheduler=scheduler, greet=False))
            for text in user_inputs:
                results.append(await session.run(user_input=text))
            return session, results, state


# -- booking -----------------------------------------------------------------------------
async def test_full_booking_flow(frozen):
    cfg = frozen
    sched = MockScheduler(cfg)
    slot = (await sched.find_slots(doctor="dr-arora"))[0]
    steps = [
        (
            "call",
            "update_patient_details",
            {
                "patient_type": "new",
                "name": "Ramesh Kumar",
                "age": 58,
                "concern": "blurry vision",
                "location_id": "noida",
            },
        ),
        ("say", "Thank you Ramesh ji. Let me check the availability."),
        ("call", "check_doctor_availability", {"location_id": "noida"}),
        ("say", "Dr. Neha Arora is available. Which time suits you?"),
        ("call", "book_appointment", {"slot_id": slot.slot_id, "use_calling_number": True}),
        ("say", "We have booked your appointment."),
    ]
    _, results, state = await run_session(
        cfg,
        steps,
        [
            "Ramesh Kumar, 58, new patient, blurry vision, Noida",
            "Any time this week",
            "First slot, use this number",
        ],
        scheduler=sched,
    )
    avail = tool_outputs(results[1])[0].output
    assert "Open slots" in avail and slot.slot_id in avail
    booked = tool_outputs(results[2])[0].output
    assert booked.startswith("BOOKED")
    # the scripted confirmation is built in code: every SOP element, exact facts
    for must in (
        "Dr. Neha Arora",
        "General Ophthalmology",
        "Noida centre",
        "Monday 19 October",
        "one to three hours",
        "fifteen hundred rupees including registration",
        "not be able to drive back",
        "Kindly report on time",
        "Is there anything else I may assist you with?",
    ):
        assert must in booked, must
    assert state.booking and state.booking["doctor"] == "Dr. Neha Arora"
    assert state.booking["phone"].endswith("3210") and "9876" not in state.booking["phone"]
    assert not state.on_hold and state.holds == []  # fast backend: no hold announced


async def test_booking_refuses_without_name_and_phone(frozen):
    cfg = frozen
    sched = MockScheduler(cfg)
    slot = (await sched.find_slots())[0]
    steps = [
        ("call", "book_appointment", {"slot_id": slot.slot_id}),
        ("say", "May I have the name?"),
    ]
    _, results, state = await run_session(
        cfg, steps, ["book it"], state=CallState(), scheduler=sched
    )
    out = tool_outputs(results[0])[0]
    assert out.is_error and "name missing" in out.output
    assert state.booking is None


async def test_double_booking_same_slot_is_safe(frozen):
    cfg = frozen
    sched = MockScheduler(cfg)
    slot = (await sched.find_slots())[0]
    state = CallState(caller_number="+919876543210")
    state.patient.name = "A"
    steps = [
        ("call", "book_appointment", {"slot_id": slot.slot_id, "use_calling_number": True}),
        ("say", "Booked."),
        ("call", "book_appointment", {"slot_id": slot.slot_id, "use_calling_number": True}),
        ("say", "Already booked."),
    ]
    _, results, _ = await run_session(
        cfg, steps, ["book", "book again"], state=state, scheduler=sched
    )
    assert "Already booked" in tool_outputs(results[1])[0].output


async def test_patient_detail_validation(frozen):
    steps = [
        (
            "call",
            "update_patient_details",
            {
                "age": 250,
                "mobile_number": "12345",
                "location_id": "goa",
                "preferred_doctor": "Dr Strange",
            },
        ),
        ("say", "Could you repeat that?"),
    ]
    _, results, state = await run_session(frozen, steps, ["..."], state=CallState())
    out = tool_outputs(results[0])[0].output
    for needle in ("age 250", "not a valid 10-digit", "unknown location", "no doctor named"):
        assert needle in out, needle
    assert state.patient.age is None and state.patient.phone is None
    assert "Still needed" in out


async def test_past_date_and_bad_location_are_rejected(frozen):
    steps = [
        ("call", "check_doctor_availability", {"date_iso": "2026-10-01"}),
        ("call", "check_doctor_availability", {"location_id": "mumbai"}),
        ("say", "Sorry."),
    ]
    _, results, _ = await run_session(frozen, steps, ["last month please"])
    outs = tool_outputs(results[0])
    assert outs[0].is_error and "past" in outs[0].output
    assert outs[1].is_error and "unknown location_id" in outs[1].output


# -- hold SOP --------------------------------------------------------------------------------
async def test_slow_lookup_triggers_hold_announcement_and_checkins(frozen, monkeypatch):
    cfg = frozen
    cfg.hold = HoldCfg(first_notice_after_s=0.2, checkin_every_s=0.4, max_hold_s=5, music_volume=0)
    monkeypatch.setenv("MOCK_SCHEDULER_LATENCY_S", "1.5")
    steps = [
        ("call", "check_doctor_availability", {"location_id": "noida"}),
        ("say", "Thank you for your patience."),
    ]
    session, results, state = await run_session(cfg, steps, ["is Dr Arora free?"])
    lines = spoken(session)
    assert LINES["hold_start"]["en"] in lines, lines
    assert LINES["hold_checkin_1"]["en"] in lines, lines
    assert lines.index(LINES["hold_start"]["en"]) < lines.index(LINES["hold_checkin_1"]["en"])
    # hold lines must stay out of the LLM context (Claude rejects a trailing assistant turn)
    assert LINES["hold_start"]["en"] not in said(session)
    out = tool_outputs(results[0])[0].output
    assert "on hold for about" in out
    assert state.holds and state.holds[0]["reason"] == "availability"
    assert not state.on_hold


async def test_hold_never_exceeds_limit(frozen, monkeypatch):
    cfg = frozen
    cfg.hold = HoldCfg(
        first_notice_after_s=0.1, checkin_every_s=0.3, max_hold_s=0.8, music_volume=0
    )
    monkeypatch.setenv("MOCK_SCHEDULER_LATENCY_S", "5")
    steps = [
        ("call", "check_doctor_availability", {}),
        ("call", "schedule_callback", {"reason": "availability"}),
        ("say", "We will call you back."),
    ]
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    _, results, state = await run_session(cfg, steps, ["any slot?"])
    elapsed = loop.time() - t0
    assert elapsed < 3, f"hold ran {elapsed:.1f}s, limit 0.8s"
    outs = tool_outputs(results[0])
    assert outs[0].output == TIMEOUT_MSG
    assert state.callbacks and "availability" in state.errors[0]


async def test_hold_in_hindi_for_hindi_caller(frozen, monkeypatch):
    cfg = frozen
    cfg.hold = HoldCfg(first_notice_after_s=0.1, checkin_every_s=5, max_hold_s=5, music_volume=0)
    monkeypatch.setenv("MOCK_SCHEDULER_LATENCY_S", "0.6")
    steps = [("call", "check_doctor_availability", {}), ("say", "धैर्य के लिए धन्यवाद।")]
    session, _, state = await run_session(cfg, steps, ["मुझे डॉक्टर से मिलना है"])
    assert state.language == "hi"
    assert LINES["hold_start"]["hi"] in spoken(session)


async def test_backend_down_gives_callback_instruction(frozen, monkeypatch):
    monkeypatch.setenv("MOCK_SCHEDULER_FAIL", "1")
    steps = [("call", "check_doctor_availability", {}), ("say", "Sorry.")]
    _, results, state = await run_session(frozen, steps, ["slots?"])
    assert tool_outputs(results[0])[0].output == FAILED_MSG
    assert state.errors


# -- transfer SOP ------------------------------------------------------------------------------
async def test_transfer_without_trunk_offers_callback(frozen, monkeypatch):
    monkeypatch.delenv("LIVEKIT_SIP_OUTBOUND_TRUNK", raising=False)
    steps = [
        ("call", "transfer_to_department", {"department_id": "billing", "summary": "refund query"}),
        ("say", "A colleague will call you."),
    ]
    _, results, state = await run_session(frozen, steps, ["I need a refund"])
    assert "not available right now" in tool_outputs(results[0])[0].output
    assert state.transfers[0]["result"] == "not_configured"


async def test_transfer_when_department_closed(cfg, monkeypatch):
    monkeypatch.setattr(
        type(cfg), "now", lambda self: datetime(2026, 10, 18, 11, 0, tzinfo=IST)
    )  # Sunday
    steps = [
        ("call", "transfer_to_department", {"department_id": "billing", "summary": "bill"}),
        ("say", "Closed today."),
    ]
    _, results, state = await run_session(cfg, steps, ["billing please"])
    assert "closed right now" in tool_outputs(results[0])[0].output
    assert state.transfers[0]["result"] == "closed"


def _enable_transfer(cfg, monkeypatch, fake_task):
    monkeypatch.setenv("LIVEKIT_SIP_OUTBOUND_TRUNK", "ST_test")
    dept = cfg.department("billing")
    dept.sip_call_to = "+911100000000"
    monkeypatch.setattr(front_desk, "WarmTransferTask", fake_task)


async def test_transfer_no_answer_returns_to_caller(frozen, monkeypatch):
    from livekit.agents import ToolError

    async def no_answer(**kw):
        await asyncio.sleep(0.05)
        raise ToolError("could not dial human agent")

    _enable_transfer(frozen, monkeypatch, no_answer)
    steps = [
        (
            "call",
            "transfer_to_department",
            {"department_id": "billing", "summary": "Ramesh, refund"},
        ),
        ("call", "schedule_callback", {"reason": "billing refund"}),
        ("say", "Our billing team will call you back."),
    ]
    session, results, state = await run_session(frozen, steps, ["refund please"])
    outs = tool_outputs(results[0])
    assert "not available at the moment" in outs[0].output
    assert state.transfers[0]["result"].startswith("failed")
    assert state.callbacks and not state.transferred and not state.on_hold
    assert said(session)[-1] == "Our billing team will call you back."


async def test_transfer_connected_makes_agent_silent(frozen, monkeypatch):
    seen = {}

    async def answered(**kw):
        seen.update(kw)
        return None

    _enable_transfer(frozen, monkeypatch, answered)
    steps = [
        (
            "call",
            "transfer_to_department",
            {"department_id": "billing", "summary": "Ramesh Kumar, 58, refund for last visit"},
        ),
        ("say", "SHOULD NOT BE SAID"),
    ]
    session, _, state = await run_session(frozen, steps, ["refund"])
    assert state.transferred and state.transfers[0]["result"] == "connected"
    assert "SHOULD NOT BE SAID" not in " ".join(said(session))
    assert seen["sip_call_to"] == "+911100000000"
    assert (
        "Ramesh Kumar, 58, refund" in seen["instructions"].extra
    )  # colleague gets the full concern


# -- emergency / callback / end ---------------------------------------------------------------
async def test_emergency_flag_alerts_staff(frozen, tmp_path):
    steps = [
        ("call", "flag_emergency", {"description": "chemical splash in eye"}),
        ("say", "Please rinse with water and go to emergency."),
    ]
    _, _, state = await run_session(frozen, steps, ["chuna went in my eye"])
    assert state.emergency_flagged
    import notify

    events = [json.loads(x) for x in notify.LOG_PATH.read_text().splitlines()]
    assert events[-1]["kind"] == "emergency_alert"


async def test_end_call_closes_session(frozen):
    steps = [
        ("call", "end_call", {}),
        (
            "say",
            "Thank you for calling Sample Eye Clinic. You were speaking with Clara. Wish you a lovely day ahead.",
        ),
    ]
    closed = asyncio.Event()
    state = CallState()
    with fake_job_context():
        async with AgentSession(llm=ScriptedLLM(steps), userdata=state) as session:
            session.on("close", lambda ev: closed.set())
            agent = FrontDeskAgent(frozen, scheduler=MockScheduler(frozen), greet=False)
            await session.start(agent)
            await session.run(user_input="No, that's all")
            await asyncio.wait_for(closed.wait(), 5)
    assert state.ending
    assert "You were speaking with Clara" in agent.closing_line


# -- provider failover -------------------------------------------------------------------------
async def test_llm_failover_keeps_call_alive(frozen):
    down = FailingLLM()
    backup = ScriptedLLM([("say", "Sure, I am glad to assist you with your query.")])
    adapter = llm.FallbackAdapter([down, backup], attempt_timeout=2.0, max_retry_per_llm=0)
    session, _, _ = await run_session(frozen, [], ["I want an appointment"], llm_override=adapter)
    assert down.attempts >= 1
    assert said(session)[-1].startswith("Sure, I am glad to assist")


async def test_slow_primary_llm_fails_over_within_budget(frozen):
    slow = ScriptedLLM([("say", "too late")], delay=10)
    backup = ScriptedLLM([("say", "Backup answered.")])
    adapter = llm.FallbackAdapter([slow, backup], attempt_timeout=0.5, max_retry_per_llm=0)
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    session, _, _ = await run_session(frozen, [], ["hello"], llm_override=adapter)
    assert said(session)[-1] == "Backup answered."
    assert loop.time() - t0 < 3


# -- greeting --------------------------------------------------------------------------------
async def test_greeting_is_spoken_on_enter(frozen):
    with fake_job_context():
        async with AgentSession(llm=ScriptedLLM([]), userdata=CallState()) as session:
            await session.start(FrontDeskAgent(frozen, scheduler=MockScheduler(frozen)))
            await asyncio.sleep(0.2)
            first = said(session)[0]
    assert first.startswith("Namaste, Sample Eye Clinic. This is Clara speaking")
