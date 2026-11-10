"""Regression tests for issues found by the independent review and by starting the agent."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from fake_llm import ScriptedLLM
from livekit.agents import AgentSession, ChatMessage, StopResponse, UserStateChangedEvent
from livekit.agents.testing import fake_job_context

import front_desk
from call_state import CallState
from config import HoldCfg
from front_desk import FrontDeskAgent
from guards import CallGuards
from lines import LINES
from scheduling import MockScheduler
from telemetry import CallTelemetry

IST = ZoneInfo("Asia/Kolkata")
ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def frozen(cfg, monkeypatch):
    monkeypatch.setattr(type(cfg), "now", lambda self: datetime(2026, 10, 19, 11, 0, tzinfo=IST))
    return cfg


def outputs(result):
    return [ev.item for ev in result.events if type(ev).__name__ == "FunctionCallOutputEvent"]


def notifications():
    import notify

    if not notify.LOG_PATH.exists():
        return []
    return [json.loads(x) for x in notify.LOG_PATH.read_text().splitlines()]


async def settle():
    for _ in range(20):  # let fire-and-forget notifications land
        await asyncio.sleep(0.01)


async def run(cfg, steps, inputs, state=None, sched=None):
    state = state or CallState(caller_number="+919876543210")
    with fake_job_context():
        async with AgentSession(
            llm=ScriptedLLM(steps), userdata=state, max_tool_steps=5
        ) as session:
            agent = FrontDeskAgent(cfg, scheduler=sched or MockScheduler(cfg), greet=False)
            await session.start(agent)
            results = [await session.run(user_input=t) for t in inputs]
            await settle()
            return session, agent, results, state


async def test_closing_after_booking_is_the_opening_script_line(frozen):
    sched = MockScheduler(frozen)
    slot = (await sched.find_slots())[0]
    state = CallState(caller_number="+919876543210")
    state.patient.name = "Asha"
    steps = [
        ("call", "book_appointment", {"slot_id": slot.slot_id, "use_calling_number": True}),
        ("say", "Booked."),
    ]
    _, agent, _, _ = await run(frozen, steps, ["book"], state=state, sched=sched)
    assert agent.closing_line.startswith(
        "Thank you for choosing Sample Eye Clinic for your treatment"
    )


async def test_closing_without_booking_is_the_general_line(frozen):
    _, agent, _, _ = await run(frozen, [("say", "Okay.")], ["just asking"])
    assert agent.closing_line.startswith("Thank you for calling Sample Eye Clinic")


async def test_emergency_transfer_alerts_staff_even_if_transfer_unavailable(frozen, monkeypatch):
    monkeypatch.delenv("LIVEKIT_SIP_OUTBOUND_TRUNK", raising=False)
    steps = [
        (
            "call",
            "transfer_to_department",
            {"department_id": "emergency", "summary": "lime in eye"},
        ),
        ("say", "Go now."),
    ]
    _, _, results, state = await run(frozen, steps, ["chuna in eye"])
    out = outputs(results[0])[0].output
    assert state.emergency_flagged
    assert "go to the nearest eye or hospital emergency right now" in out
    assert "Do NOT offer a callback as the main help" in out
    assert any(n["kind"] == "emergency_alert" for n in notifications())


async def test_slow_booking_is_not_cancelled_and_is_recorded_late(frozen, monkeypatch):
    frozen.hold = HoldCfg(
        first_notice_after_s=0.1, checkin_every_s=5, max_hold_s=0.4, music_volume=0
    )
    sched = MockScheduler(frozen)
    slot = (await sched.find_slots())[0]
    monkeypatch.setenv("MOCK_SCHEDULER_LATENCY_S", "1.0")
    state = CallState(caller_number="+919876543210")
    state.patient.name = "Asha"
    steps = [
        ("call", "book_appointment", {"slot_id": slot.slot_id, "use_calling_number": True}),
        ("say", "We will confirm shortly."),
    ]
    _, _, results, _ = await run(frozen, steps, ["book"], state=state, sched=sched)
    out = outputs(results[0])[0].output
    assert out.startswith("BOOKING PENDING") and "do NOT say it failed" in out
    await asyncio.sleep(1.2)  # the backend finishes after the hold limit
    assert state.booking and state.booking.get("late") is True
    assert any(n["kind"] == "booking_confirmed_late" for n in notifications())


async def test_sms_not_promised_when_not_sent(frozen):
    sched = MockScheduler(frozen)
    slot = (await sched.find_slots())[0]
    state = CallState(caller_number="+919876543210")
    state.patient.name = "Asha"
    steps = [
        ("call", "book_appointment", {"slot_id": slot.slot_id, "use_calling_number": True}),
        ("say", "Done."),
    ]
    _, _, results, _ = await run(frozen, steps, ["book"], state=state, sched=sched)
    out = outputs(results[0])[0].output
    assert "Our team will send you the appointment details" in out
    assert "receive an SMS" not in out
    sms = next(n for n in notifications() if n["kind"] == "appointment_confirmation_sms")
    assert sms["address"], "SMS must carry the centre address (script: 'with landmark')"
    assert sms["to"].startswith("x") and "9876543210" not in json.dumps(sms), (
        "local log must mask phone"
    )


async def test_sms_promised_when_gateway_confirms(frozen, monkeypatch):
    async def ok(kind, payload, **kw):
        return True

    monkeypatch.setattr(front_desk, "notify", ok)
    sched = MockScheduler(frozen)
    slot = (await sched.find_slots())[0]
    state = CallState(caller_number="+919876543210")
    state.patient.name = "Asha"
    steps = [
        ("call", "book_appointment", {"slot_id": slot.slot_id, "use_calling_number": True}),
        ("say", "Done."),
    ]
    _, _, results, _ = await run(frozen, steps, ["book"], state=state, sched=sched)
    assert "you will receive an SMS from Sample Eye Clinic" in outputs(results[0])[0].output


async def test_no_fallback_line_after_end_call(frozen):
    """Live eval found: the empty end-of-call turn triggered 'Sorry, could you say that again?'."""
    steps = [
        ("calls", [("end_call", {})]),
        ("say", ""),
    ]
    with fake_job_context():
        state = CallState()
        async with AgentSession(
            llm=ScriptedLLM(steps, default=("say", "")), userdata=state
        ) as session:
            await session.start(
                FrontDeskAgent(frozen, scheduler=MockScheduler(frozen), greet=False)
            )
            await session.run(user_input="no, that's all")
            await asyncio.sleep(0.2)
            texts = [m.text_content for m in session.history.messages() if m.role == "assistant"]
    assert LINES["empty_turn"]["en"] not in texts


async def test_empty_llm_turn_is_never_silent(frozen):
    """Live eval found: a truncated tool call produced no output at all (dead air)."""
    with fake_job_context():
        state = CallState()
        async with AgentSession(llm=ScriptedLLM([("say", "")]), userdata=state) as session:
            await session.start(
                FrontDeskAgent(frozen, scheduler=MockScheduler(frozen), greet=False)
            )
            await session.run(user_input="hello")
            texts = [m.text_content for m in session.history.messages() if m.role == "assistant"]
    assert texts[-1] == LINES["empty_turn"]["en"]
    assert "empty LLM turn" in " ".join(state.errors)


async def test_do_not_call_is_not_a_callback(frozen):
    steps = [
        ("call", "mark_do_not_call", {"reason": "not interested"}),
        ("say", "Sorry for the disturbance."),
    ]
    _, _, _, state = await run(frozen, steps, ["don't call me again"])
    kinds = [n["kind"] for n in notifications()]
    assert state.do_not_call and "do_not_call" in kinds and "callback_request" not in kinds


async def test_outbound_stays_silent_until_greeting(frozen):
    with fake_job_context():
        state = CallState(direction="outbound")
        async with AgentSession(llm=ScriptedLLM([]), userdata=state) as session:
            agent = FrontDeskAgent(
                frozen,
                scheduler=MockScheduler(frozen),
                direction="outbound",
                greet=False,
                wait_for_greeting=True,
            )
            await session.start(agent)
            msg = ChatMessage(role="user", content=["Hello?"])
            with pytest.raises(StopResponse):
                await agent.on_user_turn_completed(session.history, msg)
            state.greeted = True
            await agent.on_user_turn_completed(session.history, msg)  # no exception now


async def test_hangup_cannot_be_cancelled_by_caller_speaking(cfg):
    disconnected = asyncio.Event()

    async def disconnect():
        disconnected.set()

    with fake_job_context():
        state = CallState()
        session = AgentSession(llm=ScriptedLLM([]), userdata=state)
        from fake_llm import ScriptedLLM as _S  # noqa: F401

        await session.start(FrontDeskAgent(cfg, scheduler=MockScheduler(cfg), greet=False))
        guards = CallGuards(session, cfg, disconnect=disconnect, reprompt_s=0.1)
        session.emit(
            "user_state_changed", UserStateChangedEvent(old_state="listening", new_state="away")
        )
        for _ in range(100):
            if state.ending:
                break
            await asyncio.sleep(0.02)
        assert state.ending
        # caller talks over the goodbye line
        session.emit(
            "user_state_changed", UserStateChangedEvent(old_state="away", new_state="speaking")
        )
        await asyncio.wait_for(disconnected.wait(), 5)
        await guards.aclose()


async def test_transcript_phone_numbers_are_scrubbed(frozen):
    steps = [
        ("call", "update_patient_details", {"name": "Asha", "mobile_number": "98765 43210"}),
        ("say", "Thank you."),
    ]
    with fake_job_context():
        state = CallState()
        async with AgentSession(llm=ScriptedLLM(steps), userdata=state) as session:
            tel = CallTelemetry(session)
            await session.start(
                FrontDeskAgent(frozen, scheduler=MockScheduler(frozen), greet=False)
            )
            await session.run(user_input="my number is 98765 43210")
            blob = json.dumps(tel.report())
    assert "98765 43210" not in blob and "9876543210" not in blob
    assert "3210" in blob


def test_dev_mode_refuses_placeholder_data_too(tmp_path):
    from test_guards import _clean_env

    env = _clean_env(tmp_path)
    r = subprocess.run(
        [sys.executable, str(ROOT / "src" / "agent.py"), "dev"],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=120,
    )
    assert r.returncode != 0 and "placeholder_data" in (r.stdout + r.stderr)


async def test_transfer_uses_configured_hold_music_volume(frozen, monkeypatch):
    seen = {}

    async def answered(**kw):
        seen.update(kw)

    monkeypatch.setenv("LIVEKIT_SIP_OUTBOUND_TRUNK", "ST_test")
    frozen.department("billing").sip_call_to = "+911100000000"
    monkeypatch.setattr(front_desk, "WarmTransferTask", answered)
    await run(
        frozen,
        [("call", "transfer_to_department", {"department_id": "billing", "summary": "bill"})],
        ["bill"],
    )
    assert seen["hold_audio"].volume == frozen.hold.music_volume


def test_fixed_lines_never_reset_llm_failure_counter():
    from guards import _FIXED_LINES

    assert LINES["llm_retry"]["en"] in _FIXED_LINES


def test_spoken_text_filter_drops_bracketed_meta_across_chunks():
    """Live eval: the model wrote '(The caller said ..., so I should move on.)' into its reply."""
    from front_desk import SpokenTextFilter

    f = SpokenTextFilter()
    out = "".join(
        f.feed(c)
        for c in [
            "Thank you. (The caller",
            " said new, so I",
            " move on.) Which",
            " centre? [note]",
        ]
    )
    assert out == "Thank you.  Which centre? "
    assert "The caller said new" in f.dropped


async def test_bracketed_text_is_not_spoken(frozen):
    with fake_job_context():
        state = CallState()
        steps = [("say", "Which centre suits you? (internal: ask location next)")]
        async with AgentSession(llm=ScriptedLLM(steps), userdata=state) as session:
            await session.start(
                FrontDeskAgent(frozen, scheduler=MockScheduler(frozen), greet=False)
            )
            await session.run(user_input="hi")
            texts = [m.text_content for m in session.history.messages() if m.role == "assistant"]
    assert texts[-1].strip() == "Which centre suits you?"
