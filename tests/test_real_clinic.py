"""the clinic's real data (config/clinic.yaml) and the appointment-request booking mode.

Demo Clinic publishes no opening hours and no OPD timings, so the bot takes an appointment
request and the team calls back to confirm the time. These tests pin that: nothing
invented in the data, the front desk SOP in the confirmation, and impossible requests
refused.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from conftest import REAL_CLINIC
from fake_llm import ScriptedLLM
from livekit.agents import AgentSession
from livekit.agents.testing import fake_job_context

import notify
from call_state import CallState
from config import HoldCfg
from front_desk import PENDING_MSG, FrontDeskAgent
from prompts import build_instructions
from scheduling import MockScheduler

IST = ZoneInfo("Asia/Kolkata")
MONDAY_11 = datetime(2026, 10, 19, 11, 0, tzinfo=IST)
DETAILS = {"name": "Sunita Sharma", "age": 45, "city": "Gurgaon", "concern": "blurred vision"}


@pytest.fixture
def real(real_cfg, monkeypatch):
    monkeypatch.setattr(type(real_cfg), "now", lambda self: MONDAY_11)
    return real_cfg


# -- the data itself -------------------------------------------------------------------
def test_real_clinic_has_only_published_data(real_cfg):
    c = real_cfg
    assert c.placeholder_data is False and c.booking.mode == "request"
    assert c.clinic.name == "Demo Eye and Retina Clinic"
    assert c.personas.inbound.name == c.personas.outbound.name == "Clara"
    assert [loc.id for loc in c.locations] == ["greenpark"]
    assert [d.name for d in c.doctors] == ["Dr. Arjun Mehta"]
    loc = c.location("greenpark")
    assert loc.phones == ["98765 43210"] and "A-14 Green Park" in loc.address
    # the website's "Dilated Eye Examination, Refraction & Consultation" charge
    assert loc.consultation_fee_inr == 1700
    # not published by Demo Clinic, so never in the data
    assert loc.open_days == [] and loc.hours == ""
    assert c.clinic.consultation.duration_text == ""
    assert not any(d.has_schedule for d in c.doctors)
    assert all(d.hours is None and not d.sip_call_to for d in c.departments)
    assert {d.id for d in c.departments} == {"coordinator", "billing", "emergency"}
    assert c.insurance.tpas == []
    text = REAL_CLINIC.read_text(encoding="utf-8")
    assert "placeholder" not in text.replace("placeholder_data", "").lower()


@pytest.mark.parametrize(
    ("spoken", "expected"),
    [
        ("Dr. Arjun Mehta", ["Dr. Arjun Mehta"]),
        ("arjun mehta", ["Dr. Arjun Mehta"]),
        ("Dr Mehta", ["Dr. Arjun Mehta"]),
        ("Arjun Mehtaa", ["Dr. Arjun Mehta"]),  # speech-to-text spelling
        ("Dr. Verma", []),  # not at Demo Clinic
    ],
)
def test_doctor_names_match_the_way_callers_say_them(real_cfg, spoken, expected):
    assert [d.name for d in real_cfg.match_doctors(spoken)] == expected


def test_real_prompt_states_facts_and_the_sop_but_no_timings(real_cfg):
    p = build_instructions(real_cfg, direction="inbound", now=MONDAY_11)
    for must in (
        "Demo Eye and Retina Clinic (Dr. Arjun Mehta's team)",
        "Level 2, A-14 Green Park Main, New Delhi 110016",
        "the nearest metro is Green Park",
        "Rs. 1700, covering the dilated eye check-up and refraction",
        "Opening days and hours are not known to you",
        "never state, guess or promise a doctor's day or time",
        "never ask the caller which centre",
        # SOP
        "an appointment, a general query, an emergency, or billing",
        "Always take a name, even if",
        "never ask for a contact number unless",
        "name and age together in one question",
        "Can I book an appointment with the doctor for you?",
        "comfort triad",
        "we host patients rather than merely handle them",
        "never quote a price before you understand the caller's eye problem",
        "never suggest paying again for something already paid",
        "डेमो आई एंड रेटिना क्लिनिक",
        "request_appointment",
    ):
        assert must in p, must
    assert "check_doctor_availability" not in p and "book_appointment" not in p
    # the consultation charge is the only price the bot knows
    assert re.findall(r"Rs\. ?\d+(?:,\d+)*", p) == ["Rs. 1700"]
    assert not re.search(r"\d[\d,]* rupees|rupees \d", p)


def test_each_booking_mode_gets_only_its_own_tools(real_cfg, cfg):
    def names(c):
        return {t.id for t in FrontDeskAgent(c, scheduler=MockScheduler(c), greet=False).tools}

    real = names(real_cfg)
    assert "request_appointment" in real
    assert not real & {"check_doctor_availability", "book_appointment"}
    sample = names(cfg)  # sample clinic: slot booking
    assert {"check_doctor_availability", "book_appointment"} <= sample
    assert "request_appointment" not in sample


# -- the request flow --------------------------------------------------------------------
def _outputs(result) -> list:
    return [ev.item for ev in result.events if type(ev).__name__ == "FunctionCallOutputEvent"]


async def _request(
    cfg,
    args: dict,
    *,
    details: dict | None = None,
    scheduler=None,
    times=1,
    user_input="Sunita Sharma, 45, from Gurgaon, blurred vision",
    caller_number="+919876543210",
):
    """One caller turn: the model saves the details, then registers the request.
    agent.said = everything spoken in the turn."""
    steps = [("call", "update_patient_details", details or DETAILS)]
    for _ in range(times):
        steps.append(("call", "request_appointment", args))
    steps.append(("say", "Is there anything else I may assist you with?"))
    state = CallState(caller_number=caller_number)
    with fake_job_context():
        async with AgentSession(llm=ScriptedLLM(steps), userdata=state, max_tool_steps=5) as s:
            agent = FrontDeskAgent(cfg, scheduler=scheduler or MockScheduler(cfg), greet=False)
            await s.start(agent)
            result = await s.run(user_input=user_input)
            await asyncio.gather(*list(notify._BACKGROUND))
            agent.said = "\n".join(
                m.text_content or "" for m in s.history.messages() if m.role == "assistant"
            )
            return _outputs(result)[1:], state, agent


async def test_request_registered_with_the_sop_confirmation(real):
    args = {
        "preferred_day": "Tuesday",
        "preferred_time": "morning",
        "date_iso": "2026-10-20",
        "use_calling_number": True,
    }
    outs, state, agent = await _request(real, args)
    out = outs[0].output
    assert not outs[0].is_error and out.startswith("REQUEST REGISTERED")
    # English: the code speaks the scripted confirmation itself (no second LLM round trip)
    assert "do not repeat it" in out
    spoken = agent.said
    for must in (  # SOP: repeat the details back, charges, preparation instructions
        "appointment request for Sunita Sharma with Dr. Arjun Mehta at our Green Park "
        "clinic, for Tuesday 20 October, morning",
        "will call you on your number ending 3 2 1 0 to confirm the exact time",
        "seventeen hundred rupees, which covers the dilated eye check-up and refraction",
        "previous eye reports, prescriptions",
        "come fifteen minutes early",
        "please don't drive back alone",
        # the calling number was used without asking: they can change it
        "If you would like us to call a different number, please tell me.",
        "Is there anything else I may assist you with?",
    ):
        assert must in spoken, must
    assert "We have booked" not in spoken and "including registration" not in spoken
    assert "The consultation takes" not in spoken  # visit length is not published
    b = state.booking
    assert b["status"] == "requested" and b["request_id"].startswith("SEC-R")
    assert b["doctor"] == "Dr. Arjun Mehta" and b["location"] == "Green Park"
    assert b["date"] == "2026-10-20" and b["phone"].endswith("3210") and "9876" not in b["phone"]
    assert "We look forward to taking care of you" in agent.closing_line  # booking closing
    events = [json.loads(x) for x in notify.LOG_PATH.read_text(encoding="utf-8").splitlines()]
    req = next(e for e in events if e["kind"] == "appointment_request")
    assert req["doctor_name"] == "Dr. Arjun Mehta" and req["location_id"] == "greenpark"
    # the SOP's register line: name, location, eye complaint
    assert req["note"] == "Pt Sunita Sharma — Gurgaon — blurred vision"
    assert req["to"].endswith("3210") and "9876" not in req["to"]  # local log is masked


async def test_request_on_a_given_number(real):
    _, state, agent = await _request(
        real,
        {"preferred_day": "any day this week"},
        details={**DETAILS, "mobile_number": "98111 22334"},
    )
    said = agent.said
    assert "with Dr. Arjun Mehta at our Green Park clinic, for any day this week" in said
    assert "ending 2 3 3 4" in said
    assert state.booking["doctor"] == "Dr. Arjun Mehta"


async def test_unpublished_opening_days_never_refuse_a_day(real):
    """Demo Clinic publishes no opening days: the bot takes any future day and the team
    confirms (it never tells a caller the clinic is closed)."""
    outs, state, _ = await _request(
        real, {"preferred_day": "Sunday", "date_iso": "2026-10-25", "use_calling_number": True}
    )
    assert not outs[0].is_error and state.booking["date"] == "2026-10-25"


@pytest.mark.parametrize(
    ("args", "error"),
    [
        ({"preferred_day": "Monday", "doctor_name": "Verma"}, "no doctor named"),
        ({"preferred_day": "Friday", "date_iso": "2026-10-16"}, "in the past"),
        ({"preferred_day": "Monday", "specialty": "contact lens"}, "unknown specialty"),
        ({"location_id": "noida", "preferred_day": "Monday"}, "unknown location_id"),
    ],
)
async def test_impossible_requests_are_refused(real, args, error):
    outs, state, _ = await _request(real, {**args, "use_calling_number": True})
    assert outs[0].is_error and error in outs[0].output, outs[0].output
    assert state.booking is None


async def test_calling_number_is_used_without_asking(real):
    """No "shall I use this number?" turn: the number on the line is used and read back."""
    outs, state, agent = await _request(
        real, {"preferred_day": "Monday"}, details={"name": "Sunita Sharma", "age": 45}
    )
    assert not outs[0].is_error and state.booking["phone"].endswith("3210")
    assert "ending 3 2 1 0" in agent.said and "different number" in agent.said


async def test_a_number_the_caller_gave_wins_and_is_not_questioned(real):
    _, state, agent = await _request(
        real,
        {"preferred_day": "Monday"},
        details={"name": "Sunita Sharma", "mobile_number": "98111 22334"},
    )
    assert state.booking["phone"].endswith("2334") and "different number" not in agent.said


async def test_request_needs_a_name_and_a_number(real):
    outs, _, _ = await _request(
        real,
        {"preferred_day": "Monday", "use_calling_number": True},
        details={"name": "Sunita Sharma", "age": 45},
        caller_number="",  # console call, withheld caller ID
    )
    assert outs[0].is_error and "ask them to say a 10-digit mobile number" in outs[0].output
    outs, _, _ = await _request(
        real,
        {"preferred_day": "Monday", "use_calling_number": True},
        details={"age": 45},
    )
    assert outs[0].is_error and "patient name missing" in outs[0].output


async def test_repeated_request_is_registered_once(real):
    sched = MockScheduler(real)
    args = {"preferred_day": "Monday", "use_calling_number": True}
    outs, state, _ = await _request(real, args, scheduler=sched, times=2)
    first = re.search(r"SEC-R\w+", json.dumps(state.booking)).group(0)
    assert len(sched._requests) == 1 and sched._requests[0].request_id == first
    assert all(o.output.startswith("REQUEST REGISTERED") for o in outs)


async def test_slow_request_backend_uses_the_hold_script(real, monkeypatch):
    monkeypatch.setenv("MOCK_SCHEDULER_LATENCY_S", "3")
    real.hold = HoldCfg(first_notice_after_s=0.2, checkin_every_s=5, max_hold_s=1.0, music_volume=0)
    outs, state, _ = await _request(real, {"preferred_day": "Monday", "use_calling_number": True})
    assert outs[0].output == PENDING_MSG
    assert state.booking is None and state.holds


async def test_still_needed_lists_the_sop_details(real):
    """SOP: name, age, city and eye complaint (the number is confirmed at the request)."""
    _, out_state, _ = await _request(
        real, {"preferred_day": "Monday", "use_calling_number": True}, details={"name": "Asha"}
    )
    assert out_state.patient.missing_essentials() == ["age", "city", "concern"]


async def test_callback_carries_the_sop_register_line(real):
    steps = [
        ("call", "update_patient_details", DETAILS),
        ("call", "schedule_callback", {"reason": "wants to know about retina surgery"}),
        ("say", "Our team will call you back."),
    ]
    state = CallState(caller_number="+919876543210")
    with fake_job_context():
        async with AgentSession(llm=ScriptedLLM(steps), userdata=state, max_tool_steps=5) as s:
            await s.start(FrontDeskAgent(real, scheduler=MockScheduler(real), greet=False))
            await s.run(user_input="Sunita Sharma, 45, Gurgaon, blurred vision. Call me back.")
            await asyncio.gather(*list(notify._BACKGROUND))
    events = [json.loads(x) for x in notify.LOG_PATH.read_text(encoding="utf-8").splitlines()]
    cb = next(e for e in events if e["kind"] == "callback_request")
    assert cb["note"] == "Pt Sunita Sharma — Gurgaon — blurred vision"


async def test_call_is_not_ended_in_the_same_turn_as_an_emergency(real):
    steps = [
        ("call", "flag_emergency", {"description": "lime splash in child's eye"}),
        ("call", "end_call", {}),
        ("say", "This needs urgent attention. Is there anything else I can do for you?"),
    ]
    state = CallState(caller_number="+919876543210")
    with fake_job_context():
        async with AgentSession(llm=ScriptedLLM(steps), userdata=state, max_tool_steps=5) as s:
            await s.start(FrontDeskAgent(real, scheduler=MockScheduler(real), greet=False))
            result = await s.run(user_input="chuna went into my son's eye")
            outs = _outputs(result)
    end = next(o for o in outs if o.name == "end_call")
    assert end.is_error and "Do not end the call now" in end.output
    assert state.emergency_flagged and not state.ending


async def test_call_can_end_once_the_caller_has_answered_after_an_emergency(real):
    steps = [
        ("call", "flag_emergency", {"description": "sudden loss of vision"}),
        ("say", "This needs urgent attention. Is there anything else I can do for you?"),
        ("call", "end_call", {}),
        ("say", "Thank you for calling Demo Eye and Retina Clinic."),
    ]
    state = CallState(caller_number="+919876543210")
    with fake_job_context():
        async with AgentSession(llm=ScriptedLLM(steps), userdata=state, max_tool_steps=5) as s:
            await s.start(FrontDeskAgent(real, scheduler=MockScheduler(real), greet=False))
            await s.run(user_input="I suddenly cannot see from my right eye")
            result = await s.run(user_input="No, we are going to the hospital now, thank you")
            end = next(o for o in _outputs(result) if o.name == "end_call")
    assert not end.is_error and state.ending


async def test_call_is_not_ended_while_the_caller_is_about_to_answer(real):
    """Eval S19: the bot asked for the age, then called end_call in the next step."""
    steps = [("say_call", "May I know the patient's age, please?", "end_call", {})]
    state = CallState(caller_number="+919876543210")
    with fake_job_context():
        async with AgentSession(llm=ScriptedLLM(steps), userdata=state, max_tool_steps=5) as s:
            await s.start(FrontDeskAgent(real, scheduler=MockScheduler(real), greet=False))
            result = await s.run(user_input="The patient is Sunita Sharma.")
            end = next(o for o in _outputs(result) if o.name == "end_call")
    assert not end.is_error and end.output.startswith("Not ended")
    assert not state.ending


@pytest.mark.parametrize(
    "ask",
    [
        "धन्यवाद सुनीता जी। कृपया मरीज़ की उम्र बताइए।",  # a polite request, no question mark
        "धन्यवाद सुनीता जी। क्या आप मरीज़ की उम्र बता सकते हैं?",
        "Thank you, Sunita ji. Please tell me the patient's age.",
    ],
)
async def test_a_polite_request_is_not_asked_twice(real, ask):
    """Eval S19: "कृपया मरीज़ की उम्र बताइए।" has no "?", so the bot used to take a second
    step and ask for the age again straight away. Asking twice sounds pushy."""
    fake = ScriptedLLM(
        [
            (
                "say_call",
                ask,
                "update_patient_details",
                {"patient_type": "new", "name": "Sunita Sharma"},
            ),
            ("say", "माफ़ कीजिए, क्या आप मरीज़ की उम्र बता सकते हैं?"),
        ]
    )
    state = CallState(caller_number="+919876543210", language="hi")
    with fake_job_context():
        async with AgentSession(llm=fake, userdata=state, max_tool_steps=5) as s:
            await s.start(FrontDeskAgent(real, scheduler=MockScheduler(real), greet=False))
            await s.run(user_input="सुनीता शर्मा, नया मरीज़।")
            said = [m.text_content for m in s.history.messages() if m.role == "assistant"]
    assert said == [ask] and len(fake.calls) == 1


async def test_tool_hint_follows_a_hinglish_caller_without_the_phone_hook(real):
    """Typed turns (console --text, evals) skip on_user_turn_completed; the language still
    follows the caller before any tool reads it."""
    _, state, _ = await _request(
        real,
        {"preferred_day": "kal", "use_calling_number": True},
    )
    assert state.language == "en"  # "Sunita Sharma, 45, from Gurgaon..." is English
    steps = [("call", "update_patient_details", {"name": "Rohit Singh"}), ("say", "जी")]
    state = CallState()
    with fake_job_context():
        async with AgentSession(llm=ScriptedLLM(steps), userdata=state) as s:
            await s.start(FrontDeskAgent(real, scheduler=MockScheduler(real), greet=False))
            result = await s.run(user_input="Haan ji, mujhe kal subah appointment chahiye")
    assert state.language == "hi"
    assert "Your reply: in Hindi (Devanagari)" in _outputs(result)[0].output


async def test_prompt_cache_warmup_is_only_for_anthropic(real):
    from cache_warm import warm_prompt_cache

    agent = FrontDeskAgent(real, scheduler=MockScheduler(real), greet=False)
    assert await warm_prompt_cache(ScriptedLLM([]), agent) is False  # nothing to warm
    assert await warm_prompt_cache(None, agent) is False


def test_anthropic_thinking_and_effort_knobs(monkeypatch):
    from providers import ChainItem, TunedAnthropicLLM, _make_llm

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.delenv("ANTHROPIC_THINKING", raising=False)
    monkeypatch.delenv("ANTHROPIC_EFFORT", raising=False)
    tuned_default = _make_llm(ChainItem("anthropic", "claude-sonnet-5-5"))
    assert isinstance(tuned_default, TunedAnthropicLLM)
    assert tuned_default.request_defaults == {"output_config": {"effort": "low"}}  # default
    monkeypatch.setenv("ANTHROPIC_EFFORT", "")
    assert _make_llm(ChainItem("anthropic", "claude-haiku-5-5")).request_defaults == {}
    monkeypatch.setenv("ANTHROPIC_THINKING", "disabled")
    monkeypatch.setenv("ANTHROPIC_EFFORT", "low")
    tuned = _make_llm(ChainItem("anthropic", "claude-haiku-5-5"))
    assert tuned.request_defaults == {
        "thinking": {"type": "disabled"},
        "output_config": {"effort": "low"},
    }


def test_warmup_twin_matches_the_call_model(monkeypatch):
    from cache_warm import _warmup_twin
    from providers import ChainItem, _make_llm

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("ANTHROPIC_THINKING", "disabled")
    monkeypatch.delenv("ANTHROPIC_EFFORT", raising=False)
    twin = _warmup_twin(_make_llm(ChainItem("anthropic", "claude-haiku-5-5")))
    assert twin is not None and twin.model == "claude-haiku-5-5"
    assert twin.request_defaults == {
        "thinking": {"type": "disabled"},
        "output_config": {"effort": "low"},
    }
    assert _warmup_twin(ScriptedLLM([])) is None


async def test_hindi_caller_gets_the_confirmation_rendered_by_the_model(real):
    outs, state, agent = await _request(
        real,
        {
            "preferred_day": "kal",
            "date_iso": "2026-10-20",
            "use_calling_number": True,
        },
        user_input="Haan ji, mujhe kal appointment chahiye, naam Sunita Sharma",
    )
    out = outs[0].output
    assert state.language == "hi"
    assert "faithful natural Hindi rendering" in out and "I have registered" in out
    assert 'For Hindi, the day is exactly: "मंगलवार, 20 अक्टूबर"' in out
    assert "I have registered" not in agent.said  # not spoken by the code in English


async def _one_turn(cfg, steps, user_input="I am Rahul Verma, 40, new patient"):
    fake = ScriptedLLM(steps)
    state = CallState(caller_number="+919876543210")
    with fake_job_context():
        async with AgentSession(llm=fake, userdata=state, max_tool_steps=5) as s:
            await s.start(FrontDeskAgent(cfg, scheduler=MockScheduler(cfg), greet=False))
            await s.run(user_input=user_input)
            said = [m.text_content for m in s.history.messages() if m.role == "assistant"]
    return fake, state, said


async def test_details_are_saved_without_a_second_llm_round_trip(real):
    args = {"patient_type": "new", "name": "Rahul Verma", "age": 40}
    fake, state, said = await _one_turn(
        real,
        [
            (
                "say_call",
                "Thank you, Rahul ji. Which city are you coming from?",
                "update_patient_details",
                args,
            )
        ],
    )
    assert len(fake.calls) == 1  # one LLM request for the whole turn
    assert state.patient.name == "Rahul Verma" and state.patient.age == 40
    assert said == ["Thank you, Rahul ji. Which city are you coming from?"]


@pytest.mark.parametrize(
    ("text", "args"),
    [
        ("Thank you, Rahul ji.", {"name": "Rahul Verma"}),  # no question yet: must go on
        ("Which city are you coming from?", {"mobile_number": "12345"}),  # bad number: must say so
    ],
)
async def test_detail_turn_still_gets_a_reply_when_needed(real, text, args):
    fake, _, said = await _one_turn(
        real,
        [("say_call", text, "update_patient_details", args), ("say", "Could you tell me more?")],
    )
    assert len(fake.calls) == 2 and said[-1] == "Could you tell me more?"


async def test_goodbye_is_not_said_twice(real):
    closing = (
        "Thank you for calling Demo Eye and Retina Clinic. You were speaking with "
        "Clara. Take care, and have a good day."
    )
    fake, state, said = await _one_turn(
        real, [("say_call", closing, "end_call", {}), ("say", closing)], "No, that is all."
    )
    assert said == [closing] and len(fake.calls) == 1 and state.ending


def test_details_without_a_booking_reach_the_team():
    """Console call, 4 Nov: name, age, city and complaint taken, then the caller hung up
    before the request; nothing reached the team."""
    from front_desk import enquiry_note

    state = CallState(caller_number="+919876543210", room="exotel-1")
    assert enquiry_note(state) is None  # nothing said: nothing to follow up
    state.patient.name, state.patient.city, state.patient.concern = "Ashutosh", "Faridabad", "pain"
    note = enquiry_note(state)
    assert note["note"] == "Pt Ashutosh — Faridabad — pain" and note["phone"] == "+919876543210"
    state.callbacks.append({"reason": "x"})
    assert enquiry_note(state) is None  # a callback already carries it
    state.callbacks.clear()
    state.booking = {"status": "requested"}
    assert enquiry_note(state) is None


async def test_own_goodbye_without_the_thanks_still_gets_the_sop_closing(real):
    """Console call, 4 Nov: "You can hang up" got "...Take care." and the call ended without
    the SOP's "Thank you for calling"."""
    closing = (
        "Thank you for calling Demo Eye and Retina Clinic. You were speaking with "
        "Clara. Take care, and have a good day."
    )
    fake, _, said = await _one_turn(
        real,
        [
            (
                "say_call",
                "I understand, Ashutosh ji. You can call us anytime on 9 8 7 6 5, 4 3 2 1 0.",
                "end_call",
                {},
            ),
            ("say", closing),
        ],
        "No. You can hang up the call.",
    )
    # English: the code says the closing itself, no second LLM round trip
    assert len(fake.calls) == 1 and said[-1] == closing


async def test_short_text_before_end_call_still_gets_the_scripted_closing(real):
    fake, state, said = await _one_turn(
        real, [("say_call", "Okay.", "end_call", {})], "No, that is all."
    )
    closing = (
        "Thank you for calling Demo Eye and Retina Clinic. You were speaking with "
        "Clara. Take care, and have a good day."
    )
    # one LLM request: the English closing is spoken by the code, word for word
    assert len(fake.calls) == 1 and said[-1] == closing and state.ending


async def test_hindi_closing_is_still_rendered_by_the_model(real):
    fake = ScriptedLLM(
        [
            ("say_call", "जी।", "end_call", {}),
            ("say", "डेमो आई एंड रेटिना क्लिनिक को कॉल करने के लिए धन्यवाद।"),
        ]
    )
    state = CallState(caller_number="+919876543210", language="hi")
    with fake_job_context():
        async with AgentSession(llm=fake, userdata=state, max_tool_steps=5) as s:
            await s.start(FrontDeskAgent(real, scheduler=MockScheduler(real), greet=False))
            await s.run(user_input="नहीं, बस इतना ही।")
            said = [m.text_content for m in s.history.messages() if m.role == "assistant"]
    assert len(fake.calls) == 2 and said[-1].endswith("धन्यवाद।")


def test_call_cost_is_computed_from_reported_usage():
    from call_cost import CallUsage

    u = CallUsage()
    # one Sonnet request: 10,000 read from cache, 300 written, 50 plain input, 100 out
    u.add_llm(
        "api.anthropic.com/claude-sonnet-5-5",
        prompt=10_350,
        cache_read=10_000,
        cache_write=300,
        output=100,
    )
    u.add_llm(
        "livekit/google/gemma-4-31b-it", prompt=1_000, cache_read=600, cache_write=0, output=30
    )
    u.add_llm(
        "api.anthropic.com/claude-sonnet-5-5 (warm-up)",
        prompt=10_078,
        cache_read=0,
        cache_write=10_000,
        output=40,
    )
    u.add_stt("livekit/deepgram/nova-3", 90.0)
    u.add_tts("ElevenLabs/eleven_flash_v2_5", 500)
    c = u.summary()
    sonnet = 10_000 * 0.10e-6 + 300 * 2.50e-6 + 50 * 2.00e-6 + 100 * 10.00e-6
    sonnet += 10_000 * 2.50e-6 + 78 * 2.00e-6 + 40 * 10.00e-6  # the warm-up
    gemma = 400 * 0.40e-6 + 600 * 0.20e-6 + 30 * 1.20e-6
    deepgram = 1.5 * 0.0058
    assert abs(c["anthropic_usd"] - sonnet) < 1e-9
    assert abs(c["livekit_inference_usd"] - (gemma + deepgram)) < 1e-9
    assert abs(c["total_usd_priced"] - (sonnet + gemma + deepgram)) < 1e-9
    assert any("eleven_flash_v2_5 (500 characters)" in x for x in c["unpriced"])  # no guess
