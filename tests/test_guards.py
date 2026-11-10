"""Silence handling, max call length, audit report, preflight."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from fake_llm import ScriptedLLM
from livekit.agents import AgentSession, UserStateChangedEvent
from livekit.agents.testing import fake_job_context

from call_state import CallState
from front_desk import FrontDeskAgent
from guards import CallGuards
from lines import LINES
from scheduling import MockScheduler
from telemetry import CallTelemetry, pct

ROOT = Path(__file__).resolve().parent.parent


def said(session):
    return [m.text_content for m in session.history.messages() if m.role == "assistant"]


def away(session):
    session.emit(
        "user_state_changed", UserStateChangedEvent(old_state="listening", new_state="away")
    )


def speaking(session):
    session.emit(
        "user_state_changed", UserStateChangedEvent(old_state="away", new_state="speaking")
    )


async def _session(cfg, state):
    session = AgentSession(llm=ScriptedLLM([]), userdata=state)
    await session.start(FrontDeskAgent(cfg, scheduler=MockScheduler(cfg), greet=False))
    return session


async def test_silent_caller_is_reprompted_then_disconnected(cfg):
    disconnected = asyncio.Event()

    async def disconnect():
        disconnected.set()

    with fake_job_context():
        state = CallState()
        session = await _session(cfg, state)
        guards = CallGuards(session, cfg, disconnect=disconnect, reprompt_s=0.2)
        away(session)
        await asyncio.wait_for(disconnected.wait(), 5)
        lines = said(session)
        assert lines == [
            LINES["silence_1"]["en"],
            LINES["silence_2"]["en"],
            LINES["silence_hangup"]["en"],
        ]
        assert guards.hangup_reason == "caller silent" and state.ending
        await guards.aclose()


async def test_caller_speaking_cancels_silence_flow(cfg):
    async def disconnect():
        raise AssertionError("must not hang up")

    with fake_job_context():
        state = CallState()
        session = await _session(cfg, state)
        guards = CallGuards(session, cfg, disconnect=disconnect, reprompt_s=0.3)
        away(session)
        await asyncio.sleep(0.1)
        speaking(session)
        await asyncio.sleep(0.8)
        assert said(session) == [LINES["silence_1"]["en"]]
        assert not state.ending
        await guards.aclose()
        await session.aclose()


async def test_the_same_silence_check_is_never_said_twice_in_a_row(cfg):
    """Seen on a real call: a noise made the bot ask "am I audible?" twice back to back
    while the caller was answering. A sound is not an answer; only words reset the count."""
    from livekit.agents import UserInputTranscribedEvent

    async def disconnect():
        raise AssertionError("must not hang up")

    with fake_job_context():
        state = CallState()
        session = await _session(cfg, state)
        guards = CallGuards(session, cfg, disconnect=disconnect, reprompt_s=0.3)
        guards.speech_without_text = False
        away(session)
        await asyncio.sleep(0.1)
        speaking(session)  # a noise, no words
        guards.speech_without_text = False  # (the not-caught path is tested separately)
        await asyncio.sleep(0.05)
        away(session)
        await asyncio.sleep(0.1)
        assert said(session) == [LINES["silence_1"]["en"], LINES["silence_2"]["en"]]
        speaking(session)
        session.emit(
            "user_input_transcribed",
            UserInputTranscribedEvent(transcript="haan, aawaz aa rahi hai", is_final=True),
        )
        assert state.silence_prompts == 0  # real words: the next check starts gently again
        await guards.aclose()
        await session.aclose()


def test_silence_lines_are_gentle():
    for key in ("silence_1", "silence_2", "silence_hangup"):
        for text in LINES[key].values():
            assert not text.startswith(("Hello", "हैलो")), text
            assert "!" not in text and "say something" not in text and "कुछ बोलिए" not in text


async def test_no_silence_prompts_during_hold_or_after_transfer(cfg):
    async def disconnect():
        raise AssertionError("must not hang up")

    with fake_job_context():
        for flag in ("on_hold", "transferred", "awaiting_answer"):
            state = CallState()
            setattr(state, flag, True)
            session = await _session(cfg, state)
            guards = CallGuards(session, cfg, disconnect=disconnect, reprompt_s=0.1)
            rearmed: list[int] = []
            session.reset_away_timer = lambda r=rearmed: r.append(1)
            away(session)
            await asyncio.sleep(0.4)
            assert said(session) == [], flag
            # the away timer is re-armed, so silence is noticed again once the hold ends
            assert rearmed, flag
            await guards.aclose()
            await session.aclose()


async def test_hindi_silence_prompt(cfg):
    with fake_job_context():
        state = CallState(language="hi")
        session = await _session(cfg, state)

        async def disconnect():
            pass

        guards = CallGuards(session, cfg, disconnect=disconnect, reprompt_s=0.1)
        away(session)
        await asyncio.sleep(0.05)
        assert said(session)[0] == LINES["silence_1"]["hi"]
        await guards.aclose()


async def test_max_duration_ends_call_and_requests_callback(cfg):
    import notify

    done = asyncio.Event()

    async def disconnect():
        done.set()

    with fake_job_context():
        state = CallState(caller_number="+919876543210")
        session = await _session(cfg, state)
        guards = CallGuards(session, cfg, disconnect=disconnect)
        guards.start_duration_limit(0.2)
        await asyncio.wait_for(done.wait(), 5)
        assert said(session)[-1] == LINES["max_duration"]["en"]
        events = [json.loads(x) for x in notify.LOG_PATH.read_text().splitlines()]
        assert events[-1]["kind"] == "callback_request"
        await guards.aclose()


async def test_call_report_is_written_with_masked_phone(cfg, tmp_path):
    steps = [
        (
            "call",
            "update_patient_details",
            {"name": "Sunita", "age": 40, "patient_type": "new", "mobile_number": "9876543210"},
        ),
        ("say", "Thank you Sunita ji."),
    ]
    with fake_job_context():
        state = CallState(caller_number="+919812345678", room="room-1")
        async with AgentSession(llm=ScriptedLLM(steps), userdata=state) as session:
            tel = CallTelemetry(session)
            await session.start(FrontDeskAgent(cfg, scheduler=MockScheduler(cfg), greet=False))
            await session.run(user_input="Sunita, forty, new patient, 9876543210")
            path = tel.write("room-1")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    blob = json.dumps(data["call"])
    assert "9876543210" not in blob and "9812345678" not in blob
    assert data["call"]["patient"]["name"] == "Sunita"
    assert any(i.get("type") == "function_call" for i in data["transcript"])
    assert set(data["latency"]) >= {
        "e2e_latency",
        "llm_node_ttft",
        "tts_node_ttfb",
        "end_of_turn_delay",
    }


def _clean_env(tmp_path):
    """No keys, no go-live override - even if the developer's .env.local has them
    (python-dotenv never overrides variables that are already set, even to "")."""
    keys = [
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "GOOGLE_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "XAI_API_KEY",
        "ELEVEN_API_KEY",
        "DEEPGRAM_API_KEY",
        "SARVAM_API_KEY",
        "CARTESIA_API_KEY",
    ]
    env = dict(os.environ)  # keep SYSTEMROOT etc. (Windows needs them)
    env.update(dict.fromkeys(keys, ""))
    env["ALLOW_PLACEHOLDER_DATA"] = "0"
    env["CLINIC_CONFIG"] = str(ROOT / "tests" / "fixtures" / "clinic_placeholder.yaml")
    return env


def test_percentiles():
    assert pct([], 0.5) is None
    assert pct([1.0], 0.95) == 1.0
    assert pct([1, 2, 3, 4, 5], 0.5) == 3
    assert abs(pct(list(range(1, 101)), 0.95) - 95.05) < 1e-9


def test_preflight_reports_missing_keys(tmp_path):
    env = _clean_env(tmp_path)
    r = subprocess.run(
        [sys.executable, str(ROOT / "src" / "agent.py"), "preflight"],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=120,
    )
    assert r.returncode == 1, r.stdout + r.stderr
    assert "NOT READY" in r.stdout
    assert "LIVEKIT_API_KEY" in r.stdout


def test_production_start_refuses_placeholder_data(tmp_path):
    env = _clean_env(tmp_path)
    r = subprocess.run(
        [sys.executable, str(ROOT / "src" / "agent.py"), "start"],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=120,
    )
    assert r.returncode != 0
    assert "placeholder_data" in (r.stdout + r.stderr)


async def test_total_llm_outage_never_leaves_caller_in_silence(cfg):
    """Every LLM in the chain down: apologise once, then hand off to a human callback and hang up."""
    from fake_llm import FailingLLM

    import notify

    done = asyncio.Event()

    async def disconnect():
        done.set()

    with fake_job_context():
        state = CallState(caller_number="+919876543210")
        session = AgentSession(llm=FailingLLM(), userdata=state)
        await session.start(FrontDeskAgent(cfg, scheduler=MockScheduler(cfg), greet=False))
        guards = CallGuards(session, cfg, disconnect=disconnect)
        session.generate_reply(user_input="I want an appointment")
        await asyncio.sleep(0)
        for _ in range(200):  # first failure -> apology line
            if LINES["llm_retry"]["en"] in said(session):
                break
            await asyncio.sleep(0.1)
        assert LINES["llm_retry"]["en"] in said(session)
        session.generate_reply(user_input="hello?")
        await asyncio.wait_for(done.wait(), 30)
        assert said(session)[-1] == LINES["technical_issue"]["en"]
        assert guards.hangup_reason.startswith("failsafe")
        events = [json.loads(x) for x in notify.LOG_PATH.read_text().splitlines()]
        assert (
            events[-1]["kind"] == "callback_request" and "technical failure" in events[-1]["reason"]
        )


async def test_untranscribed_speech_gets_repeat_request_not_are_you_there(cfg):
    """Caller talked but the recogniser returned nothing: ask to repeat, never count it as
    silence (the "are you there? are you there?" complaint)."""
    from livekit.agents import UserInputTranscribedEvent

    with fake_job_context():
        state = CallState()
        session = await _session(cfg, state)

        async def disconnect():
            pass

        guards = CallGuards(session, cfg, disconnect=disconnect, reprompt_s=0.1)
        rearmed = []
        session.reset_away_timer = lambda: rearmed.append(1)
        speaking(session)  # VAD heard speech ...
        session.emit(
            "user_state_changed", UserStateChangedEvent(old_state="speaking", new_state="listening")
        )
        away(session)  # ... but no transcript came
        await asyncio.sleep(0.3)
        assert said(session) == [LINES["not_caught"]["en"]]
        assert state.silence_prompts == 0 and not state.ending
        assert rearmed  # staying quiet after "could you repeat?" still gets a silence prompt

        # real silence afterwards still gets the normal prompt
        away(session)
        await asyncio.sleep(0.05)
        assert said(session)[-1] == LINES["silence_1"]["en"]

        # speech that WAS transcribed is not "not caught"
        speaking(session)
        session.emit(
            "user_input_transcribed",
            UserInputTranscribedEvent(transcript="book an appointment", is_final=True),
        )
        assert guards.speech_without_text is False
        await guards.aclose()
