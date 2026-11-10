"""Demo Eye and Retina Clinic voice agent (Clara): entrypoint.

    python src/agent.py console     # talk to it in your terminal
    python src/agent.py dev         # connect to LiveKit Cloud (web console / phone)
    python src/agent.py start       # production
    python src/agent.py preflight   # check config + keys, no call

Inbound calls arrive either through an Exotel -> LiveKit SIP trunk, or (Exotel
trial accounts, no SIP) through the Voicebot applet and src/exotel_bridge.py.
Outbound calls are dispatched with JSON metadata, see bench/dial.py.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
for _stream in (sys.stdout, sys.stderr):  # Hindi text on a cp1252 Windows console
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")
load_dotenv(ROOT / ".env.local")
load_dotenv(ROOT / ".env")

from livekit import api  # noqa: E402
from livekit.agents import (  # noqa: E402
    NOT_GIVEN,
    AgentServer,
    AgentSession,
    APIConnectOptions,
    BackgroundAudioPlayer,
    JobContext,
    SimulationContext,
    STTContextOptions,
    TurnHandlingOptions,
    cli,
    inference,
    room_io,
)
from livekit.agents.voice.agent_session import SessionConnectOptions  # noqa: E402

# plugins must be imported on the main thread at startup, never inside a call
from livekit.plugins import ai_coustics  # noqa: E402

from cache_warm import warm_prompt_cache  # noqa: E402
from call_state import CallState  # noqa: E402
from config import ClinicConfig, get_config  # noqa: E402
from dashboard_feed import dashboard_enabled  # noqa: E402
from dashboard_tap import DashboardTap  # noqa: E402
from exotel_signals import wire_clear_signal  # noqa: E402
from front_desk import FrontDeskAgent, enquiry_note  # noqa: E402
from greeting_cache import GreetingCache, synthesize, voice_identity  # noqa: E402
from guards import CallGuards  # noqa: E402
from lines import fixed_lines, inbound_greeting, outbound_greeting  # noqa: E402
from notify import notify, notify_bg  # noqa: E402
from prompts import closing_lines  # noqa: E402
from providers import (  # noqa: E402
    ProviderConfigError,
    build_llm,
    build_primary_tts,
    build_stt,
    build_tts,
    preflight,
    primary_tts_spec,
)
from scheduling import MockScheduler  # noqa: E402
from speech import speak, use_recordings  # noqa: E402
from telemetry import CallTelemetry  # noqa: E402

logger = logging.getLogger("frontdesk")

AGENT_NAME = os.getenv("AGENT_NAME", "eye-front-desk")
_SCHEDULER: MockScheduler | None = None
_SCHEDULER_LOCK = threading.Lock()


def scheduler_for(cfg: ClinicConfig) -> MockScheduler:
    # one in-memory scheduler per worker process, so bookings persist across calls
    # (locked: on Windows calls run as threads, and two first calls could each make one)
    global _SCHEDULER
    with _SCHEDULER_LOCK:
        if _SCHEDULER is None:
            _SCHEDULER = MockScheduler(
                cfg, bookings_log=Path(os.getenv("LOG_DIR", "logs")) / "bookings.jsonl"
            )
        return _SCHEDULER


def keyterms(cfg: ClinicConfig) -> list[str]:
    terms = {cfg.clinic.name, "Demo Clinic", cfg.personas.inbound.name, cfg.personas.outbound.name}
    for d in cfg.doctors:
        terms.add(d.name.replace("Dr. ", "").replace("Mr. ", ""))
    for loc in cfg.locations:
        terms.add(loc.name)
    return sorted(terms)


def guard_production(cfg: ClinicConfig) -> None:
    # `dev` and `start` can both receive real phone calls once a SIP dispatch rule points
    # at this agent, so both refuse placeholder data unless explicitly allowed.
    if (
        any(cmd in sys.argv for cmd in ("start", "dev", "connect"))
        and cfg.placeholder_data
        and os.getenv("ALLOW_PLACEHOLDER_DATA") != "1"
    ):
        raise SystemExit(
            "config/clinic.yaml still has placeholder_data: true (sample doctors/fees/numbers). "
            "Replace them before real callers reach this agent, or set ALLOW_PLACEHOLDER_DATA=1 "
            "for testing."
        )


def _livekit_cloud() -> bool:
    return all(os.getenv(k) for k in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"))


def turn_handling() -> TurnHandlingOptions:
    # adaptive interruption runs on LiveKit Cloud; without it, use VAD instead of erroring
    mode = os.getenv("INTERRUPTION_MODE") or ("adaptive" if _livekit_cloud() else "vad")
    return TurnHandlingOptions(
        # audio-based end-of-turn model: v1 on LiveKit Cloud, v1-mini locally (auto fallback)
        turn_detection=inference.TurnDetector(),
        interruption={
            "mode": mode,  # type: ignore[typeddict-item]
            # if a "barge-in" turns out to be a cough/"hmm", resume where we stopped
            "resume_false_interruption": True,
            "false_interruption_timeout": 1.5,
        },
        # max_delay caps how long we wait when the turn model isn't sure the caller is done.
        # It was 3.0 s: on the first real call 9 of 13 turns hit that cap (8 kHz phone audio),
        # worst on Hindi turns. If the caller was only pausing, adaptive interruption stops
        # the bot as soon as they carry on.
        endpointing={
            "mode": "dynamic",
            "min_delay": 0.3,
            "max_delay": float(os.getenv("TURN_MAX_DELAY_S", "1.0")),
        },
        preemptive_generation={"enabled": True},
    )


def parse_metadata(raw: str) -> dict:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        logger.warning("job metadata is not JSON, ignoring")
        return {}


server = AgentServer()


def on_simulation_end(sim: SimulationContext) -> None:
    """Deterministic checks for `lk agent simulate` runs (expectations in scenario userdata)."""
    try:
        exp = sim.userdata
        session = sim.job_context.primary_session
        state: CallState = session.userdata
        tools = [i.name for i in session.history.items if getattr(i, "type", "") == "function_call"]
        for t in exp.get("expect_tools", []):
            if t not in tools:
                sim.fail(f"expected tool {t} was not called (called: {tools})")
        for t in exp.get("forbid_tools", []):
            if t in tools:
                sim.fail(f"forbidden tool {t} was called")
        if "expect_booking" in exp and bool(state.booking) != bool(exp["expect_booking"]):
            sim.fail(f"booking expected={exp['expect_booking']} got={state.booking}")
    except Exception as e:  # a broken check must not crash the worker
        logger.exception("simulation check failed")
        sim.fail(f"simulation check error: {e}")


@server.rtc_session(agent_name=AGENT_NAME, on_simulation_end=on_simulation_end)
async def entrypoint(ctx: JobContext) -> None:
    cfg = get_config()
    meta = parse_metadata(ctx.job.metadata)
    direction = "outbound" if meta.get("phone_number") else "inbound"
    persona = cfg.personas.outbound.name if direction == "outbound" else cfg.personas.inbound.name
    ctx.log_context_fields = {"room": ctx.room.name, "direction": direction}

    if direction == "outbound" and not cfg.outbound.calling_window.contains(cfg.now()):
        logger.warning(
            "outbound call refused: outside calling window %s", cfg.outbound.calling_window
        )
        return

    state = CallState(
        direction=direction,  # type: ignore[arg-type]
        persona=persona,
        room=ctx.room.name,
        # outbound: the number we dial; Exotel bridge: the caller ID it was given
        caller_number=meta.get("phone_number") or meta.get("caller_number") or "",
        lead_source=meta.get("lead_source"),
    )
    channel = meta.get("channel", "sip")
    text_only = os.getenv("FRONTDESK_TEXT_ONLY") == "1"  # `console --text`: no STT/TTS needed
    state.bot = "standard"
    ctx.log_context_fields["bot"] = "standard"
    logger.info("bot: standard (STT -> LLM -> TTS)")

    terms = keyterms(cfg)
    try:
        llm = build_llm()
        stt = None if text_only else build_stt(terms)
        tts = None if text_only else build_tts()
    except ProviderConfigError as e:
        logger.error("cannot start call: %s", e)
        raise

    session = AgentSession(
        stt=stt or NOT_GIVEN,
        llm=llm,
        tts=tts or NOT_GIVEN,
        turn_handling=turn_handling() if not text_only else NOT_GIVEN,
        # the clinic's own names are always biased (keyterms); the background detector that
        # adds names heard in the call (a LiveKit LLM request) runs every third caller turn
        stt_context_options=STTContextOptions(
            keyterms=terms, keyterm_detection={"enabled": True, "turn_interval": 3}
        ),
        userdata=state,
        # fail fast: the fallback chain (not slow retries on one provider) keeps the call alive
        conn_options=SessionConnectOptions(
            llm_conn_options=APIConnectOptions(max_retry=1, retry_interval=0.5, timeout=8.0),
            tts_conn_options=APIConnectOptions(max_retry=1, retry_interval=0.5, timeout=8.0),
        ),
        max_tool_steps=5,
        user_away_timeout=float(os.getenv("USER_AWAY_TIMEOUT_S", "15")),
    )
    telemetry = CallTelemetry(session)
    # the live dashboard (off unless DASHBOARD=1; never blocks the call)
    tap = DashboardTap(session, state, channel="phone") if dashboard_enabled() else None
    background = BackgroundAudioPlayer()

    agent = FrontDeskAgent(
        cfg,
        scheduler=scheduler_for(cfg),
        direction=direction,  # type: ignore[arg-type]
        lead=meta,
        background=background,
        greet=(direction == "inbound"),
        wait_for_greeting=(direction == "outbound"),
        # recorded greetings in the voice the caller hears for the rest of the call
        greetings=None if text_only else greeting_cache(),
    )
    # every other fixed line (closing, silence, hold...) plays from a recording too
    use_recordings(agent.greetings, recorded_texts(cfg))

    # -- safety nets: silence, max duration, clean hang-up -------------------------------
    guards = CallGuards(
        session,
        cfg,
        disconnect=lambda: _end_call(ctx),
        reprompt_s=float(os.getenv("SILENCE_REPROMPT_S", "8")),
    )
    ctx.add_shutdown_callback(guards.aclose)
    if channel == "exotel":
        wire_clear_signal(session, ctx.room, _spawn)  # barge-in -> bridge clears Exotel's buffer

    @session.on("close")
    def _on_close(ev) -> None:
        reason = getattr(ev, "reason", None)
        err = getattr(ev, "error", None)
        if err is not None and not state.ending and not state.transferred:
            # the pipeline died (e.g. every LLM in the chain failed): make sure a human calls back
            state.errors.append(f"session closed on error: {err}")
            notify_bg(
                "callback_request",
                {
                    "reason": "technical failure during call",
                    "phone": state.caller_number,
                    "patient": state.patient.for_log(),
                    "room": state.room,
                },
            )
        logger.info("session closed: %s", reason)

    async def write_report() -> None:
        if note := enquiry_note(state):  # details taken, nothing booked: the team follows up
            try:
                await notify("enquiry_note", note, timeout=3.0)
            except Exception:
                logger.exception("could not send the enquiry note")
        cost = telemetry.log_cost()  # "CALL COST: ..." line in agent.log; also in the call report
        path = telemetry.write(ctx.room.name)
        if tap is not None:
            try:
                await tap.end(guards.hangup_reason or "call ended", cost["total_usd_priced"])
            except Exception:
                logger.exception("dashboard end event failed")
        lat = telemetry.report()["latency"]
        logger.info(
            "call done: e2e p50=%s p95=%s ttft p50=%s report=%s",
            lat["e2e_latency"]["p50"],
            lat["e2e_latency"]["p95"],
            lat["llm_node_ttft"]["p50"],
            path,
        )

    ctx.add_shutdown_callback(write_report)

    audio_in = room_io.AudioInputOptions()
    # Phone audio (Exotel bridge) is 8 kHz and already clean (noise floor ~10 RMS): the
    # voice-focus filter is built for wideband audio and, on the first test calls, the
    # recogniser lost most of what the caller said while it blocked the event loop for
    # 160-180 ms at a time. Off for phone calls unless NOISE_CANCELLATION says otherwise.
    default_nc = "off" if channel == "exotel" or not _livekit_cloud() else "ai-coustics"
    nc_mode = os.getenv("NOISE_CANCELLATION", default_nc)
    logger.info("noise cancellation: %s (channel %s)", nc_mode, channel)
    if nc_mode == "ai-coustics" and not text_only:
        audio_in = room_io.AudioInputOptions(
            noise_cancellation=ai_coustics.audio_enhancement(
                model=ai_coustics.EnhancerModel.QUAIL_VF_S
            )
        )

    # while the line connects and the greeting plays: makes the caller's first reply fast
    _spawn(warm_prompt_cache(session.llm, agent, on_usage=telemetry.add_warmup_usage))
    await session.start(
        agent=agent,
        room=ctx.room,
        room_options=room_io.RoomOptions(audio_input=audio_in),
    )
    await ctx.connect()
    try:
        await background.start(room=ctx.room, agent_session=session)
    except Exception:
        logger.exception("background audio unavailable; hold will be silent")

    guards.start_duration_limit()

    if direction == "inbound":
        _spawn(_fill_caller_number(ctx, state, on_known=tap.caller_known if tap else None))
        return

    # -- outbound: dial, check it's a human, then greet --------------------------------
    trunk = os.getenv("LIVEKIT_SIP_OUTBOUND_TRUNK")
    if not trunk:
        logger.error("outbound call needs LIVEKIT_SIP_OUTBOUND_TRUNK")
        ctx.shutdown("no outbound trunk")
        return
    phone = meta["phone_number"]
    state.awaiting_answer = True  # keep the silence guard quiet while it rings
    try:
        from livekit.agents import AMD

        req = api.CreateSIPParticipantRequest(
            room_name=ctx.room.name,
            sip_trunk_id=trunk,
            sip_call_to=phone,
            participant_identity="patient",
            wait_until_answered=True,
        )
        if os.getenv("LIVEKIT_SIP_NUMBER"):
            req.sip_number = os.environ["LIVEKIT_SIP_NUMBER"]
        req.ringing_timeout.FromNanoseconds(int(cfg.outbound.ring_timeout_s * 1e9))
        async with AMD(session) as detector:
            await ctx.api.sip.create_sip_participant(req)
            result = await detector.execute()
    except api.TwirpError as e:
        logger.warning("outbound call not answered: %s %s", e.message, e.metadata.get("sip_status"))
        state.errors.append(f"not answered: {e.metadata.get('sip_status')}")
        ctx.shutdown("not answered")
        return
    finally:
        state.awaiting_answer = False

    if result.is_machine:
        logger.info(
            "outbound reached a machine (%s); hanging up without a message", result.category
        )
        state.errors.append(f"machine: {result.category}")
        await guards.hang_up(None, "answering machine")
        return
    state.greeted = True
    speak(
        session,
        outbound_greeting(
            clinic=cfg.clinic.name,
            persona=persona,
            now=cfg.now(),
            disclose_ai=cfg.clinic.disclose_ai_in_greeting,
            team=cfg.clinic.team,
            lead_source=meta.get("lead_source"),
        ),
        allow_interruptions=True,
    )


_BACKGROUND: set[asyncio.Task] = set()


def _spawn(coro) -> asyncio.Task:
    """create_task that keeps a reference until done (unreferenced tasks can be GC'd)."""
    task = asyncio.create_task(coro)
    _BACKGROUND.add(task)
    task.add_done_callback(_BACKGROUND.discard)
    return task


async def _end_call(ctx: JobContext) -> None:
    # the goodbye has finished playing here, but its last ~0.3 s is still on its way to the
    # phone (WebRTC + phone network); deleting the room at once would clip it
    await asyncio.sleep(float(os.getenv("HANGUP_TAIL_S", "0.6")))
    await ctx.delete_room()


async def _fill_caller_number(ctx: JobContext, state: CallState, on_known=None) -> None:
    try:
        p = await asyncio.wait_for(ctx.wait_for_participant(), timeout=30)
        num = p.attributes.get("sip.phoneNumber") if p.attributes else None
        if isinstance(num, str) and num:
            state.caller_number = num
            if on_known:
                on_known()
    except Exception:
        logger.debug("no SIP caller number available")


def greeting_cache() -> GreetingCache:
    return GreetingCache(voice_identity(primary_tts_spec()), make_tts=build_primary_tts)


def recorded_texts(cfg: ClinicConfig) -> list[str]:
    """Fixed lines spoken by the code, not the model: played from recordings."""
    closings = [
        v
        for persona in {cfg.personas.inbound.name, cfg.personas.outbound.name}
        for v in closing_lines(cfg, persona).values()
    ]
    return list(dict.fromkeys([*fixed_lines(), *closings]))


def greeting_texts(cfg: ClinicConfig) -> list[str]:
    """Every inbound greeting a caller can hear (one text today; kept a list in case the
    greeting goes back to good morning / afternoon / evening)."""
    today = cfg.now()
    texts = [
        inbound_greeting(
            clinic=cfg.clinic.name,
            persona=cfg.personas.inbound.name,
            now=today.replace(hour=h, minute=0),
            disclose_ai=cfg.clinic.disclose_ai_in_greeting,
            team=cfg.clinic.team,
        )
        for h in (9, 14, 19)
    ]
    return list(dict.fromkeys(texts))


CALLER_SAMPLE_TEXT = "Hello, I want to book an eye check-up for my mother."
CALLER_SAMPLE_WAV = ROOT / "bench" / "results" / "caller-sample.wav"


async def _warmup_async() -> int:
    """Record the greetings and every fixed line that has no recording yet (recordings are
    kept, so a restart costs no TTS credits for them), and report the voice's real speed
    from here with a sample caller line."""
    from livekit.agents import utils

    cfg = get_config()
    cache = greeting_cache()
    print(f"voice: {primary_tts_spec()}  (recordings in {cache.folder})")
    wanted = list(dict.fromkeys([*greeting_texts(cfg), *recorded_texts(cfg)]))
    missing = [t for t in wanted if not cache.path(t).exists()]
    print(
        f"recordings: {len(wanted) - len(missing)} of {len(wanted)} ready, recording {len(missing)}"
    )
    # the speed check needs a cold and a warm request: the sample line twice if nothing is missing
    texts = [*missing, CALLER_SAMPLE_TEXT] if missing else [CALLER_SAMPLE_TEXT, CALLER_SAMPLE_TEXT]
    worst = 0.0
    async with utils.http_context.open():
        tts = build_primary_tts()
        try:
            for i, text in enumerate(texts):
                first, frames = await synthesize(tts, text)
                secs = sum(f.samples_per_channel / f.sample_rate for f in frames)
                tag = "cold connection" if i == 0 else "warm connection"
                print(
                    f"  first audio {first * 1000 if first else -1:6.0f} ms ({tag}), {secs:4.1f} s of speech: {text[:48]}"
                )
                if i > 0 and first:
                    worst = max(worst, first)
                if text == CALLER_SAMPLE_TEXT:
                    _save_wav_8k(CALLER_SAMPLE_WAV, frames)
                else:
                    cache.save(text, frames)
        finally:
            await tts.aclose()
    verdict = "OK" if worst <= 0.45 else "SLOW: every reply will wait this long for its first word"
    print(
        f"\nvoice first-audio on a warm connection: {worst * 1000:.0f} ms (target 250, max 450): {verdict}"
    )
    return 0


def _save_wav_8k(path: Path, frames: list) -> None:
    import wave

    from livekit import rtc

    rs = rtc.AudioResampler(frames[0].sample_rate, 8000, num_channels=1)
    out = [g for f in frames for g in rs.push(f)] + list(rs.flush())
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        for f in out:
            w.writeframes(bytes(f.data))


def _preflight() -> int:
    cfg = get_config()
    report = preflight()
    print(json.dumps(report, indent=2))
    ok = (
        all(report[k]["usable"] for k in ("stt", "llm", "tts"))
        and not report["livekit_keys_missing"]
    )
    print(
        f"\nconfig: {len(cfg.locations)} locations, {len(cfg.doctors)} doctors, "
        f"{len(cfg.departments)} departments, placeholder_data={cfg.placeholder_data}, "
        f"booking mode={cfg.booking.mode}"
    )
    transfer_ready = [d.id for d in cfg.departments if d.transfer_enabled]
    print(f"live transfer enabled for: {transfer_ready or 'none (callbacks only)'}")
    print(f"outbound trunk: {'set' if os.getenv('LIVEKIT_SIP_OUTBOUND_TRUNK') else 'NOT set'}")
    print("\nREADY" if ok else "\nNOT READY: add the missing keys above")
    return 0 if ok else 1


if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "preflight":
        raise SystemExit(_preflight())
    if cmd == "warmup":
        raise SystemExit(asyncio.run(_warmup_async()))
    if cmd == "console" and "--text" in sys.argv:
        os.environ["FRONTDESK_TEXT_ONLY"] = "1"
    guard_production(get_config())
    if cmd in ("dev", "start", "connect") and not all(
        os.getenv(k) for k in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
    ):
        raise SystemExit(
            "LiveKit Cloud keys missing (LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET). "
            "Put them in .env.local, then run `python src/agent.py preflight`."
        )
    cli.run_app(server)
