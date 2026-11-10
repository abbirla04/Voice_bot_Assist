"""The front-desk agent (Clara, inbound and outbound) and its tools."""

from __future__ import annotations

import logging
import os
import re
from datetime import date
from typing import Any, Literal

from livekit.agents import (
    Agent,
    AudioConfig,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
    ChatContext,
    ChatMessage,
    RunContext,
    StopResponse,
    ToolError,
    function_tool,
    llm,
)
from livekit.agents.beta.tools import EndCallTool
from livekit.agents.beta.workflows import WarmTransferTask, WorkflowInstructions

from call_state import (
    CallState,
    detect_language,
    devanagari_share,
    mask_phone,
    normalise_indian_mobile,
)
from config import WEEKDAYS, ClinicConfig, Doctor, fmt_day, fmt_time, hi_day, rupees_in_words
from greeting_cache import GreetingCache
from greeting_cache import play as play_recording
from hold import HoldController, HoldTimeout
from lines import inbound_greeting, line, outbound_greeting
from notify import notify, notify_bg
from prompts import build_instructions, closing_lines
from scheduling import AppointmentRequest, SchedulingBackend, SchedulingError, SlotTaken
from speech import speak

logger = logging.getLogger("frontdesk.agent")

TIMEOUT_MSG = (
    "LOOKUP TIMED OUT after the one-minute hold limit. Use the hold-script callback line "
    "('Since it is taking more time to get the exact status, allow me to get back to you...'), "
    "then call schedule_callback."
)
FAILED_MSG = (
    "The appointment system is not responding right now. Apologise briefly, use the hold-script "
    "callback line, then call schedule_callback. Do not invent availability."
)
PENDING_MSG = (
    "BOOKING PENDING: the appointment system did not confirm within one minute. Do NOT "
    "say it is booked and do NOT say it failed. Use the hold-script line ('Since it is "
    "taking more time to get the exact status, allow me to get back to you...'), say our "
    "team will confirm the appointment by SMS or a call shortly, then call schedule_callback."
)
# tools that belong to the other booking mode (config booking.mode)
_OTHER_MODE_TOOLS = {
    "slots": {"request_appointment"},
    "request": {"check_doctor_availability", "book_appointment"},
}


def _centres(cfg: ClinicConfig, d: Doctor) -> str:
    names = [loc.name for lid in d.location_ids if (loc := cfg.location(lid))]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _describe(cfg: ClinicConfig, d: Doctor) -> str:
    return f"{d.name} ({d.specialty}; at {_centres(cfg, d)})"


def visit_advice(cfg: ClinicConfig, fee_inr: int | None) -> str:
    """The confirmation's visit part (SOP: consultation charges, preparation instructions:
    bring reports, arrive early, dilation if needed). Unpublished facts are left out."""
    c = cfg.clinic
    parts = []
    if c.consultation.duration_text:
        parts.append(
            f"The consultation takes {c.consultation.duration_text}, so please plan your day."
        )
    if fee_inr:
        fee = f"The consultation charge is {rupees_in_words(fee_inr)} rupees"
        if c.consultation.fee_covers:
            fee += f", which covers {c.consultation.fee_covers}"
        if c.consultation.includes_registration:
            fee += " including registration"
        parts.append(fee + ".")
    else:
        parts.append(cfg.booking.confirmation_fee_line or cfg.booking.fee_note)
    parts.append(c.preparation or "Kindly report on time.")
    parts.append(c.dilation_advice)
    return " ".join(x.strip() for x in parts if x and x.strip())


def enquiry_note(state: CallState) -> dict | None:
    """End of call: a caller who gave details but got no appointment request or callback
    would be lost to the team (console call, 4 Nov: name, age, city and complaint taken,
    then the caller hung up and nothing was sent). SOP: always take a name and number, and
    log Name, Location and Eye complaint so the team calls back."""
    p = state.patient
    if state.booking or state.callbacks or state.do_not_call or state.transferred:
        return None
    if not (p.name or p.concern or p.city):
        return None
    return {
        "reason": "call ended without an appointment request or a callback",
        "note": front_desk_note(p),
        "phone": p.phone or normalise_indian_mobile(state.caller_number) or state.caller_number,
        "patient": p.for_log(),
        "emergency_flagged": state.emergency_flagged,
        "room": state.room,
    }


def contact_number(state: CallState, use_calling_number: bool) -> str:
    """The number the team calls back on, or a ToolError saying exactly what to ask.
    A number the caller gave wins; otherwise the number they are calling from is used
    without asking (it is read back in the confirmation, where they can change it)."""
    calling = normalise_indian_mobile(state.caller_number)
    if state.patient.phone and not use_calling_number:
        return state.patient.phone
    if calling:
        return calling
    if state.patient.phone:
        return state.patient.phone
    # console calls, withheld caller IDs and landlines: no usable number on this line
    raise ToolError(
        "the number they are calling from is not available on this line: apologise briefly "
        "and ask them to say a 10-digit mobile number"
    )


def front_desk_note(p) -> str:
    """The SOP's register line: "Pt Rahul Sharma — Gurgaon — CSR" (name, location, complaint)."""
    return " — ".join(
        [
            f"Pt {p.name or 'name not given'}",
            p.city or "city not given",
            p.concern or "complaint not given",
        ]
    )


class SpokenTextFilter:
    """Streaming filter: drops text inside (...) and [...], across chunk boundaries.
    Chat keeps (...) ("Green Park Main (Green Park metro)") and drops only [...] notes.
    calm=True (the phone): "!" becomes "." - the TTS voice raises its pitch and volume on an
    exclamation, and a front desk never exclaims at a caller."""

    def __init__(self, opening: str = "([", closing: str = ")]", *, calm: bool = False) -> None:
        self._OPEN, self._CLOSE = opening, closing
        self.depth = 0
        self.dropped = ""
        self.calm = calm

    def feed(self, text: str) -> str:
        out = []
        for ch in text:
            if ch in self._OPEN:
                self.depth += 1
            elif ch in self._CLOSE and self.depth:
                self.depth -= 1
            elif self.depth:
                self.dropped += ch
            elif self.calm and ch in "!\uff01":
                out.append(".")
            else:
                out.append(ch)
        return "".join(out)


NOTE_PREFIX = "[Reply language for this turn:"  # every note we add to the caller's message
LANGUAGE_NOTES = {
    "en": f"{NOTE_PREFIX} English. The caller is speaking English now.]",
    "hi": f"{NOTE_PREFIX} Hindi, in Devanagari. The caller is speaking Hindi now.]",
}


def language_note(lang: str) -> str:
    """Appended to the caller's message for the model; never spoken (brackets are filtered)."""
    return LANGUAGE_NOTES.get(lang, LANGUAGE_NOTES["en"])


_TURN_NOTE = re.compile(r"\s*" + re.escape(NOTE_PREFIX) + r"[^\]]*\]")


def strip_turn_note(text: str) -> str:
    """The patient's own words, without the note we add for the model (logs, dashboard, evals)."""
    return _TURN_NOTE.sub("", text or "").strip()


# chat only (the phone bot keeps call_state's own list): Hindi words people type in English
# letters. Words that are also English ("main", "to", "me", "par", "hi") are left out.
_HINGLISH_CHAT = re.compile(
    r"\b(mujhe|mujhko|hume|humein|mera|meri|mere|aap|aapka|aapki|aapke|apna|apni|kya|kab|"
    r"kaise|kahan|kitna|kitne|kyun|kyunki|hai|hain|hoon|hun|tha|thi|haan|han|nahi|nahin|nhi|"
    r"ji|chahiye|karna|karni|karo|karein|karenge|kariye|kar|karke|dijiye|kijiye|batao|"
    r"bataiye|bata|milna|milni|milega|se|ka|ke|ko|ki|mein|pe|isi|usi|wala|wali|liye|aur|bhi|"
    r"lekin|abhi|aaj|kal|parso|subah|shaam|dopahar|raat|theek|thik|accha|acha|achha|"
    r"aankh|ankh|aankhon|dikh|dikhai|dard|jal|raha|rahi|rahe|gaya|gayi|hua|hui|bete|beti|"
    r"naam|umar|saal|doctor\s+sahab)\b",
    re.IGNORECASE,
)


def chat_turn_note(state: CallState, cfg: ClinicConfig, text: str) -> str:
    """WhatsApp: the note on each patient message. Language AND script: a patient who types
    Hinglish ("mujhe kal appointment chahiye") gets Hinglish back, not Devanagari."""
    state.note_user_text(text)
    letters = [c for c in text if c.isalpha()]
    if devanagari_share(text) >= 0.3:
        state.chat_script = "devanagari"
    elif letters and len(_HINGLISH_CHAT.findall(text)) >= 2:
        state.language, state.chat_script = "hi", "latin"
    elif letters and detect_language(text) is not None:
        state.chat_script = "latin"
    if state.language == "hi":
        label = (
            "Hindi, in Devanagari script"
            if state.chat_script == "devanagari"
            else "Hinglish (Hindi written in English letters)"
        )
        who = "The patient is writing Hindi now."
    else:
        label, who = "English", "The patient is writing English now."
    now = cfg.now()  # with the day: a chat can run past midnight ("tomorrow" moves)
    return f"{NOTE_PREFIX} {label}. {who} Time now: {fmt_time(now)}, {fmt_day(now)}.]"


def turn_note(state: CallState, cfg: ClinicConfig) -> str:
    """The note added to every caller message: reply language + the clock time. The time
    lives here, not in the system prompt, so the prompt is the same for every call of the
    day and Claude's prompt cache is shared across calls (a ~10k-token write per call less)."""
    return f"{language_note(state.language)[:-1]} Time now: {fmt_time(cfg.now())}.]"


# The bot asked the caller for something: a question, or a polite request without a question
# mark ("कृपया मरीज़ की उम्र बताइए।", "Please tell me the patient's age.").
_ASKS = re.compile(
    r"[?？]|बताइए|बताइये|बताएँ|बताएं|बता दीजिए|बता सकते|बता सकती|"
    r"\bplease (?:tell|share|let me know|confirm|spell|say)\b|\bbataiye\b|\bbata (?:dijiye|sakte)",
    re.IGNORECASE,
)


def asks_caller(text: str) -> bool:
    return bool(_ASKS.search(text or ""))


def last_caller_message_id(chat_ctx) -> str | None:
    return next((m.id for m in reversed(chat_ctx.messages()) if m.role == "user"), None)


def caller_messages(session) -> int:
    """How many times the caller has spoken (typed) so far: works for phone and text calls."""
    return sum(1 for m in session.history.messages() if m.role == "user")


class GuardedEndCall(EndCallTool):
    """end_call that will not hang up in the same turn as an emergency: the caller must hear
    the advice and get to answer (a live eval caught the model calling flag_emergency and
    end_call back to back on a chemical-splash call)."""

    async def _end_call(self, ctx: RunContext) -> Any | None:
        state = ctx.userdata
        if state.emergency_turn is not None and state.emergency_turn == caller_messages(
            ctx.session
        ):
            raise ToolError(
                "Do not end the call now: give the emergency advice and ask if there is "
                "anything else you can do for them."
            )
        asked = getattr(ctx.session.current_agent, "_asked_after", None)
        if asked is not None and asked == last_caller_message_id(ctx.session.history):
            # seen in an eval: the model asked for the age, then called end_call in the
            # next step. Never hang up on a caller who is about to answer.
            logger.warning("end_call refused: the caller has not answered the last question")
            return llm.ToolResult(
                "Not ended: you have just asked the caller something. Wait for their answer.",
                reply_required=False,
            )
        return await super()._end_call(ctx)


class FrontDeskAgent(Agent):
    def __init__(
        self,
        cfg: ClinicConfig,
        *,
        scheduler: SchedulingBackend,
        direction: Literal["inbound", "outbound"] = "inbound",
        lead: dict | None = None,
        background: BackgroundAudioPlayer | None = None,
        chat_ctx: ChatContext | None = None,
        greet: bool = True,
        wait_for_greeting: bool = False,
        greetings: GreetingCache | None = None,
        channel: Literal["phone", "whatsapp"] = "phone",
    ) -> None:
        """greet: speak the scripted greeting on enter (inbound).
        wait_for_greeting: outbound - stay silent (no LLM replies) until the caller-side
        code speaks the greeting after the callee answers and sets state.greeted."""
        self.cfg = cfg
        self.channel = channel
        self._asked_after: str | None = None  # caller message we last asked them something after
        self.direction = direction
        self.lead = lead or {}
        self.scheduler = scheduler
        # a chat has no hold announcements or hold music: just the time limit
        hold_cfg = (
            cfg.hold.model_copy(update={"first_notice_after_s": 1e6})
            if channel == "whatsapp"
            else cfg.hold
        )
        self.hold = HoldController(hold_cfg, background)
        self._greet = greet
        self.greetings = greetings
        self._wait_for_greeting = wait_for_greeting
        self._state: CallState | None = None
        persona = (
            cfg.personas.inbound.name if direction == "inbound" else cfg.personas.outbound.name
        )
        self.persona = persona
        self.closings = closing_lines(cfg, persona, channel)

        super().__init__(
            instructions=build_instructions(
                cfg, direction=direction, lead=self.lead, channel=channel
            ),
            chat_ctx=chat_ctx,
            tools=[]  # a chat has no call to end
            if channel == "whatsapp"
            else [
                GuardedEndCall(
                    extra_description=(
                        "Only after the caller confirms there is nothing else. Never during a "
                        "hold, a transfer, or while an emergency is being handled."
                    ),
                    end_instructions=f'Say exactly this closing line and nothing else: "{self.closings["general"]}"',
                    ignore_on_enter=True,
                    on_tool_called=self._on_end_call,
                    on_tool_completed=self._pick_closing,
                )
            ],
        )
        # one booking flow per call: slot booking or appointment request (config booking.mode)
        drop = _OTHER_MODE_TOOLS[cfg.booking.mode]
        self._tools = [t for t in self._tools if getattr(t, "id", None) not in drop]
        if chat_ctx:
            self._chat_ctx = chat_ctx.copy(tools=self._tools)

    def _one_doctor(self, spoken: str) -> Doctor:
        """The doctor the caller means, or a ToolError telling the model what to ask."""
        hits = self.cfg.match_doctors(spoken)
        if len(hits) == 1:
            return hits[0]
        if hits:
            names = "; ".join(_describe(self.cfg, d) for d in hits[:6])
            raise ToolError(
                f"several doctors match {spoken!r}: {names}. Ask the caller which one they mean."
            )
        raise ToolError(
            f"no doctor named {spoken!r} at {self.cfg.clinic.name}. Tell the caller right away "
            "and offer the listed doctors for what they need."
        )

    @property
    def closing_line(self) -> str:
        """The scripted closing for this call: booking / general (inbound) or outbound."""
        if self.direction == "outbound":
            return self.closings["outbound"]
        booked = bool(self._state and self._state.booking)
        return self.closings["booking" if booked else "general"]

    # -- lifecycle -----------------------------------------------------------------
    async def on_enter(self) -> None:
        self._state = self.session.userdata
        if not self._greet:
            if not self._wait_for_greeting:
                self.session.userdata.greeted = True
            return
        now = self.cfg.now()
        c = self.cfg.clinic
        if self.direction == "inbound":
            text = inbound_greeting(
                clinic=c.name,
                persona=self.persona,
                now=now,
                disclose_ai=c.disclose_ai_in_greeting,
                team=c.team,
            )
        else:
            text = outbound_greeting(
                clinic=c.name,
                persona=self.persona,
                now=now,
                disclose_ai=c.disclose_ai_in_greeting,
                team=c.team,
                lead_source=self.lead.get("lead_source"),
            )
        self.session.userdata.greeted = True
        frames = self.greetings.load(text) if self.greetings else None
        if frames:
            # recorded greeting: first audio at once instead of waiting on a cold TTS connection
            speak(self.session, text, audio=play_recording(frames), allow_interruptions=True)
            return
        handle = speak(self.session, text, allow_interruptions=True)
        if self.greetings:
            self.greetings.record_after(handle, text)

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        state: CallState = self.session.userdata
        state.silence_prompts = 0
        if new_message.text_content:
            state.note_user_text(new_message.text_content)
            state.language_msg_id = new_message.id
            # The language is decided here, every turn, not left to the model: on a real call
            # Claude kept answering in Hindi after the caller went back to English.
            new_message.content.append(turn_note(state, self.cfg))
        if not state.greeted:
            # outbound: the callee's "Hello?" must not trigger an LLM reply before our greeting
            raise StopResponse()

    async def _on_end_call(self, ev) -> None:  # EndCallTool callback
        state: CallState = self.session.userdata
        state.ending = True

    async def _pick_closing(self, ev) -> None:  # EndCallTool: choose the scripted closing line
        if self._said_closing(getattr(self, "_step_text", "")):
            # the model already said the closing in this response, before calling end_call:
            # no second reply (an eval caught the closing line spoken twice), just hang up
            ev.output = None
            return
        if self.session.userdata.language == "en":
            # fixed English text: the code says it, no second LLM round trip (~1.3 s of
            # silence before the goodbye). The session drains queued speech before closing.
            speak(self.session, self.closing_line, allow_interruptions=False)
            ev.output = None
            return
        ev.output = f'Say exactly this closing line and nothing else: "{self.closing_line}"'

    def _said_closing(self, text: str) -> bool:
        """Did the model already say the scripted thanks? Any other goodbye ("Take care")
        still gets the closing line: the SOP closes every call with "Thank you for calling"
        (console call, 4 Nov: "You can hang up" got a goodbye without it)."""
        said = text.strip().lower()
        thanks = self.closing_line.split(".")[0].strip().lower()  # "thank you for calling X"
        hindi = self.cfg.clinic.spoken_name_hi
        return bool(said) and (thanks in said or (hindi in text and "धन्यवाद" in text))

    def _sync_language(self, chat_ctx: ChatContext) -> None:
        """Phone calls set the language in on_user_turn_completed; a turn that arrives another
        way (typed text, generate_reply) is caught here, before any tool reads it."""
        state: CallState = self.session.userdata
        if self.channel == "whatsapp":
            return  # the chat engine sets language and script itself (chat_turn_note)
        last = next(
            (m for m in reversed(chat_ctx.items) if getattr(m, "role", None) == "user"), None
        )
        if last is not None and last.id != state.language_msg_id:
            state.language_msg_id = last.id
            state.note_user_text(last.text_content or "")

    # -- never-silent guard ------------------------------------------------------------
    async def llm_node(self, chat_ctx, tools, model_settings):
        """Pass the model's stream through, with two guards:
        - strip (...) / [...] meta-text: everything written is spoken, and a live eval
          caught the model "thinking aloud" in brackets;
        - if it produced NOTHING (truncated tool call, provider empty-response bugs),
          speak a short line instead of leaving dead air."""
        self._sync_language(chat_ctx)
        self._step_text = ""  # what this response has said so far (see update_patient_details)
        turn = last_caller_message_id(chat_ctx)  # the history may not have it yet (text mode)
        produced = False
        meta = (
            SpokenTextFilter("[", "]")
            if self.channel == "whatsapp"
            else SpokenTextFilter(calm=True)
        )
        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            if isinstance(chunk, str):
                chunk = meta.feed(chunk)
                produced = produced or bool(chunk.strip())
                self._step_text += chunk
                self._note_ask(turn)
                if chunk:
                    yield chunk
                continue
            delta = getattr(chunk, "delta", None)
            if delta is not None and delta.content:
                clean = meta.feed(delta.content)
                chunk = chunk.model_copy(
                    update={"delta": delta.model_copy(update={"content": clean})}
                )
                delta = chunk.delta
                self._step_text += clean
                self._note_ask(turn)
            if delta is not None and ((delta.content or "").strip() or delta.tool_calls):
                produced = True
            yield chunk
        if meta.dropped:
            logger.warning("dropped unspoken meta-text from LLM output: %r", meta.dropped[:200])
        state: CallState = self.session.userdata
        if not produced and not (state.ending or state.transferred):
            state.errors.append("empty LLM turn replaced by fallback line")
            logger.warning("LLM returned an empty turn; speaking fallback line")
            yield line("empty_turn", state.language)

    # -- tools ---------------------------------------------------------------------
    # Optional text arguments default to "" (not None): Anthropic refuses a request whose tools
    # have more than 16 nullable parameters in total, and then every reply came from the
    # last backup model. "" means "not said", the same as None did.
    @function_tool
    async def update_patient_details(
        self,
        ctx: RunContext[CallState],
        patient_type: str = "",
        name: str = "",
        age: int | None = None,
        mobile_number: str = "",
        location_id: str = "",
        preferred_doctor: str = "",
        concern: str = "",
        uhid: str = "",
        city: str = "",
    ) -> str:
        """Save patient details as soon as the caller gives them (the query form). Call it
        again whenever you learn something new. Only pass what the caller actually said.
        Write your whole reply to the caller FIRST, then call this in the same response:
        calling it before you speak makes the caller wait for a second round trip.

        Args:
            patient_type: "new" for a first visit, "existing" if they have visited before.
            name: The patient's name (may differ from the caller).
            age: The patient's age in years.
            mobile_number: Patient's contact number, as spoken.
            location_id: Preferred centre id from the clinic facts (only if there are several).
            preferred_doctor: A specific doctor the caller asked for.
            concern: The eye complaint in the caller's own words, briefly.
            uhid: Hospital registration ID, only for existing patients who have it.
            city: The city or area the patient lives in or is coming from.
        """
        p = ctx.userdata.patient
        notes: list[str] = []
        if city and city.strip():
            p.city = city.strip()
        if patient_type:
            # plain string, not an enum: Anthropic rejects optional enums in tool schemas
            kind = patient_type.strip().lower()
            if kind in ("new", "first", "first visit", "first time"):
                p.patient_type = "new"
            elif kind in ("existing", "old", "returning", "follow up", "follow-up", "repeat"):
                p.patient_type = "existing"
            else:
                notes.append("patient_type must be new or existing; ask the caller")
        if name and name.strip():
            p.name = name.strip()
        if age is not None:
            if 0 <= age <= 120:
                p.age = age
            else:
                notes.append(f"age {age} looks wrong, confirm it with the caller")
        if mobile_number:
            norm = normalise_indian_mobile(mobile_number)
            if norm:
                p.phone = norm
            else:
                notes.append("that mobile number is not a valid 10-digit Indian mobile, ask again")
        if location_id:
            if self.cfg.location(location_id):
                p.preferred_location_id = location_id
            else:
                valid = ", ".join(loc.id for loc in self.cfg.locations)
                notes.append(f"unknown location id; valid ids: {valid}")
        if preferred_doctor:
            hits = self.cfg.match_doctors(preferred_doctor)
            if len(hits) == 1:
                p.preferred_doctor = hits[0].id
            elif hits:
                names = "; ".join(_describe(self.cfg, d) for d in hits[:6])
                notes.append(f"several doctors match {preferred_doctor!r}: {names}; ask which one")
            else:
                notes.append(
                    f"no doctor named {preferred_doctor!r} in the clinic facts; tell the caller "
                    "and offer the doctors who are listed"
                )
        if concern:
            p.concern = concern.strip()
        if uhid:
            p.uhid = uhid.strip()

        missing = p.missing_essentials()
        out = "Saved."
        if missing:
            out += (
                " Still needed (gently, one question per reply; name and age may be asked "
                "together): " + ", ".join(missing) + "."
            )
        if notes:
            out += " Note: " + "; ".join(notes) + "."
        # the reply comes right after this result: restate the one-question rule and the
        # caller's language here (evals caught two questions at once, and English to a
        # Hinglish caller, straight after this tool)
        lang = "Hindi (Devanagari)" if ctx.userdata.language == "hi" else "English"
        out += f" Your reply: in {lang}, with only one question."
        if not notes and self.asked_already():
            # one step: the model already said its reply and next question in this same
            # response, so no second LLM round trip (measured ~2 s saved on every detail turn)
            return llm.ToolResult(out, reply_required=False)
        return out

    def asked_already(self) -> bool:
        """Did the current LLM response already ask the caller something? A polite request
        ("कृपया मरीज़ की उम्र बताइए।") counts too: asking twice in a row sounds pushy."""
        return asks_caller(getattr(self, "_step_text", "").strip())

    def _note_ask(self, turn: str | None) -> None:
        """Remember that the caller has been asked something since their last message, so
        end_call waits for the answer."""
        if turn is not None and self._asked_after != turn and self.asked_already():
            self._asked_after = turn

    @function_tool
    async def check_doctor_availability(
        self,
        ctx: RunContext[CallState],
        location_id: str = "",
        specialty: str = "",
        doctor_name: str = "",
        date_iso: str = "",
    ) -> str:
        """Find open appointment slots. Use the caller's preferred location, doctor or
        specialty if known. Pass date_iso only if the caller asked for a specific day.

        Args:
            location_id: Centre id from the clinic facts.
            specialty: Specialty as written in the clinic facts, e.g. "Retina".
            doctor_name: A specific doctor's name.
            date_iso: Specific day as YYYY-MM-DD.
        """
        on_date = None
        if date_iso:
            try:
                on_date = date.fromisoformat(date_iso)
            except ValueError:
                raise ToolError("date_iso must be YYYY-MM-DD") from None
            if on_date < self.cfg.now().date():
                raise ToolError("that date is in the past; ask the caller for another day")
        if location_id and not self.cfg.location(location_id):
            valid = ", ".join(loc.id for loc in self.cfg.locations)
            raise ToolError(f"unknown location_id; valid ids: {valid}")
        if doctor_name:
            doctor_name = self._one_doctor(doctor_name).id

        try:
            slots, held = await self.hold.run(
                ctx,
                self.scheduler.find_slots(
                    location_id=location_id,
                    specialty=specialty,
                    doctor=doctor_name,
                    on_date=on_date,
                ),
                reason="availability",
            )
        except HoldTimeout:
            ctx.userdata.errors.append("availability lookup timed out")
            return TIMEOUT_MSG
        except SchedulingError as e:
            ctx.userdata.errors.append(f"availability lookup failed: {e}")
            return FAILED_MSG

        if not slots:
            where = f" on {fmt_day(on_date)}" if on_date else " in the next three weeks"
            return (
                f"No open slots{where} for that request.{held.note()} Offer another day, location "
                "or doctor, or a callback from the appointments desk."
            )
        listing = "; ".join(
            f"[slot_id {s.slot_id}] {s.doctor_name} ({s.specialty}) at {s.location_name}, "
            f"{s.spoken()} (in Hindi say: {s.spoken_hi()})"
            for s in slots
        )
        return (
            f"Open slots: {listing}.{held.note()} Offer two or three of these in plain words "
            "(no slot ids) and ask which suits the caller. For Hindi, copy the given Hindi "
            "day/time words exactly; never convert times yourself."
        )

    @function_tool
    async def book_appointment(
        self,
        ctx: RunContext[CallState],
        slot_id: str,
        use_calling_number: bool = False,
    ) -> str:
        """Book the slot the caller chose. Call only after the caller has agreed to a
        specific slot and you have the patient's name and mobile number (or the caller
        agreed to use the number they are calling from).

        Args:
            slot_id: The slot_id from check_doctor_availability.
            use_calling_number: True if the caller agreed to use the number they called from.
        """
        state = ctx.userdata
        p = state.patient
        if state.booking and state.booking.get("slot_id") == slot_id:
            return "Already booked this exact slot; do not book again. Give the confirmation."
        if not p.name:
            raise ToolError("patient name missing: ask for it and save it first")
        phone = contact_number(state, use_calling_number)

        try:
            booking, held = await self.hold.run(
                ctx,
                self.scheduler.book(slot_id=slot_id, patient_name=p.name, patient_phone=phone),
                reason="booking",
                cancel_on_timeout=False,  # a slow backend may still save it: never double-book
            )
        except HoldTimeout as t:
            state.errors.append("booking not confirmed within the hold limit")
            if t.pending is not None:
                t.pending.add_done_callback(lambda task: self._late_booking(state, task))
            return PENDING_MSG
        except SlotTaken as e:
            return f"Could not book: {e}. Apologise and offer the next available slots."
        except SchedulingError as e:
            state.errors.append(f"booking failed: {e}")
            return FAILED_MSG + " The appointment is NOT booked; say so honestly."

        s = booking.slot
        state.booking = {
            "booking_id": booking.booking_id,
            "slot_id": s.slot_id,
            "doctor": s.doctor_name,
            "specialty": s.specialty,
            "location": s.location_name,
            "start": s.start.isoformat(),
            "fee_inr": booking.fee_inr,
            "patient": p.name,
            "phone": mask_phone(phone),
        }
        loc = self.cfg.location(s.location_id)
        sms_sent = False
        if self.cfg.clinic.sms_confirmation:
            sms_sent = await notify(
                "appointment_confirmation_sms",
                {
                    "to": phone,
                    "booking_id": booking.booking_id,
                    "patient": p.name,
                    "doctor": s.doctor_name,
                    "location": s.location_name,
                    "address": loc.area if loc else "",  # script: SMS "with landmark"
                    "start": s.start.isoformat(),
                    "fee_inr": booking.fee_inr,
                },
                timeout=2.0,
            )
        sms_line = (
            f"After this call you will receive an SMS from {self.cfg.clinic.name} confirming the "
            "appointment with the address."
            if sms_sent
            else "Our team will send you the appointment details with the address shortly."
        )
        confirmation = (
            f"We have booked your appointment with {s.doctor_name}, {s.specialty}, at our "
            f"{s.location_name} on {fmt_day(s.start)} at {fmt_time(s.start)}. "
            f"{visit_advice(self.cfg, booking.fee_inr)} {sms_line} "
            "Is there anything else I may assist you with?"
        )
        return (
            f"BOOKED.{held.note()} Now say this confirmation as ONE reply, word for word in "
            "English, or as a faithful natural Hindi rendering if the caller speaks Hindi or "
            f'Hinglish (keep every fact; add nothing): "{confirmation}" For Hindi, the day and '
            f'time are exactly: "{s.spoken_hi()}".'
        )

    @function_tool
    async def request_appointment(
        self,
        ctx: RunContext[CallState],
        preferred_day: str,
        location_id: str = "",
        preferred_time: str = "",
        doctor_name: str = "",
        specialty: str = "",
        date_iso: str = "",
        use_calling_number: bool = False,
    ) -> str:
        """Register the appointment the caller wants. Doctors' OPD days and times are not
        available to you: our team calls the patient back to confirm the exact time. Call
        only after the caller has chosen a preferred day, and you have the patient's name and
        contact number (or the caller agreed to use the number they are calling from).

        Args:
            preferred_day: The day the caller wants, in their words ("tomorrow", "Monday",
                "any day this week").
            preferred_time: The part of the day they prefer, in their words ("morning",
                "after 4 pm"), if they said one.
            doctor_name: A specific doctor the caller asked for.
            specialty: A speciality the caller asked for, as written in the clinic facts.
                Never pick one from the symptoms.
            date_iso: The exact date as YYYY-MM-DD, only if the caller named a specific day.
            use_calling_number: True if the caller agreed to use the number they called from.
            location_id: Centre id from the clinic facts; leave empty when there is one clinic.
        """
        state = ctx.userdata
        p = state.patient
        cfg = self.cfg
        if not location_id:
            if len(cfg.locations) != 1:
                raise ToolError("location_id missing: ask which centre suits the caller")
            location_id = cfg.locations[0].id
        if not p.name:
            raise ToolError("patient name missing: ask for it and save it first")
        phone = contact_number(state, use_calling_number)
        loc = cfg.location(location_id)
        if loc is None:
            valid = ", ".join(f"{x.id} ({x.name})" for x in cfg.locations)
            raise ToolError(f"unknown location_id; valid ids: {valid}")

        doctor = self._one_doctor(doctor_name) if doctor_name else None
        spec = None
        if specialty:
            spec = cfg.match_specialty(specialty)
            if spec is None and doctor is None:
                names = ", ".join(dict.fromkeys(s for d in cfg.doctors for s in d.specialties))
                raise ToolError(f"unknown specialty {specialty!r}; specialities: {names}")
        if doctor is not None:
            if not doctor.sits_at(loc.id):
                raise ToolError(
                    f"{doctor.name} sees patients only at {_centres(cfg, doctor)}, not at "
                    f"{loc.name}. Tell the caller and ask which centre suits them."
                )
            if spec not in doctor.specialties:
                spec = None  # the doctor decides; their own specialities are on record
        elif spec is not None and not any(
            d.sits_at(loc.id) and spec in d.specialties for d in cfg.doctors
        ):
            where = sorted(
                {c for d in cfg.doctors if spec in d.specialties for c in d.location_ids}
            )
            names = ", ".join(x.name for x in cfg.locations if x.id in where)
            raise ToolError(
                f"no {spec} doctor at {loc.name}; {spec} is available at: {names}. Tell the "
                "caller and ask which centre suits them."
            )

        if doctor is None and len(cfg.doctors) == 1 and cfg.doctors[0].sits_at(loc.id):
            # one doctor: name him in the request and the confirmation ("one of our doctors"
            # would sound as if there were others)
            doctor = cfg.doctors[0]
            if spec not in doctor.specialties:
                spec = None

        on_date = None
        if date_iso:
            try:
                on_date = date.fromisoformat(date_iso)
            except ValueError:
                raise ToolError("date_iso must be YYYY-MM-DD") from None
            if on_date < cfg.now().date():
                raise ToolError("that date is in the past; ask the caller for another day")
            if loc.open_days and WEEKDAYS[on_date.weekday()] not in loc.open_days:
                raise ToolError(
                    f"the {loc.name} centre is open {', '.join(loc.open_days)} only; "
                    f"{fmt_day(on_date)} is closed. Ask the caller for another day."
                )

        request = AppointmentRequest(
            request_id="",
            patient_name=p.name,
            patient_phone=phone,
            location_id=loc.id,
            location_name=loc.name,
            doctor_id=doctor.id if doctor else None,
            doctor_name=doctor.name if doctor else None,
            specialty=spec,
            on_date=on_date,
            preferred_day=preferred_day.strip() or None,
            preferred_time=(preferred_time or "").strip() or None,
            patient_type=p.patient_type,
            age=p.age,
            concern=p.concern,
        )
        try:
            request, held = await self.hold.run(
                ctx,
                self.scheduler.request_appointment(request),
                reason="booking",
                cancel_on_timeout=False,  # a slow backend may still save it: never duplicate
            )
        except HoldTimeout:
            state.errors.append("appointment request not saved within the hold limit")
            return PENDING_MSG
        except SchedulingError as e:
            state.errors.append(f"appointment request failed: {e}")
            return FAILED_MSG + " The appointment request is NOT registered; say so honestly."

        state.booking = {
            "request_id": request.request_id,
            "status": "requested",
            "doctor": request.doctor_name,
            "specialty": request.specialty,
            "location": loc.name,
            "date": on_date.isoformat() if on_date else None,
            "preferred_day": request.preferred_day,
            "preferred_time": request.preferred_time,
            "patient": p.name,
            "phone": mask_phone(phone),
        }
        notify_bg(
            "appointment_request",
            {
                **request.to_json(),
                "to": phone,  # the team calls this number back (masked in the local log)
                "patient_phone": mask_phone(phone),
                "location_address": loc.address or loc.area,
                "city": p.city,
                "note": front_desk_note(p),
                "room": state.room,
            },
        )
        if doctor is not None:
            who = f"with {doctor.name}"
        elif spec:
            who = f"with one of our {spec} doctors"
        else:
            who = "with one of our doctors"
        when = fmt_day(on_date) if on_date else request.preferred_day or "the earliest day"
        if request.preferred_time:
            when += f", {request.preferred_time}"
        # spoken digit by digit on a call; written normally in a chat
        last4 = phone[-4:] if self.channel == "whatsapp" else " ".join(phone[-4:])
        # the calling number was used without asking: let them change it in the same breath
        other_number = (
            " If you would like us to call a different number, please tell me."
            if phone != p.phone and self.channel != "whatsapp"
            else ""
        )
        place = "clinic" if len(cfg.locations) == 1 else "centre"
        # SOP: close by repeating the details back, with the preparation instructions
        confirmation = (
            f"I have registered an appointment request for {p.name} {who} at our {loc.name} "
            f"{place}, for {when}. Our team will call you on your number ending {last4} to "
            f"confirm the exact time.{other_number} {visit_advice(cfg, loc.consultation_fee_inr)} "
            "Is there anything else I may assist you with?"
        )
        if state.language == "en":
            # English: the code speaks the scripted confirmation itself, word for word.
            # Asking the model to repeat a 120-word text cost a second round trip (~1.5 s
            # before the first word) and ~150 output tokens per booking.
            if held.held:
                confirmation = "Thank you for your patience. " + confirmation
            speak(ctx.session, confirmation, allow_interruptions=True)
            return llm.ToolResult(
                "REQUEST REGISTERED. The confirmation was just spoken to the caller, word for "
                "word: do not repeat it. Wait for their answer to 'anything else'.",
                reply_required=False,
            )
        hindi_day = f' For Hindi, the day is exactly: "{hi_day(on_date)}".' if on_date else ""
        if self.channel == "whatsapp":
            hindi_day = (
                f' In Devanagari the day is exactly: "{hi_day(on_date)}". For a Hinglish writer, '
                "write the whole confirmation in Hinglish (English letters)."
                if on_date
                else " For a Hinglish writer, write the whole confirmation in Hinglish."
            )
        return (
            f"REQUEST REGISTERED (the exact time is not confirmed yet).{held.note()} Now say this "
            "as ONE reply, word for word in English (do not shorten or drop any part), or as a "
            "faithful natural Hindi rendering if "
            "the caller speaks Hindi or Hinglish (keep every fact; add nothing; never say the "
            f'appointment or time is confirmed or booked): "{confirmation}"{hindi_day}'
        )

    @function_tool
    async def schedule_callback(
        self,
        ctx: RunContext[CallState],
        reason: str,
        preferred_time: str = "",
        mobile_number: str = "",
    ) -> str:
        """Record that the team must call this person back (lookup took too long, transfer
        failed, department closed, caller asked, or bad time on an outbound call).

        Args:
            reason: What the callback is about, with enough detail for a colleague.
            preferred_time: When the caller would like the call, in their words.
            mobile_number: Number to call back, if different from the calling number.
        """
        state = ctx.userdata
        phone = (
            normalise_indian_mobile(mobile_number or "")
            or state.patient.phone
            or normalise_indian_mobile(state.caller_number)
            or state.caller_number
        )
        entry = {
            "reason": reason,
            "preferred_time": preferred_time,
            "phone": mask_phone(phone or ""),
        }
        state.callbacks.append(entry)
        notify_bg(
            "callback_request",
            {
                **entry,
                "phone": phone,
                "note": front_desk_note(state.patient),
                "patient": state.patient.for_log(),
                "room": state.room,
            },
        )
        return "Callback recorded. Tell the caller the team will call them back, then ask if there is anything else."

    @function_tool
    async def mark_do_not_call(self, ctx: RunContext[CallState], reason: str = "") -> str:
        """The person asked not to be called again. Record it so nobody calls them back.

        Args:
            reason: Anything they said about why, in their words.
        """
        state = ctx.userdata
        state.do_not_call = True
        notify_bg(
            "do_not_call",
            {"phone": state.caller_number, "reason": reason, "room": state.room},
        )
        return (
            "Noted as do-not-call. Apologise for the disturbance and close politely with end_call."
        )

    @function_tool
    async def flag_emergency(self, ctx: RunContext[CallState], description: str) -> str:
        """Call immediately when the caller describes any emergency red flag.

        Args:
            description: What the caller said about the emergency.
        """
        self._flag(ctx.userdata, description)
        return "Emergency flagged to staff. Now give the emergency advice and offer the emergency desk transfer."

    def _flag(self, state: CallState, description: str) -> None:
        if state.emergency_flagged:
            return
        state.emergency_flagged = True
        state.emergency_turn = caller_messages(self.session)
        notify_bg(
            "emergency_alert",
            {"description": description, "room": state.room, "phone": state.caller_number},
        )

    def _late_booking(self, state: CallState, task) -> None:
        """A booking that completed after the caller was told it was pending."""
        if task.cancelled() or task.exception() is not None:
            notify_bg(
                "booking_unconfirmed", {"room": state.room, "patient": state.patient.for_log()}
            )
            return
        b = task.result()
        state.booking = {"booking_id": b.booking_id, "slot_id": b.slot.slot_id, "late": True}
        notify_bg(
            "booking_confirmed_late",
            {
                "room": state.room,
                "booking_id": b.booking_id,
                "to": b.patient_phone,
                "doctor": b.slot.doctor_name,
                "start": b.slot.start.isoformat(),
            },
        )

    @function_tool
    async def transfer_to_department(
        self, ctx: RunContext[CallState], department_id: str, summary: str
    ) -> str:
        """Warm-transfer the caller to a colleague. You first brief the colleague with the
        summary; the caller is connected only after the colleague agrees. Say the transfer
        line from the script to the caller before calling this.

        Args:
            department_id: Department id from the clinic facts.
            summary: Complete concern for the colleague: patient name, age, new or existing,
                the problem, what they want, anything already done on this call.
        """
        state = ctx.userdata
        dept = self.cfg.department(department_id)
        if dept is None:
            valid = ", ".join(d.id for d in self.cfg.departments)
            raise ToolError(f"unknown department_id; valid ids: {valid}")
        attempt = {"department": dept.id, "summary": summary, "result": None}
        state.transfers.append(attempt)
        emergency = dept.id == "emergency"
        if emergency:
            self._flag(state, summary)  # staff are alerted even if the transfer fails

        if dept.hours is not None and not dept.hours.is_open(self.cfg.now()):
            attempt["result"] = "closed"
            return (
                f"{dept.name} is closed right now (open {', '.join(dept.hours.days)} "
                f"{dept.hours.start}-{dept.hours.end}). Apologise ('I am sorry, ...'), tell the caller, "
                "offer a callback and use schedule_callback."
            )
        trunk_ready = bool(os.getenv("LIVEKIT_SIP_OUTBOUND_TRUNK"))
        if not dept.transfer_enabled or not trunk_ready:
            attempt["result"] = "not_configured"
            if emergency:
                contact = self.cfg.emergency.contact_line
                return (
                    "Live transfer to the emergency desk is not possible right now, but staff have "
                    "been alerted. Do NOT offer a callback as the main help: repeat clearly that they "
                    "must go to the nearest eye or hospital emergency right now"
                    + (f", and say: {contact}" if contact else "")
                    + ". Stay calm and brief."
                )
            return (
                f"Live transfer to {dept.name} is not available right now. Apologise, say a "
                "colleague will call them back, and use schedule_callback with the summary."
            )

        briefing = WorkflowInstructions(
            extra=(
                f"You are {self.persona} from the {self.cfg.clinic.name} front desk, calling the "
                f"{dept.name}. Speak English unless the colleague answers in Hindi. Brief them in "
                f"under twenty seconds using this summary: {summary}. Answer their questions from "
                "the conversation history. When they agree to take the call, call "
                "connect_to_caller. If they cannot take it, call decline_transfer with the reason."
            )
        )
        state.on_hold = True
        try:
            async with ctx.foreground():
                await WarmTransferTask(
                    sip_call_to=dept.sip_call_to,
                    ringing_timeout=self.cfg.transfer.ringing_timeout_s,
                    chat_ctx=self.chat_ctx,
                    instructions=briefing,
                    hold_audio=AudioConfig(
                        BuiltinAudioClip.HOLD_MUSIC, volume=max(self.cfg.hold.music_volume, 0.05)
                    ),
                )
        except ToolError as e:
            attempt["result"] = f"failed: {e}"
            logger.warning("transfer to %s failed: %s", dept.id, e)
            if emergency:
                return (
                    f"Emergency desk did not pick up ({e}); staff have been alerted. Repeat that they "
                    "must go to the nearest eye or hospital emergency right now."
                )
            return (
                f"Transfer did not go through ({e}). The colleague is not available at the moment. "
                "Thank the caller for their patience, apologise, offer a callback and use "
                "schedule_callback with the summary."
            )
        except Exception as e:
            attempt["result"] = f"error: {e}"
            logger.exception("transfer to %s crashed", dept.id)
            return (
                "Transfer could not be started due to a technical issue. Apologise, offer a "
                "callback and use schedule_callback with the summary."
            )
        finally:
            state.on_hold = False

        # Connected: the colleague and caller are now together in the caller's room.
        attempt["result"] = "connected"
        state.transferred = True
        ctx.session.input.set_audio_enabled(False)
        ctx.session.output.set_audio_enabled(False)
        logger.info("transfer to %s connected; agent is now silent", dept.id)
        raise StopResponse()
