"""Claude as the agent's assistant: from the live transcript of a call a person is
handling, what they could say next, the clinic facts they may need, the booking details
heard so far, and alerts (an emergency, an upset patient). Called by the dashboard hub.

Grounded in the same clinic facts and call script as Clara (config/clinic.yaml,
prompts.clinic_facts), with the same safety rules: no diagnosis, no medicine advice, no
invented fee or timing. The agent decides; nothing here is heard by the patient.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
from typing import Any

from call_cost import LLM_PRICES

logger = logging.getLogger("frontdesk.assist")

# Haiku by default: about 1 s to the first words (Sonnet effort low: about 2.5 s) at a tenth of
# the cost. ASSIST_MODEL=claude-sonnet-5-5 for the best wording; ASSIST_KNOWLEDGE_MODEL sets the
# model of the general-knowledge answer separately (default: the same one).
ASSIST_MODEL = os.getenv("ASSIST_MODEL", "claude-haiku-5-5")
ASSIST_KNOWLEDGE_MODEL = os.getenv("ASSIST_KNOWLEDGE_MODEL", "")
OUTCOMES = (
    "Appointment request",
    "Enquiry",
    "Emergency advice",
    "Callback",
    "Transferred",
    "No reply",
)
FORM_KEYS = ("patient_type", "name", "age", "city", "centre", "doctor", "day", "time", "concern")
KINDS = ("answer", "ask", "empathy", "emergency", "closing")

# Demo Clinic Front Desk SOP: Call & Enquiry Handling
STEPS = """1. Greet: "Namaste, Demo Eye and Retina Clinic. This is <name> speaking. I am part of Dr. Arjun Mehta's team. How may I help you today?"
2. Understand the request: appointment, general query, emergency, or billing. Reduce anxiety, build trust: clarity (what is happening), control (the next step), care (reassurance).
3. Take the patient's name, age, contact number, city or area, and eye complaint, one question at a time. Always take a name and number, even if they say they will call back.
4. Never quote a price before understanding the problem; then share the consultation charge if they ask. No medical advice: offer a doctor consultation.
5. Ask: "Can I book an appointment with the doctor for you?" Agree the day and morning or afternoon; the team confirms the exact time.
6. Preparation: bring previous reports and prescriptions, come a little early, the eyes may be dilated so bring someone along.
7. Billing (bills, insurance, TPA): pass it to the billing desk; never add double payments. Someone unavailable: take a message and assure a callback.
8. Close by repeating the details, ask if there is anything else, and say "Thank you for calling"."""


def _system(cfg: Any) -> str:
    from prompts import clinic_facts

    contact = f" {cfg.emergency.contact_line}" if cfg.emergency.contact_line else ""
    return f"""You silently assist a human front-desk agent of {cfg.clinic.name}, an eye hospital, during a live phone call with a patient or a patient's relative. You read a transcript made by speech recognition: it has errors, and the two voices carry speaker numbers that may be wrong. Only the agent sees your notes, on a screen, at a glance, while talking. The patient never hears you.

Answer with JSON only, no other text, keys in exactly this order (the agent's screen shows "say" the moment it is written):
{{"say": "<what the agent could say next>",
 "kind": "answer" | "ask" | "empathy" | "emergency" | "closing",
 "why": "<at most 12 words: what the patient needs, or which step is next>",
 "alert": null | {{"level": "critical" | "serious", "text": "<at most 14 words>"}},
 "knowledge": null | {{"question": "<what the patient asked, at most 10 words>", "answer": "<the answer from general knowledge, simple English, at most 45 words>", "points": ["<at most 14 words>"], "approximate": true | false, "confirm": "<what only the clinic can confirm, at most 14 words, or empty>"}},
 "facts": [{{"title": "<2 to 5 words>", "text": "<at most 25 words>"}}],
 "form": {{"patient_type": "new" | "existing" | null, "name": null, "age": null, "centre": null, "city": null, "doctor": null, "day": null, "time": null, "concern": null}},
 "speakers": {{"<speaker number>": "agent" or "patient"}},
 "lines": {{"<line id>": "agent" or "patient"}}}}

How to fill it:
- speakers: who each speaker number is, judged from what they say: the agent speaks for the hospital, greets and asks for details; the patient asks for help. lines: only for lines whose speaker is "?".
- say: one or two short spoken sentences, at most 35 words, polite and warm, in the language the patient is using right now (Hindi in Devanagari, English, or Hinglish in English letters, as they speak). Answer the patient's latest question or need first (from the clinic facts, or else from general knowledge, see Rules); otherwise the next missing step of the call script. One question at a time. Address them by name with "ji" once known. If the agent is in the middle of saying the right thing already, say may repeat the next step.
- facts: 0 to 2, only facts from the clinic facts below that the agent needs right now (a centre's address, the doctors for a specialty, the dilation advice). Never facts that are not below.
- form: what was actually said in this call, earlier values kept unless corrected; centre as its name.
- knowledge: only when the patient's latest message asks for information the clinic facts below do not give (what a procedure such as LASIK typically costs, the kinds of lenses, recovery time, who is eligible, how a test works). Fill it from your general medical and hospital knowledge so the agent can answer with confidence. null when the clinic facts already answer it or nothing was asked.
- alert: "critical" for any emergency red flag below; "serious" if the patient is clearly upset or angry; otherwise null.

Rules (never break them):
- Never suggest a diagnosis, a likely cause, a medicine, eye drops or a dose. For a medical question: the doctor will examine and advise; offer a consultation.
- Never invent the clinic's own fee, a doctor's OPD day or time, or availability: those come only from the clinic facts below, otherwise our team will confirm it; offer a callback.
- General knowledge IS allowed for questions the clinic facts do not cover: answer in plain words (in "say" and "knowledge"). For a price, the typical range in rupees for India goes ONLY in "knowledge" (approximate: true), so the screen never shows two different figures; in "say" do not quote a price: say what it depends on and that the clinic's exact charge is confirmed by the team. Never present a general figure as the clinic's own price, never promise a result, and never diagnose this patient or name a medicine or dose for them.
- Emergency red flag: kind "emergency", alert "critical", and say is this advice: {cfg.emergency.advice}{contact}
- Upset patient: kind "empathy", say starts with a short apology.

# Call script (inbound)
{STEPS}

{clinic_facts(cfg)}"""


_WRAP_SYS = (
    "You write the after-call note for a human front-desk agent of an eye hospital, from a "
    "speech-recognition transcript of the call. Answer JSON only: "
    '{"summary": "<two or three plain sentences: who called, what they needed, what was '
    'agreed>", "outcome": one of ' + json.dumps(list(OUTCOMES)) + ', "follow_up": "<what the '
    'team should do next, at most 20 words, or empty>"}'
)


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        out = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return out if isinstance(out, dict) else None


_STR = r'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"'


def early_fields(text: str) -> dict:
    """From a JSON answer still being written: "say" and "kind" once each is complete."""
    out = {}
    for key in ("say", "kind"):
        m = re.search(_STR.format(key=key), text)
        if m:
            with contextlib.suppress(json.JSONDecodeError):
                out[key] = json.loads(f'"{m.group(1)}"')
    if out.get("kind") not in KINDS:
        out.pop("kind", None)
    return out


def clean_suggestion(out: dict) -> dict:
    """Keep only well-formed fields, bounded in size: the page shows exactly this."""

    def text(v: Any, n: int) -> str:
        return str(v).strip()[:n] if isinstance(v, (str, int, float)) else ""

    role = lambda v: v if v in ("agent", "patient") else None  # noqa: E731
    form = out.get("form") if isinstance(out.get("form"), dict) else {}
    alert = out.get("alert") if isinstance(out.get("alert"), dict) else None
    facts = [
        {"title": text(f.get("title"), 40), "text": text(f.get("text"), 220)}
        for f in (out.get("facts") or [])[:3]
        if isinstance(f, dict) and f.get("text")
    ]

    def sentences(v: Any, n: int) -> str:
        """at most n characters, cut after a full sentence rather than in the middle of a word"""
        s = text(v, 10_000)
        if len(s) <= n:
            return s
        cut = max(s.rfind(mark, 0, n) for mark in (". ", "? ", "! "))
        return s[: cut + 1] if cut > n // 3 else s[:n].rsplit(" ", 1)[0] + "..."

    kn = out.get("knowledge") if isinstance(out.get("knowledge"), dict) else None
    knowledge = (
        {
            "question": text(kn.get("question"), 100),
            "answer": sentences(kn.get("answer"), 520),
            "points": [text(p, 140) for p in (kn.get("points") or [])[:4] if text(p, 140)]
            if isinstance(kn.get("points"), list)
            else [],
            "approximate": bool(kn.get("approximate")),
            "confirm": text(kn.get("confirm"), 140),
        }
        if kn and text(kn.get("answer"), 10)
        else None
    )
    return {
        "knowledge": knowledge,
        "speakers": {str(k): role(v) for k, v in (out.get("speakers") or {}).items() if role(v)}
        if isinstance(out.get("speakers"), dict)
        else {},
        "lines": {str(k): role(v) for k, v in (out.get("lines") or {}).items() if role(v)}
        if isinstance(out.get("lines"), dict)
        else {},
        "say": text(out.get("say"), 400),
        "why": text(out.get("why"), 120),
        "kind": out.get("kind") if out.get("kind") in KINDS else "answer",
        "facts": facts,
        "form": {k: text(form.get(k), 60) or None for k in FORM_KEYS},
        "alert": {
            "level": alert.get("level")
            if alert.get("level") in ("critical", "serious")
            else "serious",
            "text": text(alert.get("text"), 120),
        }
        if alert and alert.get("text")
        else None,
    }


class AssistBrain:
    def __init__(
        self, cfg: Any = None, client: Any = None, model: str = ASSIST_MODEL, know_model: str = ""
    ) -> None:
        self.model = model
        self.know_model = know_model or ASSIST_KNOWLEDGE_MODEL or model
        self.client = client
        self.system = ""
        self.price = LLM_PRICES.get(model, (0.0, 0.0, 0.0, 0.0))
        if os.getenv("ASSIST", "1") == "0":
            return
        if self.client is None:
            if not os.getenv("ANTHROPIC_API_KEY"):
                return  # no Claude on this PC: the dashboard works, without suggestions
            try:
                import anthropic

                self.client = anthropic.AsyncAnthropic(max_retries=1, timeout=20.0)
            except Exception:
                logger.exception("agent assist off: Claude unavailable")
                return
        try:
            if cfg is None:
                from config import get_config

                cfg = get_config()
            self.system = _system(cfg)
        except Exception:
            logger.exception("agent assist off: the clinic config could not be read")

    @property
    def ready(self) -> bool:
        return bool(self.client and self.system)

    async def _ask(
        self,
        system: str,
        user: str,
        max_tokens: int,
        cache: bool,
        on_text: Any = None,
        model: str | None = None,
    ) -> tuple[str, float]:
        """on_text(text so far) after every streamed piece, when given."""
        model = model or self.model
        extra: dict = (
            {"thinking": {"type": "disabled"}}
            if "haiku" in model
            else {"output_config": {"effort": "low"}}  # Sonnet: no slow thinking first
        )
        sys_block: Any = (
            [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
            if cache
            else system
        )
        kw = dict(
            model=model,
            max_tokens=max_tokens,
            system=sys_block,
            messages=[{"role": "user", "content": user}],
            extra_body=extra,
        )
        if on_text is None:
            r = await self.client.messages.create(**kw)
        else:
            text = ""
            async with self.client.messages.stream(**kw) as st:
                async for piece in st.text_stream:
                    text += piece
                    on_text(text)
                r = await st.get_final_message()
        u = r.usage
        p = LLM_PRICES.get(model, self.price)  # per token: input, cache write, cache read, output
        usd = (
            (u.input_tokens or 0) * p[0]
            + (getattr(u, "cache_creation_input_tokens", 0) or 0) * p[1]
            + (getattr(u, "cache_read_input_tokens", 0) or 0) * p[2]
            + (u.output_tokens or 0) * p[3]
        )
        return "".join(getattr(b, "text", "") for b in r.content), usd

    async def suggest(
        self, lines: list[dict], form: dict, today: str, on_say: Any = None
    ) -> tuple[dict | None, float]:
        """lines: [{"id", "speaker", "role", "text"}] oldest first (the last 30 or so).

        Two Claude calls run at the same time: a short one for the words to say (on_say gets
        them as soon as they are written, then once more with why / kind / alert when that
        call ends) and one for the general-knowledge answer, facts and booking details,
        which finish a little later. Returns the whole suggestion and the cost."""
        if not self.ready or not lines:
            return None, 0.0
        body = "\n".join(
            f"[{ln['id']}] speaker {ln.get('speaker') or '?'}"
            f"{' (' + ln['role'] + ')' if ln.get('role') else ''}: {ln['text']}"
            for ln in lines
        )
        known = {k: v for k, v in (form or {}).items() if v}
        user = (
            f"Today: {today}.\nBooking details noted so far: {json.dumps(known, ensure_ascii=False)}\n\n"
            f"Transcript, oldest first:\n{body}\n\nThe latest line is [{lines[-1]['id']}]. "
            "What should the agent see now?"
        )
        sent: dict = {}

        def early(text: str) -> None:
            part = early_fields(text)
            if on_say and part.get("say") and part != sent:
                sent.clear()
                sent.update(part)
                on_say(dict(part))

        async def say_call() -> tuple[dict | None, float]:
            text, usd = await self._ask(
                self.system,
                user
                + '\n\nFor this request answer with ONLY the keys "say", "kind", "why", "alert".',
                320,
                cache=True,
                on_text=early if on_say else None,
            )
            part = parse_json(text)
            if part and on_say:  # the rest of the first part, before the second call is back
                done = clean_suggestion(part)
                on_say({k: done[k] for k in ("say", "kind", "why", "alert")})
            return part, usd

        async def rest_call() -> tuple[dict | None, float]:
            text, usd = await self._ask(
                self.system,
                user
                + '\n\nFor this request answer with ONLY the keys "knowledge", "facts", "form", "speakers", "lines".',
                700,
                cache=True,
                model=self.know_model,
            )
            return parse_json(text), usd

        said, rest = await asyncio.gather(say_call(), rest_call(), return_exceptions=True)
        if isinstance(said, BaseException):
            logger.warning("agent assist suggestion failed: %s", str(said)[:200])
            return None, 0.0
        if isinstance(rest, BaseException):
            logger.warning("agent assist facts and knowledge failed: %s", str(rest)[:200])
            rest = (None, 0.0)
        (first, usd1), (more, usd2) = said, rest
        if not first:
            return None, usd1 + usd2
        merged = {**(more or {}), **{k: first.get(k) for k in ("say", "kind", "why", "alert")}}
        return clean_suggestion(merged), usd1 + usd2

    async def wrap_up(self, lines: list[dict]) -> tuple[dict | None, float]:
        if not self.ready or not lines:
            return None, 0.0
        body = "\n".join(
            f"{ln.get('role') or 'speaker ' + str(ln.get('speaker'))}: {ln['text']}" for ln in lines
        )
        try:
            text, usd = await self._ask(_WRAP_SYS, f"Transcript:\n{body}", 300, cache=False)
        except Exception as e:
            logger.warning("agent assist summary failed: %s", str(e)[:200])
            return None, 0.0
        out = parse_json(text) or {}
        if not out.get("summary"):
            return None, usd
        return {
            "summary": str(out["summary"])[:600],
            "outcome": out.get("outcome") if out.get("outcome") in OUTCOMES else "Enquiry",
            "follow_up": str(out.get("follow_up") or "")[:200],
        }, usd
