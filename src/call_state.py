"""Per-call state (AgentSession userdata). Everything here lands in the audit log."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Literal

Direction = Literal["inbound", "outbound"]
Lang = Literal["en", "hi"]

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
# common romanised Hindi words callers use ("mujhe appointment chahiye")
_HINGLISH = re.compile(
    r"\b(mujhe|mera|meri|mere|aap|aapka|kya|hai|haan|nahi|nahin|chahiye|kal|parso|"
    r"aankh|ankh|dikh|dikhai|doctor\s+sahab|ji|theek|thik|kitna|kitne|kab|kahan|batao|bataiye)\b",
    re.IGNORECASE,
)


def devanagari_share(text: str) -> float:
    """Share of the words written in Devanagari. Counted by word, not letter: vowel signs
    are not letters to Python, so "मुझे appointment book करना है" scored 6 Hindi letters to
    15 English ones and a Hindi caller was answered in English."""
    words = re.findall(r"[\wऀ-ॿ]+", text)
    if not words:
        return 0.0
    return sum(bool(_DEVANAGARI.search(w)) for w in words) / len(words)


def detect_language(text: str) -> Lang | None:
    """Language of one caller turn. None = no signal (keep the current language).

    A name, a place, a number or "ok / yes / no" in Latin letters says nothing about the
    language ("Noida.", "No, no."); a real English sentence ("My age is twenty two") does.
    """
    text = re.sub(r"\[[^\]]*\]", " ", text)  # our own notes to the model are not the caller
    if not text.strip():
        return None
    if devanagari_share(text) >= 0.3:
        return "hi"
    if len(_HINGLISH.findall(text)) >= 2:
        return "hi"
    if len(re.findall(r"[A-Za-z]+", text)) < 3:
        return None
    return "en"


def mask_phone(phone: str) -> str:
    digits = re.sub(r"\D", "", phone if isinstance(phone, str) else "")
    return ("x" * max(0, len(digits) - 4) + digits[-4:]) if digits else ""


def normalise_indian_mobile(raw: str) -> str | None:
    """Return +91XXXXXXXXXX for a valid Indian mobile, else None."""
    digits = re.sub(r"\D", "", raw if isinstance(raw, str) else "")
    if digits.startswith("0091"):
        digits = digits[4:]
    elif digits.startswith("91") and len(digits) == 12:
        digits = digits[2:]
    elif digits.startswith("0") and len(digits) == 11:
        digits = digits[1:]
    if len(digits) == 10 and digits[0] in "6789":
        return "+91" + digits
    return None


@dataclass
class PatientDetails:
    patient_type: Literal["new", "existing"] | None = None
    name: str | None = None
    age: int | None = None
    phone: str | None = None
    preferred_location_id: str | None = None
    preferred_doctor: str | None = None
    concern: str | None = None
    uhid: str | None = None  # existing patients' hospital ID, if they have it
    city: str | None = None  # where the patient lives / calls from (SOP "location/city")

    # Demo Clinic SOP: always record name, age, contact number, city and eye complaint (the
    # contact number is asked for separately: the calling number is usually fine)
    ESSENTIAL = ("name", "age", "city", "concern")

    def missing_essentials(self) -> list[str]:
        return [f for f in self.ESSENTIAL if getattr(self, f) in (None, "")]

    def for_log(self) -> dict:
        d = asdict(self)
        d["phone"] = mask_phone(self.phone or "")
        return d


@dataclass
class CallState:
    direction: Direction = "inbound"
    persona: str = "Clara"
    room: str = ""
    caller_number: str = ""  # E.164 from SIP, if any
    lead_source: str | None = None
    started_at: float = field(default_factory=time.time)
    language: Lang = "en"
    bot: str = "standard"  # STT -> LLM -> TTS

    patient: PatientDetails = field(default_factory=PatientDetails)
    booking: dict | None = None
    callbacks: list[dict] = field(default_factory=list)
    transfers: list[dict] = field(default_factory=list)
    holds: list[dict] = field(default_factory=list)
    emergency_flagged: bool = False
    do_not_call: bool = False
    errors: list[str] = field(default_factory=list)

    # live flags (not about the patient)
    greeted: bool = False
    on_hold: bool = False
    awaiting_answer: bool = False
    transferred: bool = False
    ending: bool = False
    silence_prompts: int = 0
    # caller messages so far when an emergency was flagged (end_call waits for the next one)
    emergency_turn: int | None = None
    language_msg_id: str | None = None  # the caller message the language was last taken from
    chat_script: str = "latin"  # WhatsApp: "latin" (English / Hinglish) or "devanagari"

    def note_user_text(self, text: str) -> None:
        lang = detect_language(text)
        if lang is not None:
            self.language = lang

    def for_log(self) -> dict:
        return {
            "direction": self.direction,
            "persona": self.persona,
            "room": self.room,
            "caller_number": mask_phone(self.caller_number),
            "lead_source": self.lead_source,
            "started_at": self.started_at,
            "duration_s": round(time.time() - self.started_at, 1),
            "language_last_detected": self.language,
            "bot": self.bot,
            "patient": self.patient.for_log(),
            "booking": self.booking,
            "callbacks": self.callbacks,
            "transfers": self.transfers,
            "holds": self.holds,
            "emergency_flagged": self.emergency_flagged,
            "do_not_call": self.do_not_call,
            "transferred": self.transferred,
            "errors": self.errors,
        }
