"""Clinic configuration: loaded once, validated strictly, shared read-only."""

from __future__ import annotations

import os
import re
from datetime import date, datetime, time
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "clinic.yaml"


# Portable date formatting: "%-d" / "%-I" do not exist on Windows.
def fmt_day(d: date | datetime) -> str:
    return f"{d:%A} {d.day} {d:%B}"


def fmt_date(d: date | datetime) -> str:
    return f"{d:%A}, {d.day} {d:%B %Y}"


def fmt_time(t: datetime) -> str:
    hour = t.hour % 12 or 12
    return f"{hour} {t:%p}" if t.minute == 0 else f"{hour}:{t:%M %p}"


_HI_DAYS = ["सोमवार", "मंगलवार", "बुधवार", "गुरुवार", "शुक्रवार", "शनिवार", "रविवार"]
_HI_MONTHS = [
    "जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून",
    "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर",
]  # fmt: skip
_HI_HOURS = [
    "बारह",
    "एक",
    "दो",
    "तीन",
    "चार",
    "पाँच",
    "छह",
    "सात",
    "आठ",
    "नौ",
    "दस",
    "ग्यारह",
    "बारह",
]


def hi_day(d: date | datetime) -> str:
    return f"{_HI_DAYS[d.weekday()]}, {d.day} {_HI_MONTHS[d.month - 1]}"


def hi_time(t: datetime) -> str:
    """Spoken Hindi clock time: 10:30 -> "सुबह साढ़े दस बजे" (LLMs get these wrong)."""
    h, m = t.hour, t.minute
    period = "सुबह" if h < 12 else ("दोपहर" if h < 16 else ("शाम" if h < 19 else "रात"))
    h12 = h % 12 or 12
    nxt = h12 % 12 + 1
    if m == 0:
        words = f"{_HI_HOURS[h12]} बजे"
    elif m == 15:
        words = f"सवा {_HI_HOURS[h12]} बजे"
    elif m == 30:
        words = {1: "डेढ़ बजे", 2: "ढाई बजे"}.get(h12, f"साढ़े {_HI_HOURS[h12]} बजे")
    elif m == 45:
        words = f"पौने {_HI_HOURS[nxt]} बजे"
    else:
        words = f"{_HI_HOURS[h12]} बजकर {m} मिनट"
    return f"{period} {words}"


_ONES = [
    "",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _below_100(n: int) -> str:
    if n < 20:
        return _ONES[n]
    return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])


def rupees_in_words(n: int) -> str:
    """How a Delhi front desk says a fee: 1500 -> "fifteen hundred", 2000 -> "two thousand"."""
    if n <= 0 or n >= 100000:
        return str(n)
    if 1100 <= n < 10000 and n % 100 == 0 and n % 1000 != 0:
        return f"{_below_100(n // 100)} hundred"  # 1800 -> eighteen hundred
    parts = []
    if n >= 1000:
        parts.append(f"{_below_100(n // 1000)} thousand")
        n %= 1000
    if n >= 100:
        parts.append(f"{_ONES[n // 100]} hundred")
        n %= 100
    if n:
        parts.append(_below_100(n))
    return " ".join(parts)


def _parse_hhmm(value: str) -> time:
    try:
        hh, mm = value.split(":")
        return time(int(hh), int(mm))
    except Exception as e:  # pragma: no cover - message is what matters
        raise ValueError(f"expected HH:MM, got {value!r}") from e


def _check_days(days: list[str]) -> list[str]:
    bad = [d for d in days if d not in WEEKDAYS]
    if bad:
        raise ValueError(f"unknown weekday(s) {bad}; use {WEEKDAYS}")
    return days


class Consultation(BaseModel):
    duration_text: str = ""  # empty = not published: the confirmation leaves it out
    includes_registration: bool = True
    fee_covers: str = ""  # what the consultation charge covers, said with the amount


class Clinic(BaseModel):
    name: str  # as spoken ("and", not "&")
    name_hi: str = ""  # the name in Devanagari, for Hindi replies and fixed Hindi lines
    description: str = "an eye hospital"  # "<name>, <description>" in the bot's identity
    team: str = ""  # whose team the bot is part of ("Dr. Arjun Mehta's team")
    timezone: str = "Asia/Kolkata"
    closing_line: str
    booking_closing_line: str
    outbound_closing_line: str
    disclose_ai_in_greeting: bool = True
    consultation: Consultation
    dilation_advice: str
    preparation: str = ""  # what to bring / when to come, said in the confirmation
    sms_confirmation: bool = True

    @property
    def spoken_name_hi(self) -> str:
        return self.name_hi or self.name

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)


class Persona(BaseModel):
    name: str


class Personas(BaseModel):
    inbound: Persona
    outbound: Persona


class Location(BaseModel):
    id: str
    name: str
    area: str  # short form for speech and SMS ("Green Park, New Delhi")
    address: str = ""  # full postal address
    landmark: str = ""
    phones: list[str] = Field(default_factory=list)
    # None = the clinic does not publish it: the bot never states an amount
    consultation_fee_inr: int | None = Field(None, gt=0)
    # empty = not published: the bot never states days or hours, and never says "closed"
    open_days: list[str] = Field(default_factory=list)
    hours: str = ""

    @field_validator("open_days")
    @classmethod
    def _valid_days(cls, v: list[str]) -> list[str]:
        return _check_days(v)


class SlotRule(BaseModel):
    start: str
    end: str
    every_min: int = Field(gt=0, le=240)

    @model_validator(mode="after")
    def _check(self) -> SlotRule:
        if _parse_hhmm(self.start) >= _parse_hhmm(self.end):
            raise ValueError("slot start must be before end")
        return self


class Doctor(BaseModel):
    """A doctor (or optometrist) and where they see patients.

    `days` / `slots` are their OPD schedule; leave them out when it is not known (the real
    a clinic that publishes no OPD timings). Only the slot-booking mode uses them.
    """

    id: str
    name: str
    title: str = ""  # as the clinic words it: "Medical Director", "Optometrist"
    specialties: list[str] = Field(min_length=1)
    location_ids: list[str] = Field(min_length=1)
    days: list[str] = Field(default_factory=list)
    slots: SlotRule | None = None

    @model_validator(mode="before")
    @classmethod
    def _single_values(cls, data: object) -> object:
        # older configs: one `specialty` and one `location_id`
        if isinstance(data, dict):
            data = dict(data)
            if "specialty" in data and "specialties" not in data:
                data["specialties"] = [data.pop("specialty")]
            if "location_id" in data and "location_ids" not in data:
                data["location_ids"] = [data.pop("location_id")]
        return data

    @field_validator("days")
    @classmethod
    def _valid_days(cls, v: list[str]) -> list[str]:
        return _check_days(v)

    @property
    def specialty(self) -> str:
        return ", ".join(self.specialties)

    @property
    def location_id(self) -> str:
        """Where the slot-booking mode books this doctor (their first centre)."""
        return self.location_ids[0]

    @property
    def has_schedule(self) -> bool:
        return bool(self.days and self.slots)

    def sits_at(self, location_id: str) -> bool:
        return location_id in self.location_ids


class Hours(BaseModel):
    days: list[str]
    start: str
    end: str

    @field_validator("days")
    @classmethod
    def _valid_days(cls, v: list[str]) -> list[str]:
        return _check_days(v)

    def is_open(self, now: datetime) -> bool:
        if WEEKDAYS[now.weekday()] not in self.days:
            return False
        return _parse_hhmm(self.start) <= now.time() <= _parse_hhmm(self.end)


class Department(BaseModel):
    id: str
    name: str
    handles: str
    sip_call_to: str = ""
    hours: Hours | None = None  # None = not published: the bot never says it is closed

    @property
    def transfer_enabled(self) -> bool:
        return bool(self.sip_call_to.strip())


class Emergency(BaseModel):
    red_flags: list[str]
    advice: str
    contact_line: str = ""


class HoldCfg(BaseModel):
    first_notice_after_s: float = Field(2.0, ge=0)
    checkin_every_s: float = Field(20.0, gt=0)
    max_hold_s: float = Field(60.0, gt=0, le=120)
    music_volume: float = Field(0.25, ge=0, le=1)


class TransferCfg(BaseModel):
    ringing_timeout_s: float = Field(25.0, gt=0, le=90)


class Window(BaseModel):
    start: str
    end: str

    def contains(self, now: datetime) -> bool:
        return _parse_hhmm(self.start) <= now.time() <= _parse_hhmm(self.end)


class OutboundCfg(BaseModel):
    calling_window: Window
    ring_timeout_s: float = 40.0


class Limits(BaseModel):
    max_call_minutes: float = Field(15, gt=0)
    silence_prompts_before_hangup: int = Field(2, ge=0)


class BookingCfg(BaseModel):
    # "slots": book a real time slot from the scheduling backend (needs doctor schedules or
    #          the hospital's appointment system).
    # "request": take an appointment request (centre, doctor or speciality, preferred day and
    #          time); the clinic's appointments team calls back to confirm the exact time.
    mode: Literal["slots", "request"] = "slots"
    # what the bot says about charges when a centre has no consultation_fee_inr
    fee_note: str = ""
    # the short version spoken inside the appointment confirmation (default: fee_note)
    confirmation_fee_line: str = ""


class Insurance(BaseModel):
    notes: list[str] = Field(default_factory=list)
    tpas: list[str] = Field(default_factory=list)  # insurers / TPAs on the clinic's list


class ClinicConfig(BaseModel):
    placeholder_data: bool = False
    clinic: Clinic
    personas: Personas
    locations: list[Location]
    doctors: list[Doctor]
    departments: list[Department]
    emergency: Emergency
    hold: HoldCfg = HoldCfg()
    transfer: TransferCfg = TransferCfg()
    outbound: OutboundCfg
    limits: Limits = Limits()
    booking: BookingCfg = BookingCfg()
    # facts the bot may state when asked (contacts, visit, payment...), one per line
    contacts: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    insurance: Insurance = Insurance()
    sources: list[str] = Field(default_factory=list)  # where the data came from (not spoken)

    @model_validator(mode="after")
    def _cross_refs(self) -> ClinicConfig:
        loc_ids = {loc.id for loc in self.locations}
        for d in self.doctors:
            for lid in d.location_ids:
                if lid not in loc_ids:
                    raise ValueError(f"doctor {d.id} points at unknown location {lid}")
        if self.booking.mode == "slots":
            if not any(d.has_schedule for d in self.doctors):
                raise ValueError(
                    "booking.mode is 'slots' but no doctor has days and slots; "
                    "use booking.mode: request"
                )
            if any(loc.consultation_fee_inr is None for loc in self.locations):
                raise ValueError("booking.mode 'slots' needs consultation_fee_inr for every centre")
        elif any(loc.consultation_fee_inr is None for loc in self.locations) and not (
            self.booking.fee_note.strip()
        ):
            raise ValueError("a centre has no consultation_fee_inr: set booking.fee_note")
        for kind, items in (
            ("location", self.locations),
            ("doctor", self.doctors),
            ("department", self.departments),
        ):
            ids = [i.id for i in items]
            dupes = {i for i in ids if ids.count(i) > 1}
            if dupes:
                raise ValueError(f"duplicate {kind} id(s): {sorted(dupes)}")
        return self

    # -- lookups -----------------------------------------------------------
    def location(self, location_id: str) -> Location | None:
        return next((loc for loc in self.locations if loc.id == location_id), None)

    def doctor(self, doctor_id: str) -> Doctor | None:
        return next((d for d in self.doctors if d.id == doctor_id), None)

    def match_doctors(self, spoken: str) -> list[Doctor]:
        """Doctors a caller may mean: "Dr Arjun Mehta" -> one; "Dr Gupta" -> both Guptas;
        speech-to-text spellings ("Arjan Mehta") still match. Empty list = nobody."""
        key = norm_name(spoken)
        if not key:
            return []
        for d in self.doctors:
            if key in (d.id.replace("-", " "), norm_name(d.name)):
                return [d]
        tokens = key.split()
        exact = [d for d in self.doctors if all(t in norm_name(d.name).split() for t in tokens)]
        if exact:
            return exact

        def close(t: str, words: list[str]) -> bool:
            return any(w[:1] == t[:1] and SequenceMatcher(None, t, w).ratio() >= 0.8 for w in words)

        return [d for d in self.doctors if all(close(t, norm_name(d.name).split()) for t in tokens)]

    def match_specialty(self, spoken: str) -> str | None:
        """The configured speciality a caller or the model named ("retina" -> "Retina and
        Vitreous", "cataract surgery" -> "Cataract and IOL surgery"), or None."""
        names = list(dict.fromkeys(s for d in self.doctors for s in d.specialties))
        key = norm_name(spoken)
        for s in names:
            if key == norm_name(s):
                return s
        words = [_SPECIALTY_ALIASES.get(w, w) for w in key.split()]
        words = [w for w in words if w not in _GENERIC_WORDS]
        if not words:
            return None
        hits = [s for s in names if all(w in norm_name(s).split() for w in words)]
        if len(hits) != 1:
            hits = [s for s in names if any(w in norm_name(s).split() for w in words)]
        return hits[0] if len(hits) == 1 else None

    def department(self, department_id: str) -> Department | None:
        return next((d for d in self.departments if d.id == department_id), None)

    def now(self) -> datetime:
        return datetime.now(self.clinic.tz)


_GENERIC_WORDS = {"eye", "eyes", "doctor", "specialist", "surgery", "surgeon", "and", "the", "of"}
_SPECIALTY_ALIASES = {
    "pediatric": "paediatric",
    "child": "paediatric",
    "children": "paediatric",
    "uveitis": "uvea",
    "oculoplastic": "oculoplasty",
    "spectacle": "lasik",
    "specs": "lasik",
}


_TITLES = re.compile(r"^(dr|doctor|mr|mrs|ms|miss|prof)\b\.?\s*")


def norm_name(s: str) -> str:
    """Lower-case, no titles or punctuation: "Dr. Soma S. Talukdar" -> "soma s talukdar"."""
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower().replace("&", " and "))
    s = re.sub(r"\s+", " ", s).strip()
    while True:
        new = _TITLES.sub("", s)
        if new == s:
            return s.strip()
        s = new


def load_config(path: str | os.PathLike[str] | None = None) -> ClinicConfig:
    path = Path(path or os.getenv("CLINIC_CONFIG", DEFAULT_CONFIG_PATH))
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ClinicConfig.model_validate(raw)


@lru_cache(maxsize=1)
def get_config() -> ClinicConfig:
    return load_config()
