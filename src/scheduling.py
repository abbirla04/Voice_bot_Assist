"""Scheduling backend.

`SchedulingBackend` is the seam to the hospital's real HIS / appointment system.
Two booking modes (config `booking.mode`):
  slots    find_slots + book: real time slots (needs doctor schedules / the HIS)
  request  request_appointment: the caller's preferred centre, doctor and day are
           recorded and the appointments team calls back to confirm the exact time
           (no published OPD timings, so this is what the live bot uses)
`MockScheduler` implements both from config/clinic.yaml. Swap it by implementing the
same methods.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Protocol

from config import (
    WEEKDAYS,
    ClinicConfig,
    Doctor,
    _parse_hhmm,
    fmt_day,
    fmt_time,
    hi_day,
    hi_time,
    norm_name,
)


class SchedulingError(Exception):
    """Backend is down or returned something unusable. Message is caller-safe."""


class SlotTaken(SchedulingError):
    pass


@dataclass(frozen=True)
class Slot:
    slot_id: str
    doctor_id: str
    doctor_name: str
    specialty: str
    location_id: str
    location_name: str
    start: datetime

    def spoken(self) -> str:
        return f"{fmt_day(self.start)} at {fmt_time(self.start)}"

    def spoken_hi(self) -> str:
        return f"{hi_day(self.start)}, {hi_time(self.start)}"


@dataclass
class Booking:
    booking_id: str
    slot: Slot
    patient_name: str
    patient_phone: str
    fee_inr: int
    created_at: datetime = field(default_factory=datetime.now)

    def to_json(self) -> dict:
        d = asdict(self)
        d["slot"]["start"] = self.slot.start.isoformat()
        d["created_at"] = self.created_at.isoformat()
        return d


@dataclass
class AppointmentRequest:
    """An appointment the caller asked for; the clinic's team confirms the exact time."""

    request_id: str
    patient_name: str
    patient_phone: str
    location_id: str
    location_name: str
    doctor_id: str | None = None
    doctor_name: str | None = None
    specialty: str | None = None
    on_date: date | None = None
    preferred_day: str | None = None  # caller's words when there is no exact date
    preferred_time: str | None = None  # "morning", "after 4 pm"...
    patient_type: str | None = None
    age: int | None = None
    concern: str | None = None
    created_at: datetime = field(default_factory=datetime.now)

    def same_as(self, other: AppointmentRequest) -> bool:
        keys = ("patient_name", "patient_phone", "location_id", "doctor_id", "specialty", "on_date")
        return all(getattr(self, k) == getattr(other, k) for k in keys)

    def to_json(self) -> dict:
        d = asdict(self)
        d["on_date"] = self.on_date.isoformat() if self.on_date else None
        d["created_at"] = self.created_at.isoformat()
        return d


class SchedulingBackend(Protocol):
    async def find_slots(
        self,
        *,
        location_id: str | None = None,
        specialty: str | None = None,
        doctor: str | None = None,
        on_date: date | None = None,
        limit: int = 6,
    ) -> list[Slot]: ...

    async def book(self, *, slot_id: str, patient_name: str, patient_phone: str) -> Booking: ...

    async def request_appointment(self, request: AppointmentRequest) -> AppointmentRequest: ...

    def resolve_doctor(self, doctor: str) -> Doctor | None: ...


_norm = norm_name


class MockScheduler:
    """Deterministic, in-memory scheduler driven by clinic.yaml.

    Env knobs for testing the hold/fallback paths:
      MOCK_SCHEDULER_LATENCY_S  - artificial delay per call (default 0)
      MOCK_SCHEDULER_FAIL       - "1" makes every call raise SchedulingError
    """

    horizon_days = 21
    min_lead = timedelta(hours=1)

    def __init__(self, cfg: ClinicConfig, *, bookings_log: Path | None = None) -> None:
        self._cfg = cfg
        self._booked: dict[str, Booking] = {}
        self._requests: list[AppointmentRequest] = []
        # a thread lock: on Windows LiveKit runs each call as a thread with its own event loop,
        # and an asyncio.Lock can neither span loops nor stop two threads booking one slot.
        # Never held across an await.
        self._lock = threading.Lock()
        self._log = bookings_log

    # -- helpers -------------------------------------------------------------
    async def _simulate_backend(self) -> None:
        latency = float(os.getenv("MOCK_SCHEDULER_LATENCY_S", "0") or 0)
        if latency > 0:
            await asyncio.sleep(latency)
        if os.getenv("MOCK_SCHEDULER_FAIL") == "1":
            raise SchedulingError("appointment system is not responding")

    def resolve_doctor(self, doctor: str) -> Doctor | None:
        """The one doctor meant, or None (nobody, or ambiguous: "Dr Gupta")."""
        hits = self._cfg.match_doctors(doctor)
        if not hits and doctor.strip() in {d.id for d in self._cfg.doctors}:
            hits = [self._cfg.doctor(doctor.strip())]  # type: ignore[list-item]
        return hits[0] if len(hits) == 1 else None

    def _slots_for(self, d: Doctor, day: date) -> list[Slot]:
        if d.slots is None or WEEKDAYS[day.weekday()] not in d.days:
            return []
        loc = self._cfg.location(d.location_id)
        assert loc is not None
        if WEEKDAYS[day.weekday()] not in loc.open_days:
            return []
        tz = self._cfg.clinic.tz
        start = datetime.combine(day, _parse_hhmm(d.slots.start), tz)
        end = datetime.combine(day, _parse_hhmm(d.slots.end), tz)
        out = []
        t = start
        while t < end:
            sid = f"{d.id}@{t:%Y%m%dT%H%M}"
            out.append(
                Slot(
                    slot_id=sid,
                    doctor_id=d.id,
                    doctor_name=d.name,
                    specialty=d.specialty,
                    location_id=loc.id,
                    location_name=loc.name,
                    start=t,
                )
            )
            t += timedelta(minutes=d.slots.every_min)
        return out

    def _parse_slot_id(self, slot_id: str) -> Slot | None:
        try:
            doc_id, stamp = slot_id.split("@", 1)
            d = self._cfg.doctor(doc_id)
            if d is None:
                return None
            when = datetime.strptime(stamp, "%Y%m%dT%H%M").replace(tzinfo=self._cfg.clinic.tz)
        except ValueError:
            return None
        return next((s for s in self._slots_for(d, when.date()) if s.start == when), None)

    # -- API -------------------------------------------------------------------
    async def find_slots(
        self,
        *,
        location_id: str | None = None,
        specialty: str | None = None,
        doctor: str | None = None,
        on_date: date | None = None,
        limit: int = 6,
    ) -> list[Slot]:
        await self._simulate_backend()
        now = self._cfg.now()
        doctors = list(self._cfg.doctors)
        if doctor:
            d = self.resolve_doctor(doctor)
            doctors = [d] if d else []
        if location_id:
            doctors = [d for d in doctors if d.location_id == location_id]
        if specialty:
            sk = _norm(specialty)
            doctors = [d for d in doctors if sk in _norm(d.specialty)]

        days = (
            [on_date]
            if on_date
            else [now.date() + timedelta(days=i) for i in range(self.horizon_days)]
        )
        found: list[Slot] = []
        for day in days:
            for d in doctors:
                for s in self._slots_for(d, day):
                    if s.start < now + self.min_lead or s.slot_id in self._booked:
                        continue
                    found.append(s)
        found.sort(key=lambda s: (s.start, s.doctor_name))
        # spread results across doctors/days instead of six consecutive half-hours
        picked: list[Slot] = []
        seen: set[tuple[str, date]] = set()
        for s in found:
            k = (s.doctor_id, s.start.date())
            if k in seen:
                continue
            seen.add(k)
            picked.append(s)
            if len(picked) >= limit:
                break
        if on_date and len(picked) < limit:
            # on a specific day, offer more times with the same doctor
            for s in found:
                if s not in picked:
                    picked.append(s)
                if len(picked) >= limit:
                    break
            picked.sort(key=lambda s: s.start)
        return picked

    async def book(self, *, slot_id: str, patient_name: str, patient_phone: str) -> Booking:
        await self._simulate_backend()
        with self._lock:
            slot = self._parse_slot_id(slot_id)
            if slot is None:
                raise SchedulingError("that slot does not exist")
            if slot.start < self._cfg.now() + self.min_lead:
                raise SlotTaken("that time has already passed or is too soon")
            existing = self._booked.get(slot_id)
            if existing is not None:
                if (
                    existing.patient_phone == patient_phone
                    and existing.patient_name == patient_name
                ):
                    return existing  # idempotent re-booking of the same slot
                raise SlotTaken("that slot was just taken")
            loc = self._cfg.location(slot.location_id)
            assert loc is not None and loc.consultation_fee_inr is not None
            booking = Booking(
                booking_id="SEC-" + uuid.uuid4().hex[:6].upper(),
                slot=slot,
                patient_name=patient_name,
                patient_phone=patient_phone,
                fee_inr=loc.consultation_fee_inr,
            )
            self._booked[slot_id] = booking
        self._audit(booking.to_json())
        return booking

    async def request_appointment(self, request: AppointmentRequest) -> AppointmentRequest:
        await self._simulate_backend()
        with self._lock:
            for existing in self._requests:
                if existing.same_as(request):
                    return existing  # the model called twice: one request, same id
            request.request_id = "SEC-R" + uuid.uuid4().hex[:6].upper()
            self._requests.append(request)
        self._audit({"kind": "appointment_request", **request.to_json()})
        return request

    def _audit(self, row: dict) -> None:
        if self._log is None:
            return
        try:
            self._log.parent.mkdir(parents=True, exist_ok=True)
            with self._log.open("a", encoding="utf-8") as f:
                row["patient_phone"] = "x" * 6 + row["patient_phone"][-4:]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            pass  # audit log failure must never fail the booking
