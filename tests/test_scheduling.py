from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from config import WEEKDAYS
from scheduling import MockScheduler, SchedulingError, SlotTaken

IST = ZoneInfo("Asia/Kolkata")


@pytest.fixture
def sched(cfg, monkeypatch):
    # freeze "now" at Sunday 18 Oct 2026 10:00 IST so results are deterministic
    monkeypatch.setattr(type(cfg), "now", lambda self: datetime(2026, 10, 18, 10, 0, tzinfo=IST))
    return MockScheduler(cfg)


async def test_slots_respect_doctor_days_and_location(sched, cfg):
    slots = await sched.find_slots(location_id="central-delhi", specialty="retina")
    assert slots
    for s in slots:
        assert s.doctor_id == "dr-khanna"
        assert WEEKDAYS[s.start.weekday()] in ("Tue", "Thu", "Sat")
        assert s.location_id == "central-delhi"


async def test_no_slots_in_the_past_or_on_closed_days(sched):
    slots = await sched.find_slots()
    now = datetime(2026, 10, 18, 10, 0, tzinfo=IST)
    assert all(s.start > now + timedelta(hours=1) for s in slots)
    assert all(s.start.weekday() != 6 for s in slots)  # every centre is closed Sunday


async def test_specific_date_returns_more_times(sched):
    monday = date(2026, 10, 19)
    slots = await sched.find_slots(doctor="Dr Mehta", on_date=monday)
    assert len(slots) >= 3
    assert {s.start.date() for s in slots} == {monday}


async def test_doctor_resolution(sched):
    assert sched.resolve_doctor("dr. anjali mehta").id == "dr-mehta"
    assert sched.resolve_doctor("Mehta").id == "dr-mehta"
    assert sched.resolve_doctor("Dr Who") is None
    assert sched.resolve_doctor("") is None


async def test_booking_is_idempotent_and_blocks_double_booking(sched):
    slot = (await sched.find_slots(doctor="dr-arora"))[0]
    b1 = await sched.book(slot_id=slot.slot_id, patient_name="A", patient_phone="+919999999999")
    b2 = await sched.book(slot_id=slot.slot_id, patient_name="A", patient_phone="+919999999999")
    assert b1.booking_id == b2.booking_id
    with pytest.raises(SlotTaken):
        await sched.book(slot_id=slot.slot_id, patient_name="B", patient_phone="+918888888888")
    assert slot.slot_id not in [s.slot_id for s in await sched.find_slots(doctor="dr-arora")]
    assert b1.fee_inr == 1500  # Noida fee from config


async def test_bad_slot_ids(sched):
    for bad in ("nonsense", "dr-arora@2026", "dr-ghost@20261019T1000", "dr-arora@20261019T1015"):
        with pytest.raises(SchedulingError):
            await sched.book(slot_id=bad, patient_name="A", patient_phone="+919999999999")


async def test_failure_knob(sched, monkeypatch):
    monkeypatch.setenv("MOCK_SCHEDULER_FAIL", "1")
    with pytest.raises(SchedulingError):
        await sched.find_slots()
