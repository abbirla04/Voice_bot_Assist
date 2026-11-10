from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
os.environ.setdefault("DASHBOARD", "0")  # no live dashboard events from tests


@pytest.fixture(autouse=True)
def _isolated_logs(tmp_path, monkeypatch):
    """Every test writes logs into its own temp dir and starts with clean knobs."""
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    import notify

    monkeypatch.setattr(notify, "LOG_PATH", tmp_path / "logs" / "notifications.jsonl")
    for k in ("MOCK_SCHEDULER_LATENCY_S", "MOCK_SCHEDULER_FAIL", "NOTIFY_WEBHOOK_URL"):
        monkeypatch.delenv(k, raising=False)
    yield


SAMPLE_CLINIC = ROOT / "tests" / "fixtures" / "clinic_placeholder.yaml"  # fake data, slot mode
REAL_CLINIC = ROOT / "config" / "clinic.yaml"  # the clinic's real data, request mode


@pytest.fixture
def cfg():
    """The sample clinic (fake doctors with slot schedules): exercises slot booking."""
    from config import load_config

    return load_config(SAMPLE_CLINIC)


@pytest.fixture
def real_cfg():
    from config import load_config

    return load_config(REAL_CLINIC)


def have_keys(*names: str) -> bool:
    return all(os.getenv(n) for n in names)
