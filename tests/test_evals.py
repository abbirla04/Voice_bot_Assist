"""LLM-judged SOP/safety evals with the REAL model chain (skipped without keys).

    uv run pytest -m live -v      (excluded from the default `uv run pytest`)

For repeated runs and the scorecard use `python bench/evaluate.py`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))

from providers import DEFAULT_LLM_CHAIN, usable_items

_usable, _ = usable_items("llm", os.getenv("LLM_CHAIN", DEFAULT_LLM_CHAIN))
_judge_ok = bool(os.getenv("ANTHROPIC_API_KEY")) or all(
    os.getenv(k) for k in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
)
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not (_usable and _judge_ok), reason="needs an LLM key and a judge key"),
]

from evaluate import make_judge, run_scenario  # noqa: E402
from scenarios import SCENARIOS  # noqa: E402


@pytest.mark.parametrize("sc", SCENARIOS, ids=[s.id for s in SCENARIOS])
async def test_scenario(sc):
    r = await run_scenario(sc, llm_spec=None, judge_llm=make_judge(), run=0)
    failed = [f"{c.name} -> {c.detail}" for c in r.checks if not c.passed]
    assert r.error is None, r.error
    assert not failed, "\n".join(failed)
