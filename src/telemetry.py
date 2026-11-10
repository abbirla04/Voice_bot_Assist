"""Per-call latency tracking + the audit report written when a call ends.

Every call leaves logs/calls/<room>.json with: call state (masked), full
transcript with tool calls, per-turn latency, provider errors, and fallbacks.
bench/summarize.py turns a folder of these into the benchmark scorecard.
"""

from __future__ import annotations

import json
import logging
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any

from livekit.agents import (
    AgentFalseInterruptionEvent,
    AgentSession,
    ConversationItemAddedEvent,
    ErrorEvent,
    MetricsCollectedEvent,
)

from call_cost import CallUsage
from call_state import CallState

logger = logging.getLogger("frontdesk.telemetry")
latency_log = logging.getLogger("frontdesk.latency")  # one line per measurement, grep-able
TTFA_TARGET_S = 0.9

TURN_FIELDS = {
    # user message metrics
    "end_of_turn_delay": "user",
    "transcription_delay": "user",
    # assistant message metrics
    "llm_node_ttft": "assistant",
    "tts_node_ttfb": "assistant",
    "e2e_latency": "assistant",
}


_DIGIT_RUN = re.compile(r"(?<!\d)(?:\+?\d[\s-]?){10,13}(?!\d)")


def scrub(obj: Any) -> Any:
    """Mask phone-number-like digit runs anywhere in the transcript (text and tool args)."""
    if isinstance(obj, str):
        return _DIGIT_RUN.sub(lambda m: "x" * 6 + re.sub(r"\D", "", m.group())[-4:], obj)
    if isinstance(obj, list):
        return [scrub(x) for x in obj]
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items()}
    return obj


def _ms(v: float | None) -> str:
    return "-" if v is None else f"{v * 1000:.0f} ms"


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    vs = sorted(values)
    k = (len(vs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(vs) - 1)
    return vs[lo] + (vs[hi] - vs[lo]) * (k - lo)


def summarize(samples: dict[str, list[float]]) -> dict[str, dict[str, float | int | None]]:
    out = {}
    for k, vs in samples.items():
        out[k] = {
            "n": len(vs),
            "p50": pct(vs, 0.5),
            "p95": pct(vs, 0.95),
            "mean": statistics.fmean(vs) if vs else None,
        }
    return out


class CallTelemetry:
    def __init__(self, session: AgentSession[CallState]) -> None:
        self.session = session
        self.samples: dict[str, list[float]] = {k: [] for k in TURN_FIELDS}
        self.samples["stt_audio_to_final_s"] = []
        self.provider_errors: list[dict[str, Any]] = []
        # interruptions = bot replies cut off by the caller; false = resumed after a cough/"hmm"
        self.interruptions = {"interruptions": 0, "false_interruptions": 0, "false_resumed": 0}
        self._started = time.time()
        self._turn = 0
        self._last_user: dict[str, float] = {}
        self.usage = CallUsage()  # what the call used, for its exact cost (call_cost.py)
        session.on("conversation_item_added", self._on_item)
        session.on("metrics_collected", self._on_metrics)
        session.on("agent_false_interruption", self._on_false_interruption)
        session.on("error", self._on_error)

    def _on_item(self, ev: ConversationItemAddedEvent) -> None:
        item = ev.item
        metrics = getattr(item, "metrics", None) or {}
        role = getattr(item, "role", None)
        for key, want_role in TURN_FIELDS.items():
            if role == want_role and key in metrics and metrics[key] is not None:
                self.samples[key].append(float(metrics[key]))
        if role == "assistant" and getattr(item, "interrupted", False):
            self.interruptions["interruptions"] += 1
        if role == "user" and getattr(item, "text_content", None):
            self.session.userdata.note_user_text(item.text_content)
        if role == "user":
            self._last_user = dict(metrics)
        elif role == "assistant":
            self._log_turn(item, metrics)

    def _log_turn(self, item: Any, m: dict) -> None:
        """One line per bot reply: TTFA and where it went."""
        self._turn += 1
        cut = " (interrupted)" if getattr(item, "interrupted", False) else ""
        text = (getattr(item, "text_content", "") or "")[:60]
        e2e = m.get("e2e_latency")
        if e2e is None:  # greeting or scripted line: no caller turn before it
            latency_log.info(
                "reply %d (scripted): TTS first audio %s%s | %r",
                self._turn,
                _ms(m.get("tts_node_ttfb")),
                cut,
                text,
            )
            return
        u = self._last_user
        latency_log.info(
            "reply %d: TTFA %s %s = end-of-turn %s + LLM first token %s + TTS first audio %s "
            "(transcript ready %s after speech end)%s | %r",
            self._turn,
            _ms(e2e),
            "OK" if e2e <= TTFA_TARGET_S else f"SLOW (target {_ms(TTFA_TARGET_S)})",
            _ms(u.get("end_of_turn_delay")),
            _ms(m.get("llm_node_ttft")),
            _ms(m.get("tts_node_ttfb")),
            _ms(u.get("transcription_delay")),
            cut,
            text,
        )

    def _on_metrics(self, ev: MetricsCollectedEvent) -> None:
        mt = ev.metrics
        kind = getattr(mt, "type", "")
        meta = getattr(mt, "metadata", None)
        model = getattr(meta, "model_name", None) or getattr(mt, "label", "?")
        provider = getattr(meta, "model_provider", None) or ""
        who = f"{provider}/{model}" if provider else model
        if kind == "tts_metrics":
            self.usage.add_tts(who, mt.characters_count)
            latency_log.info(
                "TTS %s: first audio %s, %d chars, connection %s (acquire %s)%s",
                who,
                _ms(mt.ttfb),
                mt.characters_count,
                "reused" if mt.connection_reused else "NEW",
                _ms(mt.acquire_time),
                " cancelled" if mt.cancelled else "",
            )
        elif kind == "llm_metrics":
            written = getattr(mt, "cache_creation_tokens", 0) or 0
            self.usage.add_llm(
                who,
                prompt=mt.prompt_tokens,
                cache_read=mt.prompt_cached_tokens,
                cache_write=written,
                output=mt.completion_tokens,
            )
            latency_log.info(
                "LLM %s: first token %s, total %s, prompt %d tok (%d cached, %d written), "
                "out %d tok%s",
                who,
                _ms(mt.ttft),
                _ms(mt.duration),
                mt.prompt_tokens,
                mt.prompt_cached_tokens,
                written,
                mt.completion_tokens,
                " cancelled" if mt.cancelled else "",
            )
        elif kind == "realtime_model_metrics":
            latency_log.info(
                "LIVE %s: first audio/text %s, total %s, in %d tok, out %d tok%s",
                who,
                _ms(mt.ttft if mt.ttft and mt.ttft > 0 else None),
                _ms(mt.duration),
                mt.input_tokens,
                mt.output_tokens,
                " cancelled" if mt.cancelled else "",
            )
        elif kind == "eou_metrics":
            latency_log.info(
                "end of turn: decided %s after speech end, transcript %s, hook %s",
                _ms(mt.end_of_utterance_delay),
                _ms(mt.transcription_delay),
                _ms(mt.on_user_turn_completed_delay),
            )
        elif kind == "stt_metrics":
            self.usage.add_stt(who, getattr(mt, "audio_duration", 0.0) or 0.0)
            if not getattr(mt, "streamed", True):
                latency_log.info(
                    "STT %s: %s for %s of audio", who, _ms(mt.duration), _ms(mt.audio_duration)
                )

    def add_warmup_usage(self, model: str, usage: Any) -> None:
        """The prompt-cache warm-up runs on its own client (cache_warm.py): count it too."""
        self.usage.add_llm(
            f"{model} (warm-up)",
            prompt=usage.prompt_tokens,
            cache_read=usage.cache_read_tokens or usage.prompt_cached_tokens,
            cache_write=usage.cache_creation_tokens,
            output=usage.completion_tokens,
        )

    def _on_false_interruption(self, ev: AgentFalseInterruptionEvent) -> None:
        self.interruptions["false_interruptions"] += 1
        if ev.resumed:
            self.interruptions["false_resumed"] += 1

    def _on_error(self, ev: ErrorEvent) -> None:
        err = ev.error
        self.provider_errors.append(
            {
                "at": time.time(),
                "source": type(ev.source).__name__,
                "error": str(getattr(err, "error", err))[:500],
                "recoverable": getattr(err, "recoverable", None),
            }
        )
        logger.warning("pipeline error from %s: %s", type(ev.source).__name__, err)

    def cost(self) -> dict[str, Any]:
        return self.usage.summary()

    def log_cost(self) -> dict[str, Any]:
        c = self.cost()
        parts = [f"{x['item']} ${x['usd']:.6f}" for x in c["lines"] if x.get("usd") is not None]
        logger.info(
            "CALL COST: $%.6f priced (Anthropic $%.6f, LiveKit Inference $%.6f) | %s%s",
            c["total_usd_priced"],
            c["anthropic_usd"],
            c["livekit_inference_usd"],
            "; ".join(parts),
            f" | not priced: {', '.join(c['unpriced'])}" if c["unpriced"] else "",
        )
        return c

    def report(self) -> dict[str, Any]:
        history = []
        try:
            history = scrub(self.session.history.to_dict().get("items", []))
        except Exception:
            logger.exception("could not serialise history")
        return {
            "version": 1,
            "ended_at": time.time(),
            "call": self.session.userdata.for_log(),
            "cost": self.cost(),
            "latency": summarize(self.samples),
            "latency_samples": self.samples,
            "interruptions": self.interruptions,
            "provider_errors": self.provider_errors,
            "transcript": history,
        }

    def write(self, room_name: str) -> Path | None:
        out_dir = Path(os.getenv("LOG_DIR", "logs")) / "calls"
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{room_name or 'call'}-{int(self._started)}.json"
            path.write_text(
                json.dumps(self.report(), ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            logger.info("call report written to %s", path)
            return path
        except Exception:
            logger.exception("could not write call report")
            return None
