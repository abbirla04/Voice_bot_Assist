"""What one call cost, from the usage each provider reported during the call.

Every LLM request (including the prompt-cache warm-up and LiveKit's keyterm detection),
every TTS request and the speech-recognition audio is counted while the call runs; at the
end the call report gets a `cost` section and the agent log a "CALL COST" line.

Prices are list prices in USD (checked 18 Oct 2026). A provider not in PRICES is counted
but not priced (listed under `unpriced`), never guessed:
  Anthropic   https://platform.claude.com/docs/en/about-claude/pricing
  LiveKit     https://livekit.com/pricing/inference (Build/Ship plan rates)
  ElevenLabs  plan credits; set ELEVEN_CREDITS_PER_CHAR and ELEVEN_USD_PER_CREDIT to price it
              (Creator plan: $22 / 121,000 credits per month)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

M = 1_000_000

# per token (input, 5-minute cache write, cache read, output)
LLM_PRICES: dict[str, tuple[float, float, float, float]] = {
    "claude-sonnet-5-5": (2.00 / M, 2.50 / M, 0.10 / M, 10.00 / M),
    "claude-haiku-5-5": (0.10 / M, 0.125 / M, 0.01 / M, 0.50 / M),  # prompts up to 100k tokens
    # LiveKit Inference; it bills no cache writes, cached input at the cached rate
    "google/gemma-4-31b-it": (0.40 / M, 0.40 / M, 0.20 / M, 1.20 / M),
    "gemma-4-31b-it": (0.40 / M, 0.40 / M, 0.20 / M, 1.20 / M),
}
# per minute of audio
STT_PRICES: dict[str, float] = {
    "nova-3": 0.0058,  # Deepgram Nova-3 (multilingual) through LiveKit Inference
    "deepgram/nova-3": 0.0058,
}
# billed by these companies (who sends the invoice), for the summary
LIVEKIT_BILLED = ("gemma", "nova-3", "deepgram")


def _model_key(model: str) -> str:
    """'api.anthropic.com/claude-sonnet-5-5' -> 'claude-sonnet-5-5'."""
    m = model.split(" (")[0].strip().lower()  # "... (warm-up)" is priced like the model
    for key in LLM_PRICES:
        if m == key or m.endswith("/" + key):
            return key
    for key in STT_PRICES:
        if m == key or m.endswith("/" + key) or key in m:
            return key
    return m


@dataclass
class LLMUsage:
    model: str
    requests: int = 0
    input: int = 0  # uncached input tokens
    cache_write: int = 0
    cache_read: int = 0
    output: int = 0

    def add(self, prompt: int, cache_read: int, cache_write: int, output: int) -> None:
        self.requests += 1
        self.cache_read += cache_read
        self.cache_write += cache_write
        self.input += max(0, prompt - cache_read - cache_write)
        self.output += output

    def usd(self) -> float | None:
        p = LLM_PRICES.get(_model_key(self.model))
        if p is None:
            return None
        return (
            self.input * p[0]
            + self.cache_write * p[1]
            + self.cache_read * p[2]
            + self.output * p[3]
        )


@dataclass
class CallUsage:
    llm: dict[str, LLMUsage] = field(default_factory=dict)
    tts_chars: dict[str, int] = field(default_factory=dict)
    stt_audio_s: dict[str, float] = field(default_factory=dict)

    def add_llm(
        self, model: str, *, prompt: int, cache_read: int, cache_write: int, output: int
    ) -> None:
        self.llm.setdefault(model, LLMUsage(model)).add(prompt, cache_read, cache_write, output)

    def add_tts(self, model: str, chars: int) -> None:
        self.tts_chars[model] = self.tts_chars.get(model, 0) + int(chars)

    def add_stt(self, model: str, audio_s: float) -> None:
        self.stt_audio_s[model] = self.stt_audio_s.get(model, 0.0) + float(audio_s)

    def summary(self) -> dict:
        lines: list[dict] = []
        unpriced: list[str] = []
        for u in self.llm.values():
            usd = u.usd()
            lines.append(
                {
                    "item": f"LLM {u.model}",
                    "requests": u.requests,
                    "tokens": {
                        "input": u.input,
                        "cache_write": u.cache_write,
                        "cache_read": u.cache_read,
                        "output": u.output,
                    },
                    "usd": None if usd is None else round(usd, 7),
                }
            )
            if usd is None:
                unpriced.append(f"LLM {u.model}")
        for model, secs in self.stt_audio_s.items():
            rate = STT_PRICES.get(_model_key(model))
            usd = None if rate is None else secs / 60 * rate
            lines.append(
                {
                    "item": f"STT {model}",
                    "audio_s": round(secs, 2),
                    "usd": None if usd is None else round(usd, 7),
                }
            )
            if usd is None:
                unpriced.append(f"STT {model}")
        credits_per_char = os.getenv("ELEVEN_CREDITS_PER_CHAR")
        usd_per_credit = os.getenv("ELEVEN_USD_PER_CREDIT")
        for model, chars in self.tts_chars.items():
            line: dict = {"item": f"TTS {model}", "characters": chars, "usd": None}
            if "eleven" in model.lower() and credits_per_char and usd_per_credit:
                credits = chars * float(credits_per_char)
                line["credits"] = round(credits, 3)
                line["usd"] = round(credits * float(usd_per_credit), 7)
            else:
                unpriced.append(f"TTS {model} ({chars} characters)")
            lines.append(line)
        priced = [x["usd"] for x in lines if x["usd"] is not None]
        anthropic = sum(
            x["usd"] for x in lines if x["usd"] is not None and "claude" in x["item"].lower()
        )
        livekit = sum(
            x["usd"]
            for x in lines
            if x["usd"] is not None and any(k in x["item"].lower() for k in LIVEKIT_BILLED)
        )
        return {
            "total_usd_priced": round(sum(priced), 7),
            "anthropic_usd": round(anthropic, 7),
            "livekit_inference_usd": round(livekit, 7),
            "lines": lines,
            "unpriced": unpriced,
            "prices_checked": "2026-10-18",
        }
