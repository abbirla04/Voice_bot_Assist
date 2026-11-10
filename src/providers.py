"""Build STT / LLM / TTS from env-configured fallback chains.

Each chain is a comma-separated list, primary first:

    STT_CHAIN="livekit/deepgram/nova-3@multi, elevenlabs/scribe_v2_realtime"
    LLM_CHAIN="anthropic/claude-sonnet-5-5, anthropic/claude-haiku-5-5, livekit/openai/gpt-4.1-mini"
    TTS_CHAIN="elevenlabs/eleven_flash_v2_5, livekit/cartesia/sonic-3@hi"

Item syntax: ``<provider>/<model>[@<language>]``. ``livekit/...`` items run on
LiveKit Inference and only need the LiveKit Cloud keys. Items whose API key is
missing are skipped with a warning; if a whole chain ends up empty the agent
refuses to start and says exactly which keys are missing.

With two or more usable items the chain is wrapped in LiveKit's FallbackAdapter,
so a provider outage mid-call fails over instead of killing the call.
"""

from __future__ import annotations

import importlib
import logging
import os
from dataclasses import dataclass
from types import ModuleType

from livekit.agents import inference, llm, stt, tokenize, tts
from livekit.agents.types import NOT_GIVEN
from livekit.agents.utils import is_given

# Plugins must be imported at startup on the main thread (LiveKit registers them on
# import, and importing inside a call stalls the event loop - measured ~80 ms+ for anthropic).
# Anthropic always (TunedAnthropicLLM below subclasses it); the others only when a chain
# names them, see preload_plugins() at the end of this module. Importing all eight took
# ~17 s per process on Windows, the default chains' ones ~10 s, and every call's worker
# process imports them again.
from livekit.plugins import anthropic

logger = logging.getLogger("frontdesk.providers")

DEFAULT_STT_CHAIN = "livekit/deepgram/nova-3@multi, elevenlabs/scribe_v2_realtime"
# Gemini removed (18 Oct): slowest from India (3.6 s to first token) and failed every call
# Sonnet with effort low: 100% on the clinic evals (Haiku 94%) and a faster first word on
# plain replies (0.9 s vs 1.4 s, its default thinking cut by ANTHROPIC_EFFORT=low).
# Haiku covers a Sonnet outage or overload; GPT-4.1-mini (LiveKit Inference) an Anthropic one.
DEFAULT_LLM_CHAIN = (
    "anthropic/claude-sonnet-5-5, anthropic/claude-haiku-5-5, livekit/openai/gpt-4.1-mini"
)
DEFAULT_TTS_CHAIN = "elevenlabs/eleven_flash_v2_5, livekit/cartesia/sonic-3@hi"

# provider -> env vars it needs
REQUIRED_KEYS: dict[str, tuple[str, ...]] = {
    "livekit": ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"),
    "google": ("GOOGLE_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "xai": ("XAI_API_KEY",),
    "elevenlabs": ("ELEVEN_API_KEY",),
    "deepgram": ("DEEPGRAM_API_KEY",),
    "sarvam": ("SARVAM_API_KEY",),
    "cartesia": ("CARTESIA_API_KEY",),
}

SUPPORTED = {
    "stt": {"livekit", "deepgram", "elevenlabs", "sarvam", "xai"},
    "llm": {"livekit", "google", "anthropic", "openai", "xai"},
    "tts": {"livekit", "elevenlabs", "cartesia", "sarvam", "xai"},
}


class ProviderConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChainItem:
    provider: str
    model: str
    language: str | None = None

    @property
    def spec(self) -> str:
        return f"{self.provider}/{self.model}" + (f"@{self.language}" if self.language else "")

    def missing_keys(self) -> list[str]:
        return [k for k in REQUIRED_KEYS.get(self.provider, ()) if not os.getenv(k)]


def parse_chain(kind: str, raw: str) -> list[ChainItem]:
    items: list[ChainItem] = []
    for part in (p.strip() for p in raw.split(",")):
        if not part:
            continue
        lang = None
        if "@" in part:
            part, lang = part.rsplit("@", 1)
            lang = lang.strip() or None
        if "/" not in part:
            raise ProviderConfigError(f"{kind.upper()}_CHAIN item {part!r} must be provider/model")
        provider, model = part.split("/", 1)
        provider = provider.strip().lower()
        if provider not in SUPPORTED[kind]:
            raise ProviderConfigError(
                f"{kind.upper()}_CHAIN: provider {provider!r} not supported for {kind}; "
                f"use one of {sorted(SUPPORTED[kind])}"
            )
        items.append(ChainItem(provider, model.strip(), lang))
    if not items:
        raise ProviderConfigError(f"{kind.upper()}_CHAIN is empty")
    return items


def _plugin(provider: str) -> ModuleType:
    """The provider's LiveKit plugin (already imported by preload_plugins() in a worker)."""
    return importlib.import_module(f"livekit.plugins.{provider}")


def usable_items(kind: str, raw: str) -> tuple[list[ChainItem], dict[str, list[str]]]:
    usable, skipped = [], {}
    for item in parse_chain(kind, raw):
        missing = item.missing_keys()
        if missing:
            skipped[item.spec] = missing
        else:
            usable.append(item)
    return usable, skipped


def _chain_env(kind: str) -> str:
    default = {"stt": DEFAULT_STT_CHAIN, "llm": DEFAULT_LLM_CHAIN, "tts": DEFAULT_TTS_CHAIN}[kind]
    return os.getenv(f"{kind.upper()}_CHAIN", default)


def _select(kind: str) -> list[ChainItem]:
    usable, skipped = usable_items(kind, _chain_env(kind))
    for spec, missing in skipped.items():
        logger.warning("%s: skipping %s (missing %s)", kind.upper(), spec, ", ".join(missing))
    if not usable:
        needed = sorted({k for ks in skipped.values() for k in ks})
        raise ProviderConfigError(
            f"No usable {kind.upper()} provider. Set {kind.upper()}_CHAIN or add keys: {needed}"
        )
    logger.info("%s chain: %s", kind.upper(), " -> ".join(i.spec for i in usable))
    return usable


# -- STT -----------------------------------------------------------------------
def _make_stt(item: ChainItem, keyterms: list[str]) -> stt.STT:
    p, m, lang = item.provider, item.model, item.language
    if p == "livekit":
        return inference.STT(model=m, language=lang or "multi")
    if p == "deepgram":
        return _plugin("deepgram").STT(model=m, language=lang or "multi", keyterms=keyterms or None)
    if p == "elevenlabs":
        kw = {"model": m, "use_realtime": True}
        if lang:
            kw["language_code"] = lang
        if keyterms:
            kw["keyterms"] = keyterms
        return _plugin("elevenlabs").STT(**kw)
    if p == "sarvam":
        return _plugin("sarvam").STT(model=m, language=lang or "hi-IN", mode="codemix")
    if p == "xai":
        return _plugin("xai").STT(language=lang or "hi")
    raise AssertionError(p)


def build_stt(keyterms: list[str] | None = None) -> stt.STT:
    items = _select("stt")
    built = [_make_stt(i, keyterms or []) for i in items]
    if len(built) == 1:
        return built[0]
    return _log_switches(
        stt.FallbackAdapter(built, attempt_timeout=8.0, max_retry_per_stt=1, retry_interval=2.0),
        "stt",
        items,
    )


# -- LLM -----------------------------------------------------------------------
DEFAULT_ANTHROPIC_EFFORT = "low"  # set ANTHROPIC_EFFORT= (empty) for the model's own default


class TunedAnthropicLLM(anthropic.LLM):
    """anthropic.LLM plus request settings the plugin has no option for.

    ANTHROPIC_THINKING  thinking type sent with every request: "disabled" (Haiku),
                        "between_tools" (the "off" setting Sonnet accepts), "adaptive"
    ANTHROPIC_EFFORT    output_config.effort: low (default) | medium | high; empty = model default
    The models' default adaptive thinking is dead air before the first word on a phone line
    (measured 0.5 s on Haiku, 1.3 s on Sonnet); effort low keeps Sonnet's eval score at 100%.
    """

    def __init__(self, *, request_defaults: dict, **kw) -> None:
        super().__init__(**kw)
        self.request_defaults = request_defaults

    def chat(self, *args, extra_kwargs=NOT_GIVEN, **kw):
        merged = {**self.request_defaults, **(extra_kwargs if is_given(extra_kwargs) else {})}
        return super().chat(*args, extra_kwargs=merged or NOT_GIVEN, **kw)


def anthropic_request_defaults() -> dict:
    out: dict = {}
    thinking = os.getenv("ANTHROPIC_THINKING", "").strip().lower()
    if thinking:
        out["thinking"] = {"type": thinking}
    effort = os.getenv("ANTHROPIC_EFFORT", DEFAULT_ANTHROPIC_EFFORT).strip().lower()
    if effort:
        out["output_config"] = {"effort": effort}
    return out


def _make_llm(item: ChainItem, *, max_tokens: int = 1024) -> llm.LLM:
    """max_tokens: Anthropic only (400 truncated tool calls; the cache warm-up needs few)."""
    p, m = item.provider, item.model
    # Only send temperature when explicitly set: newer models (e.g. Claude Haiku 5.5) reject it.
    kw = (
        {"temperature": float(os.environ["LLM_TEMPERATURE"])}
        if os.getenv("LLM_TEMPERATURE")
        else {}
    )
    if p == "livekit":
        return inference.LLM(model=m)
    if p == "google":
        # Gemini rejects request deadlines under 10 s ("Manually set deadline 4s is too short"),
        # and the plugin would send our 4 s failover budget as the deadline: every call 400'd.
        # The FallbackAdapter still fails over after LLM_ATTEMPT_TIMEOUT_S on its own.
        from google.genai import types as genai_types

        return _plugin("google").LLM(
            model=m, http_options=genai_types.HttpOptions(timeout=10_000), **kw
        )
    if p == "anthropic":
        return TunedAnthropicLLM(
            request_defaults=anthropic_request_defaults(),
            model=m,
            max_tokens=max_tokens,
            # Strict tool schemas turn every optional argument into a nullable union, and
            # Anthropic refuses requests with more than 16 of those: at 18, every reply came
            # from the last backup model. The tools validate their own arguments anyway.
            _strict_tool_schema=False,
            caching="ephemeral",
            **kw,
        )
    if p == "openai":
        return _plugin("openai").LLM(model=m, **kw)
    if p == "xai":
        return _plugin("openai").LLM.with_x_ai(model=m, **kw)
    raise AssertionError(p)


def build_llm() -> llm.LLM:
    items = _select("llm")
    built = [_make_llm(i) for i in items]
    if len(built) == 1:
        return built[0]
    # attempt_timeout bounds how long we wait for a dead provider before failing over
    return _log_switches(
        llm.FallbackAdapter(
            built,
            attempt_timeout=float(os.getenv("LLM_ATTEMPT_TIMEOUT_S", "4.0")),
            max_retry_per_llm=0,
            retry_on_chunk_sent=False,
        ),
        "llm",
        items,
    )


# -- TTS -----------------------------------------------------------------------
# A front desk must sound the same calm way on every line. The voice's own ElevenLabs
# settings (stability around 0.5, speaker boost on) let it swing in emotion, and short
# lines came out sharp. Higher stability keeps the delivery even, style 0 adds no acting,
# and a touch slower reads as patient. Override any of them in .env.local, or set
# ELEVEN_VOICE_SETTINGS=voice to use the voice's own saved settings.
ELEVEN_CALM = {
    "ELEVEN_STABILITY": 0.75,
    "ELEVEN_SIMILARITY": 0.75,
    "ELEVEN_STYLE": 0.0,
    "ELEVEN_SPEED": 0.95,
    "ELEVEN_SPEAKER_BOOST": 0.0,  # 0 = off
}


def _num(name: str, default: float, lo: float, hi: float) -> float:
    raw = os.getenv(name, "").strip()
    try:
        v = float(raw) if raw else default
    except ValueError:
        logger.warning("%s=%r is not a number: using %s", name, raw, default)
        v = default
    return min(hi, max(lo, v))


def eleven_voice_settings():
    if os.getenv("ELEVEN_VOICE_SETTINGS", "").strip().lower() == "voice":
        return None
    c = ELEVEN_CALM
    return _plugin("elevenlabs").VoiceSettings(
        stability=_num("ELEVEN_STABILITY", c["ELEVEN_STABILITY"], 0.0, 1.0),
        similarity_boost=_num("ELEVEN_SIMILARITY", c["ELEVEN_SIMILARITY"], 0.0, 1.0),
        style=_num("ELEVEN_STYLE", c["ELEVEN_STYLE"], 0.0, 1.0),
        speed=_num("ELEVEN_SPEED", c["ELEVEN_SPEED"], 0.8, 1.2),
        use_speaker_boost=_num("ELEVEN_SPEAKER_BOOST", c["ELEVEN_SPEAKER_BOOST"], 0, 1) >= 0.5,
    )


def eleven_settings_tag() -> str:
    """For the greeting cache: a recording is only valid for the settings that made it."""
    s = eleven_voice_settings()
    if s is None:
        return "voice-default"
    return (
        f"st{s.stability:.2f}-si{s.similarity_boost:.2f}-sy{s.style:.2f}"
        f"-sp{s.speed:.2f}-b{int(bool(s.use_speaker_boost))}"
    )


def first_phrase_tokenizer() -> tokenize.SentenceTokenizer:
    """Hand the voice each sentence as soon as it is written. LiveKit's default holds back
    a sentence under 20 characters and merges it with the next one, and Hindi sentences
    are short in characters: "धन्यवाद, आशुतोष जी।" (19) waited for the whole next
    sentence, ~0.5 s of extra silence before the first word on most Hindi replies.
    Same words, same voice: only the first audio starts sooner."""
    n = int(_num("TTS_MIN_SENTENCE_CHARS", 10, 4, 40))
    return tokenize.blingfire.SentenceTokenizer(min_sentence_len=n)


def _make_tts(item: ChainItem) -> tts.TTS:
    p, m, lang = item.provider, item.model, item.language
    if p == "livekit":
        kw = {}
        if voice := os.getenv("LIVEKIT_TTS_VOICE"):
            kw["voice"] = voice
        if lang:
            kw["language"] = lang
        return inference.TTS(model=m, **kw)
    if p == "elevenlabs":
        kw = {"model": m}
        if voice := os.getenv("ELEVEN_VOICE_ID"):
            kw["voice_id"] = voice
        else:
            logger.warning("ELEVEN_VOICE_ID not set: using ElevenLabs default (non-Indian) voice")
        if lang:
            kw["language"] = lang
        if settings := eleven_voice_settings():
            kw["voice_settings"] = settings
        kw["word_tokenizer"] = first_phrase_tokenizer()
        return _plugin("elevenlabs").TTS(**kw)
    if p == "cartesia":
        kw = {"model": m, "language": lang or "hi"}
        if voice := os.getenv("CARTESIA_VOICE_ID"):
            kw["voice"] = voice
        return _plugin("cartesia").TTS(**kw)
    if p == "sarvam":
        return _plugin("sarvam").TTS(
            model=m, target_language_code=lang or "hi-IN", speaker=os.getenv("SARVAM_SPEAKER")
        )
    if p == "xai":
        return _plugin("xai").TTS(language=lang or "auto")
    raise AssertionError(p)


def build_tts() -> tts.TTS:
    items = _select("tts")
    built = [_make_tts(i) for i in items]
    if len(built) == 1:
        return built[0]
    return _log_switches(tts.FallbackAdapter(built, max_retry_per_tts=1), "tts", items)


def build_primary_tts() -> tts.TTS:
    """The first usable TTS in the chain: the voice callers normally hear."""
    return _make_tts(_select("tts")[0])


def primary_tts_spec() -> str:
    return _select("tts")[0].spec


def _log_switches(adapter, kind: str, items: list[ChainItem]):
    """Fallbacks are silent by default: log every switch, it explains slow or odd calls."""
    names = {}
    for inst, item in zip(getattr(adapter, f"_{kind}_instances", []), items, strict=False):
        names[id(inst)] = item.spec

    def on_change(ev) -> None:
        inst = getattr(ev, kind, None)
        name = names.get(id(inst), type(inst).__name__)
        if ev.available:
            logger.info("%s %s is available again", kind.upper(), name)
        else:
            logger.warning(
                "%s %s FAILED: switching to the next one in %s_CHAIN",
                kind.upper(),
                name,
                kind.upper(),
            )

    adapter.on(f"{kind}_availability_changed", on_change)
    return adapter


def preflight() -> dict[str, object]:
    """Report which providers will be used, without constructing clients."""
    report: dict[str, object] = {}
    for kind in ("stt", "llm", "tts"):
        usable, skipped = usable_items(kind, _chain_env(kind))
        report[kind] = {
            "usable": [i.spec for i in usable],
            "skipped": skipped,
        }
    report["livekit_keys_missing"] = [k for k in REQUIRED_KEYS["livekit"] if not os.getenv(k)]
    return report


def chain_plugins() -> list[str]:
    """Plugins the configured chains need (livekit/... items run on LiveKit Inference)."""
    needed: set[str] = set()
    for kind in ("stt", "llm", "tts"):
        try:
            items = parse_chain(kind, _chain_env(kind))
        except ProviderConfigError:
            continue  # reported properly when the chain is built
        for i in items:
            if i.provider != "livekit":
                # xAI's LLM goes through the OpenAI plugin
                needed.add("openai" if (kind, i.provider) == ("llm", "xai") else i.provider)
    return sorted(needed)


def preload_plugins() -> None:
    """Import the chains' plugins now, at startup on the main thread, never inside a call."""
    for name in chain_plugins():
        try:
            _plugin(name)
        except Exception:
            logger.exception("could not load the %s plugin", name)


preload_plugins()
