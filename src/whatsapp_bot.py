"""Demo Eye and Retina Clinic front desk on WhatsApp.

The same agent as the phone bot (clinic data, scripts, appointment requests, emergency
handling, safety rules, Claude Sonnet), answering in writing:

    uv run python src/whatsapp_bot.py chat     # try it in this window: no WhatsApp needed
    uv run python src/whatsapp_bot.py serve    # webhook for the WhatsApp Cloud API

Meta  -> https://<ngrok domain>/whatsapp/webhook -> Exotel bridge (port 8765) -> here (8766)

Keys in .env.local (never in code or chat):
  WHATSAPP_TOKEN            system-user access token (or the temporary one, for a first test)
  WHATSAPP_PHONE_NUMBER_ID  "Phone number ID" from WhatsApp > API Setup
  WHATSAPP_APP_SECRET       App settings > Basic > App secret (checks Meta signed each call)
  WHATSAPP_VERIFY_TOKEN     any random text; the same text goes into Meta's webhook form
  WHATSAPP_GRAPH_VERSION    optional, default v23.0
Each chat is kept in logs/whatsapp/chats/ (one file per number) and starts fresh after
24 hours without messages (WHATSAPP_CHAT_TTL_H). Cost of every reply: logs/whatsapp/costs.jsonl
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import hmac
import json
import logging
import os
import re
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
for _stream in (sys.stdout, sys.stderr):  # Hindi on a Windows console
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env.local")
load_dotenv(ROOT / ".env")

import aiohttp  # noqa: E402
from aiohttp import web  # noqa: E402
from livekit.agents import AgentSession, llm  # noqa: E402
from livekit.agents.testing import fake_job_context  # noqa: E402

from call_cost import CallUsage  # noqa: E402
from call_state import CallState, PatientDetails, mask_phone  # noqa: E402
from config import ClinicConfig, get_config  # noqa: E402
from dashboard_feed import feed as dash_feed  # noqa: E402
from dashboard_tap import tool_event  # noqa: E402
from front_desk import FrontDeskAgent, asks_caller, chat_turn_note  # noqa: E402
from lines import chat_greeting  # noqa: E402
from providers import build_llm  # noqa: E402
from scheduling import MockScheduler  # noqa: E402
from telemetry import scrub  # noqa: E402

# a chat can wait a few seconds for Claude: no 4 s switch to the backup model as on a call
os.environ["LLM_ATTEMPT_TIMEOUT_S"] = os.getenv("WHATSAPP_LLM_TIMEOUT_S", "15")
logger = logging.getLogger("frontdesk.whatsapp")
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
CHAT_TTL_S = float(os.getenv("WHATSAPP_CHAT_TTL_H", "24")) * 3600
PORT = int(os.getenv("WHATSAPP_BOT_PORT", "8766"))
MAX_WA_TEXT = 4096  # WhatsApp's limit for one text message
# the patient has been quiet this long after the bot's reply: the chat is closed (with a polite
# message if the bot was waiting for an answer; 0 = never). Their next message starts over.
IDLE_CLOSE_S = float(os.getenv("WHATSAPP_IDLE_CLOSE_S", "120"))
IDLE_CLOSE = {
    "en": "Thank you for choosing {clinic}. I haven't heard from you for a while, so I'm "
    "closing this chat for now. Whenever you need us, just send a message and I'll be happy "
    "to help.",
    "hinglish": "{clinic} chunne ke liye dhanyavaad. Kuch der se aapki taraf se koi message "
    "nahi aaya, isliye main abhi yeh chat band kar rahi hoon. Jab bhi zarurat ho, bas ek "
    "message bhejiye, main madad ke liye yahin hoon.",
    "devanagari": "{clinic_hi} चुनने के लिए धन्यवाद। कुछ देर से आपकी ओर से कोई संदेश नहीं "
    "आया, इसलिए मैं अभी यह चैट बंद कर रही हूँ। जब भी ज़रूरत हो, बस एक संदेश भेजिए, मैं मदद के "
    "लिए यहीं हूँ।",
}


def error_reply(cfg) -> str:
    """Something broke: point the patient to the clinic's own phone line."""
    phone = next((p for loc in cfg.locations for p in loc.phones), "")
    call = f", or call {cfg.clinic.name} on {phone}" if phone else ""
    return f"Sorry, something went wrong on our side. Please try again in a minute{call}."


NOT_TEXT_REPLY = (
    "Sorry, I can read only typed messages for now. Please type your question and I will help you."
)
_HELLO = re.compile(
    r"^\s*(hi+|hello+|hey+|hii+|namaste|namaskar|good (morning|afternoon|evening))[\s!.]*$",
    re.IGNORECASE,
)


# -- conversations ------------------------------------------------------------------------
@dataclass
class Conversation:
    wa_id: str
    state: CallState
    history: llm.ChatContext
    started_at: float
    updated_at: float
    closed: bool = False  # went quiet or finished: the next message starts a new chat

    @property
    def dash(self) -> str:
        """This conversation's id on the dashboard: each restart is a session of its own."""
        return f"{dash_id(self.wa_id)}-{int(self.started_at):x}"


def _state_to_dict(state: CallState) -> dict:
    return dataclasses.asdict(state)


def _state_from_dict(d: dict) -> CallState:
    d = dict(d)
    patient = PatientDetails(**d.pop("patient", {}))
    names = {f.name for f in dataclasses.fields(CallState)}
    return CallState(**{k: v for k, v in d.items() if k in names}, patient=patient)


def dash_id(wa_id: str) -> str:
    """The chat's id on the dashboard: never the phone number itself."""
    return "wa-" + hashlib.sha256(wa_id.encode()).hexdigest()[:10]


def _keep(item: Any) -> bool:
    """Only the conversation itself is stored: the agent's instructions are rebuilt each time."""
    if item.type in ("function_call", "function_call_output"):
        return True
    return item.type == "message" and item.role in ("user", "assistant")


class ChatStore:
    """One JSON file per WhatsApp number (file named by a hash of the number)."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder / "chats"

    def _path(self, wa_id: str) -> Path:
        return self.folder / (hashlib.sha256(wa_id.encode()).hexdigest()[:20] + ".json")

    def load(self, wa_id: str) -> Conversation | None:
        path = self._path(wa_id)
        if not path.exists():
            return None
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            if time.time() - d["updated_at"] > CHAT_TTL_S:
                return None  # quiet for a day: a new chat starts (with a new greeting)
            return Conversation(
                wa_id=wa_id,
                state=_state_from_dict(d["state"]),
                history=llm.ChatContext.from_dict(d["history"]),
                started_at=d["started_at"],
                updated_at=d["updated_at"],
                closed=d.get("closed", False),
            )
        except Exception:
            logger.exception("could not read the stored chat; starting a new one")
            return None

    def save(self, conv: Conversation) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        history = llm.ChatContext([i for i in conv.history.items if _keep(i)])
        data = {
            "number": mask_phone(conv.wa_id),
            "started_at": conv.started_at,
            "updated_at": conv.updated_at,
            "closed": conv.closed,
            "state": _state_to_dict(conv.state),
            "history": history.to_dict(exclude_timestamp=False),
        }
        path = self._path(conv.wa_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str), "utf-8")
        tmp.replace(path)

    def reset(self, wa_id: str) -> None:
        path = self._path(wa_id)
        if path.exists():
            path.unlink()


# -- the front desk, in writing ---------------------------------------------------------------
class WhatsAppFrontDesk:
    def __init__(
        self,
        cfg: ClinicConfig | None = None,
        *,
        model: llm.LLM | None = None,
        store: ChatStore | None = None,
        scheduler: Any = None,
    ) -> None:
        self.cfg = cfg or get_config()
        self.llm = model or build_llm()
        self.store = store or ChatStore(LOG_DIR / "whatsapp")
        self.scheduler = scheduler or MockScheduler(
            self.cfg, bookings_log=LOG_DIR / "bookings.jsonl"
        )
        self._locks: dict[str, asyncio.Lock] = {}
        self.idle_close_s = IDLE_CLOSE_S  # quiet this long: the session is over

    def _new(self, wa_id: str) -> Conversation:
        now = time.time()
        state = CallState(
            direction="inbound",
            persona=self.cfg.personas.inbound.name,
            room=f"whatsapp-{mask_phone(wa_id)}",
            caller_number="+" + wa_id.lstrip("+"),
            bot="whatsapp",
        )
        return Conversation(wa_id, state, llm.ChatContext.empty(), now, now)

    async def reply(self, wa_id: str, text: str) -> list[str]:
        """The patient's message in; the messages to send back out (usually one)."""
        lock = self._locks.setdefault(wa_id, asyncio.Lock())
        async with lock:  # one message at a time per patient, in order
            conv = self.store.load(wa_id)
            if conv is not None and not conv.closed and self.went_quiet(conv):
                # quiet past the close time but never closed (the bot or the laptop was off
                # then): the session is over all the same
                dash_feed().send(conv.dash, "session_end", reason="no activity")
                conv.closed = True
            if conv is None or conv.closed:
                # a closed chat starts over: greeting, details and booking from the beginning
                conv = self._new(wa_id)
            fresh = not conv.history.items
            text = text.strip()[:2000]
            sid = conv.dash
            if fresh:
                dash_feed().send(
                    sid,
                    "session_start",
                    channel="whatsapp",
                    direction="inbound",
                    persona=conv.state.persona,
                    caller=mask_phone(wa_id),
                )
            dash_feed().send(sid, "user_text", text=scrub(text), language=conv.state.language)
            if fresh and _HELLO.match(text):
                # a plain "hi" gets the scripted greeting at once, no model call
                greeting = chat_greeting(
                    clinic=self.cfg.clinic.name,
                    persona=conv.state.persona,
                    disclose_ai=self.cfg.clinic.disclose_ai_in_greeting,
                    team=self.cfg.clinic.team,
                )
                conv.history.add_message(role="user", content=text)
                conv.history.add_message(role="assistant", content=greeting)
                conv.updated_at = time.time()
                self.store.save(conv)
                dash_feed().send(sid, "bot_text", text=greeting)
                return [greeting]
            note = chat_turn_note(conv.state, self.cfg, text)
            replies = await self._run(conv, f"{text}\n{note}")
            conv.updated_at = time.time()
            self.store.save(conv)
            return replies

    def went_quiet(self, conv: Conversation) -> bool:
        return 0 < self.idle_close_s < time.time() - conv.updated_at

    async def idle_close(self, wa_id: str) -> str | None:
        """The patient went quiet: the chat is closed, so their next message starts over.
        Returns the closing message to send, or None when the chat needs none: already
        closed, already said goodbye (the bot's last message asked nothing), or an emergency
        (never a cheerful goodbye after emergency advice)."""
        lock = self._locks.setdefault(wa_id, asyncio.Lock())
        async with lock:
            conv = self.store.load(wa_id)
            if conv is None or conv.closed:
                return None
            said = [m for m in conv.history.messages() if m.role in ("user", "assistant")]
            if not said or said[-1].role != "assistant":
                return None  # our reply never went out: keep the chat for their retry
            if conv.state.emergency_flagged or not asks_caller(said[-1].text_content):
                conv.closed = True  # closed quietly: no message
                self.store.save(conv)
                reason = (
                    "emergency chat went quiet" if conv.state.emergency_flagged else "chat finished"
                )
                dash_feed().send(conv.dash, "session_end", reason=reason)
                return None
            script = (
                "en"
                if conv.state.language != "hi"
                else ("devanagari" if conv.state.chat_script == "devanagari" else "hinglish")
            )
            text = IDLE_CLOSE[script].format(
                clinic=self.cfg.clinic.name, clinic_hi=self.cfg.clinic.spoken_name_hi
            )
            conv.history.add_message(role="assistant", content=text)
            conv.closed = True
            conv.updated_at = time.time()
            self.store.save(conv)
            dash_feed().send(conv.dash, "bot_text", text=text)
            dash_feed().send(conv.dash, "session_end", reason="closed for inactivity")
            logger.info(
                "chat with %s closed after %.0f s of silence", mask_phone(wa_id), IDLE_CLOSE_S
            )
            return text

    async def _run(self, conv: Conversation, user_text: str) -> list[str]:
        usage = CallUsage()

        def on_metrics(ev: Any) -> None:
            mt = ev.metrics
            if getattr(mt, "type", "") == "llm_metrics":
                meta = getattr(mt, "metadata", None)
                model = getattr(meta, "model_name", None) or getattr(mt, "label", "?")
                provider = getattr(meta, "model_provider", None) or ""
                usage.add_llm(
                    f"{provider}/{model}" if provider else model,
                    prompt=mt.prompt_tokens,
                    cache_read=mt.prompt_cached_tokens,
                    cache_write=getattr(mt, "cache_creation_tokens", 0) or 0,
                    output=mt.completion_tokens,
                )

        started = time.perf_counter()
        with fake_job_context():  # the agent runs in-process: no LiveKit room for a chat
            async with AgentSession(llm=self.llm, userdata=conv.state, max_tool_steps=5) as s:
                s.on("metrics_collected", on_metrics)
                agent = FrontDeskAgent(
                    self.cfg,
                    scheduler=self.scheduler,
                    channel="whatsapp",
                    chat_ctx=conv.history,
                    greet=False,
                )
                await s.start(agent)
                before = {i.id for i in agent.chat_ctx.items}
                await asyncio.wait_for(s.run(user_input=user_text), timeout=90)
                items = agent.chat_ctx.items
                replies = [
                    (i.text_content or "").strip()
                    for i in items
                    if i.id not in before and i.type == "message" and i.role == "assistant"
                ]
                outputs = {
                    i.call_id: i
                    for i in items
                    if i.id not in before and i.type == "function_call_output"
                }
                for i in items:
                    if i.id not in before and i.type == "function_call":
                        dash_feed().send(
                            conv.dash,
                            "tool",
                            **tool_event(i.name, i.arguments, outputs.get(i.call_id)),
                        )
                conv.history = llm.ChatContext([i for i in items if _keep(i)])
        replies = [r for r in replies if r]
        seconds = time.perf_counter() - started
        _log_cost(conv, usage.summary(), seconds)
        for r in replies:
            dash_feed().send(
                conv.dash, "bot_text", text=scrub(r), latency={"ttfa": round(seconds, 2)}
            )
        return replies


def _log_cost(conv: Conversation, cost: dict, seconds: float) -> None:
    logger.info(
        "reply for %s in %.1f s, cost $%.6f",
        mask_phone(conv.wa_id),
        seconds,
        cost["total_usd_priced"],
    )
    try:
        path = LOG_DIR / "whatsapp" / "costs.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {"at": time.time(), "number": mask_phone(conv.wa_id), "seconds": round(seconds, 2)}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({**row, **cost}, ensure_ascii=False) + "\n")
    except OSError:
        pass


# -- WhatsApp Cloud API ------------------------------------------------------------------------
DEFAULT_ABOUT = (
    f"{get_config().clinic.name} · Virtual assistant for appointments and eye-care queries"
)


class WhatsAppSender:
    def __init__(self) -> None:
        self.token = os.getenv("WHATSAPP_TOKEN", "").strip()
        self.phone_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "").strip()
        version = os.getenv("WHATSAPP_GRAPH_VERSION", "v23.0").strip()
        self.base = f"https://graph.facebook.com/{version}"
        self.url = f"{self.base}/{self.phone_id}/messages"

    @property
    def ready(self) -> bool:
        return bool(self.token and self.phone_id)

    async def _post(self, payload: dict) -> bool:
        if not self.ready:
            logger.warning("WHATSAPP_TOKEN / WHATSAPP_PHONE_NUMBER_ID not set: not sent")
            return False
        headers = {"Authorization": f"Bearer {self.token}"}
        try:
            async with (
                aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as http,
                http.post(self.url, json=payload, headers=headers) as r,
            ):
                if r.status >= 300:
                    logger.warning("WhatsApp API %s: %s", r.status, (await r.text())[:400])
                    return False
                return True
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning("WhatsApp API unreachable: %s", e)
            return False

    async def link_business_account(self) -> str:
        """Messages to the number reach a webhook only if the WhatsApp Business Account has
        subscribed this app (Meta's dashboard doesn't always do it for you). Idempotent."""
        waba = os.getenv("WHATSAPP_WABA_ID", "").strip()
        if not (self.token and waba):
            return "skipped: WHATSAPP_TOKEN or WHATSAPP_WABA_ID not set"
        base = self.base
        headers = {"Authorization": f"Bearer {self.token}"}
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as http:
                async with http.post(f"{base}/{waba}/subscribed_apps", headers=headers) as r:
                    body = await r.text()
                    if r.status >= 300:
                        return f"FAILED {r.status}: {body[:300]}"
                async with http.get(f"{base}/{waba}/subscribed_apps", headers=headers) as r:
                    apps = [
                        (a.get("whatsapp_business_api_data") or {}).get("name", "?")
                        for a in (await r.json()).get("data", [])
                    ]
            return f"OK: apps subscribed to the business account: {', '.join(apps) or 'none'}"
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
            return f"FAILED: {e}"

    async def set_profile(self, *, about: str, photo: Path | None) -> str:
        """The number's WhatsApp profile: the "About" line and the profile photo (DP).
        The photo goes up through Meta's resumable upload (app id), which returns a handle."""
        app_id = os.getenv("WHATSAPP_APP_ID", "").strip()
        if not self.ready:
            return "skipped: WHATSAPP_TOKEN or WHATSAPP_PHONE_NUMBER_ID not set"
        base = self.base
        bearer = {"Authorization": f"Bearer {self.token}"}
        body: dict = {"messaging_product": "whatsapp", "about": about[:139]}
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as http:
            if photo is not None:
                if not app_id:
                    return (
                        "FAILED: set WHATSAPP_APP_ID (App settings > Basic > App ID) for the photo"
                    )
                data = photo.read_bytes()
                kind = "image/png" if photo.suffix.lower() == ".png" else "image/jpeg"
                params = {"file_name": photo.name, "file_length": str(len(data)), "file_type": kind}
                async with http.post(
                    f"{base}/{app_id}/uploads", params=params, headers=bearer
                ) as r:
                    started = await r.json(content_type=None)
                    if r.status >= 300 or "id" not in started:
                        return f"FAILED starting the photo upload {r.status}: {str(started)[:300]}"
                headers = {"Authorization": f"OAuth {self.token}", "file_offset": "0"}
                async with http.post(f"{base}/{started['id']}", data=data, headers=headers) as r:
                    done = await r.json(content_type=None)
                    if r.status >= 300 or "h" not in done:
                        return f"FAILED uploading the photo {r.status}: {str(done)[:300]}"
                body["profile_picture_handle"] = done["h"]
            url = f"{base}/{self.phone_id}/whatsapp_business_profile"
            async with http.post(url, json=body, headers=bearer) as r:
                if r.status >= 300:
                    return f"FAILED saving the profile {r.status}: {(await r.text())[:300]}"
            async with http.get(
                url, params={"fields": "about,profile_picture_url"}, headers=bearer
            ) as r:
                now = ((await r.json(content_type=None)).get("data") or [{}])[0]
        has_photo = "yes" if now.get("profile_picture_url") else "not yet"
        return f"OK: About = {now.get('about')!r}; profile photo set: {has_photo}"

    async def send_text(self, to: str, body: str) -> bool:
        ok = True
        for part in _split(body):
            ok = (
                await self._post(
                    {
                        "messaging_product": "whatsapp",
                        "recipient_type": "individual",
                        "to": to,
                        "type": "text",
                        "text": {"preview_url": False, "body": part},
                    }
                )
                and ok
            )
        return ok

    async def read_and_typing(self, message_id: str) -> None:
        """Blue ticks + "typing..." while Claude writes the reply."""
        await self._post(
            {
                "messaging_product": "whatsapp",
                "status": "read",
                "message_id": message_id,
                "typing_indicator": {"type": "text"},
            }
        )


def _split(body: str) -> list[str]:
    parts, rest = [], body
    while len(rest) > MAX_WA_TEXT:
        cut = rest.rfind("\n", 0, MAX_WA_TEXT)
        cut = cut if cut > 0 else MAX_WA_TEXT
        parts.append(rest[:cut])
        rest = rest[cut:].lstrip()
    return [*parts, rest] if rest else parts


def message_text(msg: dict) -> str | None:
    """Text of an incoming WhatsApp message, or None for media we do not read yet."""
    kind = msg.get("type")
    if kind == "text":
        return (msg.get("text") or {}).get("body")
    if kind == "interactive":
        ir = msg.get("interactive") or {}
        return (ir.get("button_reply") or ir.get("list_reply") or {}).get("title")
    if kind == "button":
        return (msg.get("button") or {}).get("text")
    return None


def signature_ok(raw: bytes, header: str, secret: str) -> bool:
    expected = "sha256=" + hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
    return hmac.compare_digest(header or "", expected)


class Webhook:
    def __init__(self, desk: WhatsAppFrontDesk, sender: WhatsAppSender | None = None) -> None:
        self.desk = desk
        self.sender = sender or WhatsAppSender()
        self.verify_token = os.getenv("WHATSAPP_VERIFY_TOKEN", "").strip()
        self.app_secret = os.getenv("WHATSAPP_APP_SECRET", "").strip()
        self._seen: OrderedDict[str, None] = OrderedDict()  # Meta can deliver a message twice
        self._tasks: set[asyncio.Task] = set()
        self._idle: dict[str, asyncio.Task] = {}  # per patient: the inactivity close timer

    @property
    def idle_close_s(self) -> float:  # one setting, shared with the desk
        return self.desk.idle_close_s

    @idle_close_s.setter
    def idle_close_s(self, seconds: float) -> None:
        self.desk.idle_close_s = seconds

    async def verify(self, request: web.Request) -> web.Response:
        q = request.query
        if (
            q.get("hub.mode") == "subscribe"
            and self.verify_token
            and hmac.compare_digest(q.get("hub.verify_token", ""), self.verify_token)
        ):
            logger.info("webhook verified by Meta")
            return web.Response(text=q.get("hub.challenge", ""))
        return web.Response(status=403, text="verification failed")

    async def receive(self, request: web.Request) -> web.Response:
        raw = await request.read()
        if self.app_secret:
            if not signature_ok(
                raw, request.headers.get("X-Hub-Signature-256", ""), self.app_secret
            ):
                logger.warning("rejected a webhook call with a bad signature")
                return web.Response(status=401, text="bad signature")
        else:
            logger.warning("WHATSAPP_APP_SECRET not set: webhook signatures are not checked")
        try:
            data = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return web.Response(status=400, text="bad json")
        for entry in data.get("entry") or []:
            for change in entry.get("changes") or []:
                value = change.get("value") or {}
                for msg in value.get("messages") or []:
                    mid = msg.get("id", "")
                    if mid in self._seen:
                        continue
                    self._seen[mid] = None
                    while len(self._seen) > 2000:
                        self._seen.popitem(last=False)
                    task = asyncio.create_task(self._handle(msg))
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)
        return web.Response(text="ok")  # answer Meta at once; the reply goes out separately

    def _stop_idle(self, wa_id: str) -> None:
        task = self._idle.pop(wa_id, None)
        if task and not task.done():
            task.cancel()

    def _start_idle(self, wa_id: str) -> None:
        self._stop_idle(wa_id)
        if self.idle_close_s > 0:
            self._idle[wa_id] = asyncio.create_task(self._idle_watch(wa_id))

    async def _idle_watch(self, wa_id: str) -> None:
        try:
            await asyncio.sleep(self.idle_close_s)
            text = await self.desk.idle_close(wa_id)
            if text:
                await self.sender.send_text(wa_id, text)
        except asyncio.CancelledError:
            pass  # the patient wrote again
        except Exception:
            logger.exception("could not close the idle chat with %s", mask_phone(wa_id))
        finally:
            if self._idle.get(wa_id) is asyncio.current_task():
                self._idle.pop(wa_id, None)

    async def _handle(self, msg: dict) -> None:
        wa_id = msg.get("from", "")
        text = message_text(msg)
        self._stop_idle(wa_id)  # they are here: no closing message while we answer
        try:
            if text is None:
                await self.sender.send_text(wa_id, NOT_TEXT_REPLY)
                self._start_idle(wa_id)  # a photo or voice note still restarts the 120 s clock
                return
            await self.sender.read_and_typing(msg.get("id", ""))
            for reply in await self.desk.reply(wa_id, text):
                await self.sender.send_text(wa_id, reply)
            self._start_idle(wa_id)
        except Exception:
            logger.exception("could not answer a WhatsApp message from %s", mask_phone(wa_id))
            await self.sender.send_text(wa_id, error_reply(self.desk.cfg))

    async def health(self, _: web.Request) -> web.Response:
        return web.json_response(
            {"ok": True, "sending": self.sender.ready, "signature_check": bool(self.app_secret)}
        )


HOOK = web.AppKey("hook", Webhook)


def make_app(desk: WhatsAppFrontDesk | None = None, sender: WhatsAppSender | None = None):
    hook = Webhook(desk or WhatsAppFrontDesk(), sender)
    app = web.Application()
    app.router.add_get("/whatsapp/webhook", hook.verify)
    app.router.add_post("/whatsapp/webhook", hook.receive)
    app.router.add_get("/whatsapp/health", hook.health)
    app[HOOK] = hook
    return app


# -- commands ---------------------------------------------------------------------------------
async def console() -> None:
    """A WhatsApp chat in this window, as the number WHATSAPP_TEST_NUMBER."""
    wa_id = os.getenv("WHATSAPP_TEST_NUMBER", "919000000001")
    desk = WhatsAppFrontDesk()
    print(
        f"{get_config().clinic.name} on WhatsApp - test chat. Type a message; /new starts a new chat, /quit exits."
    )
    while True:
        try:
            text = (await asyncio.to_thread(input, "\nYou: ")).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text in ("/quit", "/exit"):
            break
        if text == "/new":
            desk.store.reset(wa_id)
            print("(new chat)")
            continue
        for reply in await desk.reply(wa_id, text):
            print(f"\nClara: {reply}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("livekit").setLevel(logging.WARNING)
    import log_masking

    log_masking.install()  # no verify token in the access log
    cmd = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if cmd == "chat":
        logging.getLogger().setLevel(logging.WARNING)
        asyncio.run(console())
        return
    if cmd == "profile":
        # whatsapp_bot.py profile [photo.jpg] ["About text"]
        photo = Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "-" else None
        about = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_ABOUT
        if photo is not None and not photo.is_file():
            print(f"No such picture: {photo}")
            return
        print("Setting the WhatsApp profile (About line and photo)...")
        print(asyncio.run(WhatsAppSender().set_profile(about=about, photo=photo)))
        return
    if cmd == "link":
        print("Linking the WhatsApp Business Account to this app...")
        print(asyncio.run(WhatsAppSender().link_business_account()))
        return
    if cmd == "serve":
        sender = WhatsAppSender()
        logger.info(
            "WhatsApp bot on 127.0.0.1:%d (sending: %s, signature check: %s)",
            PORT,
            "ready"
            if sender.ready
            else "NOT SET - add WHATSAPP_TOKEN and WHATSAPP_PHONE_NUMBER_ID",
            "on" if os.getenv("WHATSAPP_APP_SECRET") else "OFF - add WHATSAPP_APP_SECRET",
        )
        app = make_app(sender=sender)

        async def _link(_: web.Application) -> None:
            logger.info("WhatsApp business account link: %s", await sender.link_business_account())

        app.on_startup.append(_link)
        web.run_app(app, host="127.0.0.1", port=PORT, print=None)
        return
    raise SystemExit(f"unknown command {cmd!r}: use chat or serve")


if __name__ == "__main__":
    main()
