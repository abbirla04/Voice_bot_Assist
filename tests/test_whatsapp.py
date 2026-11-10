"""The WhatsApp front desk: same agent as the phone bot, in writing. Offline (scripted LLM)."""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from aiohttp.test_utils import TestClient, TestServer
from fake_llm import ScriptedLLM

import front_desk
import whatsapp_bot as wb
from call_state import CallState
from front_desk import FrontDeskAgent, chat_turn_note
from prompts import build_instructions, for_chat
from scheduling import MockScheduler

IST = ZoneInfo("Asia/Kolkata")
MONDAY_11 = datetime(2026, 10, 19, 11, 0, tzinfo=IST)
WA = "919810012345"


@pytest.fixture
def real(real_cfg, monkeypatch):
    monkeypatch.setattr(type(real_cfg), "now", lambda self: MONDAY_11)

    async def sent(*a, **k):
        return True

    monkeypatch.setattr(front_desk, "notify", sent)
    return real_cfg


def desk(cfg, steps, tmp_path):
    fake = ScriptedLLM(steps)
    d = wb.WhatsAppFrontDesk(
        cfg, model=fake, store=wb.ChatStore(tmp_path / "wa"), scheduler=MockScheduler(cfg)
    )
    return d, fake


GREETING = (
    "Namaste, welcome to Demo Eye and Retina Clinic. I'm Clara, a virtual assistant "
    "from Dr. Arjun Mehta's team. How may I help you today?"
)


# -- prompt and agent --------------------------------------------------------------------
def test_chat_prompt_is_written_for_whatsapp_and_phone_prompt_is_not(real):
    chat = build_instructions(real, direction="inbound", channel="whatsapp")
    phone = build_instructions(real, direction="inbound")
    for must in (
        "chatting with patients on WhatsApp",
        "Hinglish",
        "98765 43210",  # written normally in a chat
        "You were chatting with Clara",
        "WhatsApp number they are writing from",
        "There is no live transfer in a chat",
    ):
        assert must in chat, must
    assert "end_call" not in chat and "8 1 7 8 5" not in chat
    assert "You are on a phone call" in phone and "9 8 7 6 5, 4 3 2 1 0" in phone


def test_for_chat_writes_numbers_normally():
    assert for_chat("call us on 9 8 7 6 5, 4 3 2 1 0.") == "call us on 98765 43210."


def test_chat_agent_has_no_end_call_and_keeps_brackets(real):
    agent = FrontDeskAgent(real, scheduler=MockScheduler(real), channel="whatsapp", greet=False)
    assert "end_call" not in {t.id for t in agent.tools}
    assert "request_appointment" in {t.id for t in agent.tools}
    f = front_desk.SpokenTextFilter("[", "]")
    assert (
        f.feed("Green Park Main (Green Park metro) [note]") == "Green Park Main (Green Park metro) "
    )


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("haan isi number pe call karna", "Hinglish"),
        ("मुझे कल अपॉइंटमेंट चाहिए", "Devanagari"),
        ("I want to book for tomorrow", "English"),
    ],
)
def test_chat_reply_language_follows_the_script_typed(real, text, label):
    assert label in chat_turn_note(CallState(), real, text)


# -- the conversation -------------------------------------------------------------------
async def test_hello_gets_the_greeting_without_a_model_call(real, tmp_path):
    d, fake = desk(real, [], tmp_path)
    replies = await d.reply(WA, "Hi")
    assert replies == [GREETING]
    assert fake.calls == []


async def test_chat_remembers_the_conversation_and_registers_the_request(real, tmp_path):
    steps = [
        (
            "say_call",
            "Thank you, Ramesh ji. Which city are you coming from?",
            "update_patient_details",
            {"name": "Ramesh Kumar", "age": 62, "concern": "eye check-up"},
        ),
        (
            "call",
            "request_appointment",
            {
                "preferred_day": "tomorrow",
                "preferred_time": "morning",
                "date_iso": "2026-10-20",
                "use_calling_number": True,
            },
        ),
    ]
    d, fake = desk(real, steps, tmp_path)
    first = await d.reply(WA, "New patient, Ramesh Kumar, 62, eye check-up")
    assert first == ["Thank you, Ramesh ji. Which city are you coming from?"]
    # a new message, a new session: the earlier messages and the patient's details carry over
    second = await d.reply(WA, "From Saket. Tomorrow morning, use this number")
    assert len(second) == 1
    for must in (
        "appointment request for Ramesh Kumar",
        "with Dr. Arjun Mehta at our Green Park clinic",
        "Tuesday 20 October, morning",
        "number ending 2345",  # written normally, not digit by digit
    ):
        assert must in second[0], must
    seen = " ".join(m.text_content or "" for m in fake.calls[-1].messages())
    assert "Which city are you coming from?" in seen  # the model saw the earlier chat
    conv = d.store.load(WA)
    assert conv.state.booking["status"] == "requested" and conv.state.patient.age == 62
    assert conv.state.caller_number == "+" + WA


async def test_chat_starts_fresh_after_a_quiet_day(real, tmp_path, monkeypatch):
    d, _ = desk(real, [], tmp_path)
    await d.reply(WA, "hello")
    assert d.store.load(WA) is not None
    monkeypatch.setattr(wb, "CHAT_TTL_S", -1)
    assert d.store.load(WA) is None


# -- the webhook --------------------------------------------------------------------------
class FakeSender:
    ready = True

    def __init__(self):
        self.sent: list[tuple[str, str]] = []
        self.typing: list[str] = []

    async def send_text(self, to, body):
        self.sent.append((to, body))
        return True

    async def read_and_typing(self, message_id):
        self.typing.append(message_id)


def _payload(text: str | None, mid: str = "wamid.1", kind: str = "text") -> bytes:
    msg = {"from": WA, "id": mid, "type": kind}
    if kind == "text":
        msg["text"] = {"body": text}
    return json.dumps(
        {"entry": [{"changes": [{"value": {"messages": [msg]}, "field": "messages"}]}]}
    ).encode()


def _sign(raw: bytes, secret: str = "s3cret") -> str:
    return "sha256=" + hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()


async def test_webhook_verification(real, tmp_path, monkeypatch):
    monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", "my-verify")
    d, _ = desk(real, [], tmp_path)
    async with TestClient(TestServer(wb.make_app(d, FakeSender()))) as c:
        ok = await c.get(
            "/whatsapp/webhook",
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": "my-verify",
                "hub.challenge": "42",
            },
        )
        assert ok.status == 200 and await ok.text() == "42"
        bad = await c.get(
            "/whatsapp/webhook",
            params={"hub.mode": "subscribe", "hub.verify_token": "wrong", "hub.challenge": "42"},
        )
        assert bad.status == 403


async def test_webhook_answers_signed_messages_once(real, tmp_path, monkeypatch):
    import asyncio

    monkeypatch.setenv("WHATSAPP_APP_SECRET", "s3cret")
    d, _ = desk(real, [], tmp_path)
    sender = FakeSender()
    app = wb.make_app(d, sender)
    async with TestClient(TestServer(app)) as c:
        raw = _payload("Hi")
        forged = await c.post(
            "/whatsapp/webhook", data=raw, headers={"X-Hub-Signature-256": "sha256=0"}
        )
        assert forged.status == 401
        for _ in range(2):  # Meta may deliver the same message twice
            r = await c.post(
                "/whatsapp/webhook",
                data=raw,
                headers={"X-Hub-Signature-256": _sign(raw), "Content-Type": "application/json"},
            )
            assert r.status == 200
        await asyncio.gather(*list(app[wb.HOOK]._tasks))
    assert sender.typing == ["wamid.1"]
    assert len(sender.sent) == 1 and sender.sent[0][1] == GREETING


async def test_webhook_asks_for_text_when_a_voice_note_arrives(real, tmp_path, monkeypatch):
    import asyncio

    monkeypatch.delenv("WHATSAPP_APP_SECRET", raising=False)
    d, _ = desk(real, [], tmp_path)
    sender = FakeSender()
    app = wb.make_app(d, sender)
    async with TestClient(TestServer(app)) as c:
        r = await c.post("/whatsapp/webhook", data=_payload(None, "wamid.2", "audio"))
        assert r.status == 200
        await asyncio.gather(*list(app[wb.HOOK]._tasks))
        assert WA in app[wb.HOOK]._idle  # the inactivity clock still runs
    assert sender.sent == [(WA, wb.NOT_TEXT_REPLY)]


class FakeGraph:
    """Stands in for aiohttp.ClientSession against Meta's Graph API: records each request and
    answers from `reply(method, url) -> (status, body)`."""

    class Resp:
        def __init__(self, status, body):
            self.status, self._body = status, body

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def json(self, content_type=None):
            return self._body

        async def text(self):
            return json.dumps(self._body)

    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def __call__(self, *a, **k):  # used as aiohttp.ClientSession(timeout=...)
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def post(self, url, **k):
        self.calls.append(("POST", url, k))
        return self.Resp(*self.reply("POST", url))

    def get(self, url, **k):
        self.calls.append(("GET", url, k))
        return self.Resp(*self.reply("GET", url))


async def test_business_account_is_linked_to_the_app(monkeypatch):
    """Without /{waba}/subscribed_apps, Meta never sends the number's messages to the webhook."""
    graph = FakeGraph(
        lambda method, url: (
            (200, {"success": True})
            if method == "POST"
            else (200, {"data": [{"whatsapp_business_api_data": {"name": "Demo Clinic Bot"}}]})
        )
    )
    monkeypatch.setattr(wb.aiohttp, "ClientSession", graph)
    monkeypatch.setenv("WHATSAPP_TOKEN", "tok")
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "123")
    monkeypatch.setenv("WHATSAPP_WABA_ID", "999")
    out = await wb.WhatsAppSender().link_business_account()
    assert out.startswith("OK") and "Demo Clinic Bot" in out
    method, url, kw = graph.calls[0]
    assert (method, url) == ("POST", "https://graph.facebook.com/v23.0/999/subscribed_apps")
    assert kw["headers"]["Authorization"] == "Bearer tok"
    monkeypatch.delenv("WHATSAPP_WABA_ID")
    assert (await wb.WhatsAppSender().link_business_account()).startswith("skipped")


async def test_profile_photo_and_about_are_set(monkeypatch, tmp_path):
    def reply(method, url):
        if method == "GET":
            return 200, {"data": [{"about": "Demo Clinic", "profile_picture_url": "https://x"}]}
        if url.endswith("/uploads"):
            return 200, {"id": "upload:abc"}
        if url.endswith("/upload:abc"):
            return 200, {"h": "HANDLE"}
        return 200, {"success": True}

    graph = FakeGraph(reply)
    monkeypatch.setattr(wb.aiohttp, "ClientSession", graph)
    monkeypatch.setenv("WHATSAPP_TOKEN", "tok")
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "123")
    monkeypatch.setenv("WHATSAPP_APP_ID", "777")
    pic = tmp_path / "dp.jpg"
    pic.write_bytes(b"\xff\xd8jpeg")
    out = await wb.WhatsAppSender().set_profile(about="Demo Clinic", photo=pic)
    assert out.startswith("OK") and "photo set: yes" in out
    start, upload, save = [c for c in graph.calls if c[0] == "POST"]
    assert start[1].endswith("/777/uploads") and start[2]["params"]["file_type"] == "image/jpeg"
    assert (
        upload[2]["headers"]["Authorization"] == "OAuth tok"
        and upload[2]["data"] == b"\xff\xd8jpeg"
    )
    assert save[1].endswith("/123/whatsapp_business_profile")
    assert save[2]["json"] == {
        "messaging_product": "whatsapp",
        "about": "Demo Clinic",
        "profile_picture_handle": "HANDLE",
    }
    assert len(wb.DEFAULT_ABOUT) <= 139


def test_verify_token_never_reaches_the_log(caplog):
    import logging

    import log_masking

    log_masking.install()
    with caplog.at_level(logging.INFO, logger="aiohttp.access"):
        logging.getLogger("aiohttp.access").info(
            '127.0.0.1 "GET /whatsapp/webhook?hub.mode=subscribe&hub.challenge=1&hub.verify_token=s3cret HTTP/1.1" 200'
        )
    assert "s3cret" not in caplog.text and "hub.verify_token=***" in caplog.text


# -- closing a quiet chat ---------------------------------------------------------------------
async def test_quiet_chat_is_closed_politely_once(real, tmp_path, monkeypatch):
    import asyncio

    monkeypatch.delenv("WHATSAPP_APP_SECRET", raising=False)
    d, _ = desk(real, [], tmp_path)
    sender = FakeSender()
    app = wb.make_app(d, sender)
    app[wb.HOOK].idle_close_s = 0.2
    async with TestClient(TestServer(app)) as c:
        await c.post("/whatsapp/webhook", data=_payload("Hi", "wamid.10"))
        await asyncio.sleep(0.6)
    assert [b for _, b in sender.sent] == [
        GREETING,
        wb.IDLE_CLOSE["en"].format(clinic="Demo Eye and Retina Clinic"),
    ]
    conv = d.store.load(WA)
    assert conv.closed and conv.history.messages()[-1].text_content.startswith(
        "Thank you for choosing"
    )
    assert await d.idle_close(WA) is None  # never twice


async def test_a_message_before_the_close_cancels_it(real, tmp_path, monkeypatch):
    import asyncio

    monkeypatch.delenv("WHATSAPP_APP_SECRET", raising=False)
    d, _ = desk(real, [("say", "Thank you. Is this a first visit?")], tmp_path)
    sender = FakeSender()
    app = wb.make_app(d, sender)
    app[wb.HOOK].idle_close_s = 0.4
    async with TestClient(TestServer(app)) as c:
        await c.post("/whatsapp/webhook", data=_payload("Hi", "wamid.20"))
        await asyncio.sleep(0.2)  # they answer before the 0.4 s are up
        await c.post("/whatsapp/webhook", data=_payload("I need an appointment", "wamid.21"))
        await asyncio.sleep(0.25)
        assert not any("closing this chat" in b for _, b in sender.sent)
        await asyncio.sleep(0.5)
    closes = [b for _, b in sender.sent if "closing this chat" in b]
    assert len(closes) == 1  # one close, timed from their last message
    assert d.store.load(WA).closed


@pytest.mark.parametrize(
    ("language", "script", "start"),
    [
        ("hi", "latin", "Demo Eye and Retina Clinic chunne"),
        ("hi", "devanagari", "डेमो आई एंड रेटिना क्लिनिक चुनने"),
    ],
)
async def test_close_message_follows_the_patients_script(real, tmp_path, language, script, start):
    d, _ = desk(real, [], tmp_path)
    conv = d._new(WA)
    conv.state.language, conv.state.chat_script = language, script
    conv.history.add_message(role="user", content="appointment chahiye")
    conv.history.add_message(role="assistant", content="Kya aap patient ka naam bata sakte hain?")
    d.store.save(conv)
    assert (await d.idle_close(WA)).startswith(start)


@pytest.mark.parametrize(
    ("last_bot", "emergency"),
    [
        ("Thank you for contacting Demo Eye and Retina Clinic. Take care.", False),
        ("Please go to the nearest eye emergency now. Is there anything else?", True),
    ],
)
async def test_no_close_after_goodbye_or_emergency(real, tmp_path, last_bot, emergency):
    d, _ = desk(real, [], tmp_path)
    conv = d._new(WA)
    conv.state.emergency_flagged = emergency
    conv.history.add_message(role="user", content="that's all")
    conv.history.add_message(role="assistant", content=last_bot)
    d.store.save(conv)
    assert await d.idle_close(WA) is None
    assert d.store.load(WA).closed  # no message, but the next one starts a new chat


async def test_chat_shows_up_on_the_dashboard(real, tmp_path, monkeypatch):
    sent = []

    class Sink:
        def send(self, sid, kind, **d):
            sent.append((sid, kind, d))

    monkeypatch.setattr(wb, "dash_feed", lambda: Sink())
    d, _ = desk(real, [], tmp_path)
    await d.reply(WA, "Hi")
    kinds = [k for _, k, _ in sent]
    assert kinds == ["session_start", "user_text", "bot_text"]
    sid = sent[0][0]
    assert sid.startswith("wa-") and WA not in sid and sent[0][2]["channel"] == "whatsapp"
    assert sent[0][2]["caller"].endswith("2345") and WA not in sent[0][2]["caller"]


async def test_a_finished_chat_comes_back_as_a_new_dashboard_session(real, tmp_path, monkeypatch):
    sent = []

    class Sink:
        def send(self, sid, kind, **d):
            sent.append((kind, d))

    monkeypatch.setattr(wb, "dash_feed", lambda: Sink())
    d, _ = desk(real, [("say", "Sure, our number is 98765 43210.")], tmp_path)
    conv = d._new(WA)
    conv.history.add_message(role="user", content="that's all")
    conv.history.add_message(role="assistant", content="Wish you a lovely day ahead.")
    d.store.save(conv)
    assert await d.idle_close(WA) is None  # goodbye already said: no closing message
    assert d.store.load(WA).closed and [k for k, _ in sent] == ["session_end"]
    await d.reply(WA, "what is your phone number 9876512345")
    kinds = [k for k, _ in sent]
    assert kinds[1:3] == ["session_start", "user_text"]
    assert "9876512345" not in sent[2][1]["text"]  # masked before it reaches the dashboard


async def test_after_a_close_the_next_message_starts_the_whole_flow_again(real, tmp_path):
    """Closed for inactivity, then "hi" a few minutes later: greeting, details and booking
    from the beginning, nothing carried over from the closed chat."""
    d, _ = desk(real, [], tmp_path)
    conv = d._new(WA)
    conv.state.language, conv.state.chat_script = "hi", "latin"
    conv.state.patient.name = "Kamla Devi"
    conv.history.add_message(role="user", content="mummy ka appointment chahiye")
    conv.history.add_message(role="assistant", content="Kya aap patient ki umar bata sakte hain?")
    d.store.save(conv)
    assert await d.idle_close(WA)  # the polite close
    out = await d.reply(WA, "Hi")
    assert out == [GREETING]
    conv = d.store.load(WA)
    assert not conv.closed and conv.state.patient.name is None and conv.state.language == "en"
    assert [m.text_content for m in conv.history.messages()] == ["Hi", out[0]]


async def test_a_chat_left_open_while_the_bot_was_off_still_starts_over(
    real, tmp_path, monkeypatch
):
    """The laptop was off when the 120 s ran out, so nothing closed the chat. Hours later
    "Hi" still starts from the beginning (no "Hello again" about the old booking)."""
    import time

    sent = []

    class Sink:
        def send(self, sid, kind, **d):
            sent.append((sid, kind))

    monkeypatch.setattr(wb, "dash_feed", lambda: Sink())
    d, _ = desk(real, [], tmp_path)
    conv = d._new(WA)
    conv.started_at = conv.updated_at = time.time() - 2 * 3600
    conv.state.patient.name = "Ashutosh"
    conv.history.add_message(role="user", content="Bye")
    conv.history.add_message(role="assistant", content="Goodbye, Ashutosh ji. Take care.")
    d.store.save(conv)
    old_sid = conv.dash
    out = await d.reply(WA, "Hi")
    assert out[0] == GREETING
    assert d.store.load(WA).state.patient.name is None
    assert sent[0] == (old_sid, "session_end")  # the old session is closed on the dashboard
    assert sent[1][1] == "session_start" and sent[1][0] != old_sid  # a session of its own


async def test_a_quick_reply_continues_the_same_chat(real, tmp_path):
    d, _ = desk(real, [("say", "Thank you. May I know the patient's name?")], tmp_path)
    await d.reply(WA, "Hi")
    await d.reply(WA, "appointment please")
    assert len(d.store.load(WA).history.messages()) == 4


def test_chat_note_has_the_day_as_well_as_the_time(real):
    from call_state import CallState
    from front_desk import chat_turn_note

    note = chat_turn_note(CallState(), real, "I need an appointment")
    day = real.now()
    assert f"{day:%A} {day.day} {day:%B}" in note  # "Monday 19 October": a chat can cross midnight
