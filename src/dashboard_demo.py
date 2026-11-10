"""Sample calls and chats for the dashboard (DASHBOARD-DEMO.bat): the page fills up exactly
as it does on a real call, through the same /events door, with made-up patients.

  - an outbound call (Clara) that goes to plan: verified, details, request, closing
  - an inbound call (Clara) with a patient who gets angry and raises their voice
  - a WhatsApp chat in Hinglish that asks where the clinic is
  - an agent-assisted call: a person answers, Claude suggests replies (needs the Anthropic key)

They repeat with new ids every few minutes. Demo scorecards go to logs/dashboard/demo/,
never into the real history."""

from __future__ import annotations

import asyncio
import math
import random
import time
import uuid

from dashboard_feed import Feed
from sop_checklists import INBOUND as INBOUND_SCRIPT
from sop_checklists import OUTBOUND as OUTBOUND_SCRIPT

OUTBOUND = [
    (
        "bot",
        "Demo Eye and Retina Clinic, good evening. This call is from Demo Eye and Retina Clinic. My name is Clara, a virtual assistant. We received your enquiry through our website, so how may I help you?",
    ),
    ("user", "Yes, I filled the form for my father's eye check-up.", -24, 2.8),
    (
        "bot",
        "Thank you. Before we go ahead, may I confirm I'm speaking with the person who sent the enquiry, or a family member handling it?",
    ),
    ("user", "Yes, I'm his son, Vikram.", -25, 1.6),
    (
        "bot",
        "Thank you, Vikram ji. Is this your father's first visit with us, or has he visited before?",
    ),
    ("user", "First visit. His name is Suresh Mehta, he's 68.", -24, 2.6),
    (
        "tool",
        "update_patient_details",
        {"patient_type": "new", "name": "Suresh Mehta", "age": 68},
        "Saved.",
    ),
    (
        "bot",
        "Thank you for sharing the details. Which day and time of day would suit him at our Green Park clinic?",
    ),
    ("user", "Saturday morning, if possible.", -23, 2.2),
    (
        "tool",
        "request_appointment",
        {
            "location_id": "greenpark",
            "preferred_day": "Saturday",
            "preferred_time": "morning",
            "use_calling_number": True,
        },
        "REQUEST REGISTERED SEC-R4821",
    ),
    (
        "bot",
        "I have registered an appointment request for Suresh Mehta with Dr. Arjun Mehta at our Green Park clinic for Saturday morning. Our team will call you to confirm the exact time. Please bring his previous eye reports and prescriptions, and come fifteen minutes early. If his eyes are dilated he will not be able to drive back, so please bring someone along, and please report on time. Is there anything else I may assist you with?",
    ),
    ("user", "No, that's perfect, thank you so much.", -24, 1.9),
    (
        "bot",
        "Thank you for your time and for connecting with Demo Eye and Retina Clinic. You were speaking with Clara. Wish you a lovely day ahead.",
    ),
]

INBOUND_ANGRY = [
    (
        "bot",
        "Hello, welcome to Demo Eye and Retina Clinic. I'm Clara, your virtual assistant. How may I help you today?",
    ),
    ("user", "मुझे अपनी माँ के लिए अपॉइंटमेंट चाहिए।", -22, 2.0),
    (
        "bot",
        "जी ज़रूर, मुझे आपकी मदद करके ख़ुशी होगी। आगे बढ़ने से पहले, मैं मरीज़ की कुछ जानकारी ले लेती हूँ। क्या यह उनकी पहली विज़िट है, या वे पहले भी आ चुकी हैं?",
    ),
    ("user", "पहले भी आ चुकी हैं। मैंने कल भी फ़ोन किया था, किसी ने कॉल बैक नहीं किया।", -18, 3.4),
    ("bot", "माफ़ कीजिए, आपको परेशानी हुई। क्या आप मरीज़ का नाम बता सकते हैं?"),
    ("user", "कितनी बार बताऊँ? कमला देवी! यह बहुत बेकार सर्विस है।", -12, 2.5, True),
    (
        "tool",
        "update_patient_details",
        {"patient_type": "existing", "name": "Kamla Devi"},
        "Saved.",
    ),
    ("bot", "मैं समझ सकती हूँ, और देरी के लिए माफ़ी चाहती हूँ। कमला देवी जी की उम्र क्या आप बता सकते हैं?"),
    ("user", "बहत्तर साल। गुड़गाँव सेंटर में चाहिए।", -16, 2.2),
    ("tool", "update_patient_details", {"age": 72, "location_id": "greenpark"}, "Saved."),
    (
        "tool",
        "request_appointment",
        {"location_id": "greenpark", "preferred_day": "Monday", "use_calling_number": True},
        "REQUEST REGISTERED SEC-R4822",
    ),
    (
        "bot",
        "धन्यवाद। मैंने कमला देवी जी के लिए ग्रीन पार्क क्लिनिक में सोमवार का अपॉइंटमेंट अनुरोध दर्ज कर दिया है। हमारी टीम आपको कॉल करके समय पक्का करेगी। आँखें फैलाई जाएँ तो किसी को साथ लाइए, और समय पर आइए। क्या मैं आपकी किसी और चीज़ में मदद कर सकती हूँ?",
    ),
    ("user", "ठीक है, बस यही। धन्यवाद।", -21, 1.6),
    ("bot", "डेमो आई एंड रेटिना क्लिनिक को कॉल करने के लिए धन्यवाद। आप क्लारा से बात कर रहे थे। अपना ध्यान रखिए।"),
]

WHATSAPP = [
    ("user", "Hi"),
    (
        "bot",
        "Hello, welcome to Demo Eye and Retina Clinic. I'm Clara, your virtual assistant. How may I help you today?",
    ),
    ("user", "Clinic kahan hai? Parking milegi?"),
    (
        "bot",
        "Hamara clinic Level 2, A-14 Green Park Main, New Delhi mein hai, Green Park metro ke paas. Parking ki details ke liye aap 98765 43210 par call kar sakte hain. Kya main aapke liye appointment request bhi register kar doon?",
    ),
    ("user", "Nahi abhi nahi, shukriya!"),
    (
        "bot",
        "Thank you for contacting Demo Eye and Retina Clinic. You were chatting with Clara. Wish you a lovely day ahead.",
    ),
]


# agent assist: speaker 0 = the person at the clinic, 1 = the caller (as the recogniser labels them)
ASSIST = [
    ("0", "Demo Eye and Retina Clinic, good afternoon, how may I help you?"),
    (
        "1",
        "Hello, I want to book an eye check-up for my father. His vision is blurry for some weeks.",
    ),
    ("0", "Sure, I can help with that. Is this his first visit with us?"),
    (
        "1",
        "Yes, first time. Which doctor is good for cataract, and what are the charges?",
    ),
    ("0", "Let me check that for you."),
    ("1", "His name is Ramesh Gupta, he is seventy."),
    ("0", "Thank you. Which day would suit you?"),
    ("1", "Saturday morning, if possible."),
    ("0", "Noted. Our team will call you to confirm the time. Is there anything else?"),
    ("1", "No, that's all. Thank you."),
    ("0", "Thank you for calling Demo Eye and Retina Clinic. Have a good day."),
]


async def _assist(f: Feed, script: list) -> None:
    sid = f"assist-demo-{uuid.uuid4().hex[:6]}"
    f.send(
        sid,
        "session_start",
        channel="assist",
        direction="inbound",
        persona="Agent",
        caller="xxxxxx5310",
    )
    await asyncio.sleep(1.0)
    for n, (speaker, text) in enumerate(script, 1):
        words = text.split()
        for k in range(1, len(words) + 1, 3):
            f.send(sid, "caption", text=" ".join(words[:k]))
            await asyncio.sleep(0.18)
        f.send(sid, "assist_segment", id=f"s{n}", text=text, speaker=speaker)
        await asyncio.sleep(5.5 if speaker == "1" else 2.0)  # time to read the suggestion
    await asyncio.sleep(2)
    f.send(sid, "session_end", reason="call ended")


async def _speak(f: Feed, sid: str, level: float, dur: float, loud_peaks: bool = False) -> None:
    """The bridge's voice events while the patient talks: ~5 a second, then the utterance."""
    steps = max(2, int(dur / 0.2))
    for k in range(steps):
        wobble = 4 * math.sin(k * 1.7) + random.uniform(-2.5, 2.5)
        peak = 6 if loud_peaks and k % 4 == 2 else 0
        f.send(sid, "voice", level=round(level + wobble + peak, 1), speaking=True)
        await asyncio.sleep(0.2)
    f.send(sid, "voice", level=-60.0, speaking=False)
    f.send(sid, "utterance", dur_s=dur, mean_db=level, peak_db=level + 9)


async def _call(f: Feed, script: list, *, direction: str, persona: str, caller: str) -> None:
    sid = f"demo-{direction}-{uuid.uuid4().hex[:6]}"
    f.send(
        sid, "session_start", channel="phone", direction=direction, persona=persona, caller=caller
    )
    await asyncio.sleep(1.0)
    n = 0
    for step in script:
        if step[0] == "bot":
            n += 1
            f.send(sid, "bot_state", state="thinking")
            await asyncio.sleep(random.uniform(0.6, 1.1))
            f.send(sid, "bot_state", state="speaking")
            f.send(
                sid,
                "bot_text",
                id=f"b{n}",
                text=step[1],
                latency={
                    "ttfa": round(random.uniform(0.65, 1.25), 2),
                    "llm": round(random.uniform(0.35, 0.8), 2),
                    "tts": 0.2,
                },
            )
            await asyncio.sleep(min(7.0, 1.2 + len(step[1]) / 45))
            f.send(sid, "bot_state", state="listening")
        elif step[0] == "user":
            _, text, level, dur, *rest = step
            barge = bool(rest and rest[0])
            if barge:
                f.send(sid, "barge_in")
            words = text.split()
            for k in range(1, len(words) + 1, 2):
                f.send(sid, "caption", text=" ".join(words[:k]))
                await asyncio.sleep(0.15)
            await _speak(f, sid, level, dur, loud_peaks=barge)
            f.send(sid, "user_text", id=f"u{n}", text=text)
            await asyncio.sleep(0.4)
        else:
            _, name, args, out = step
            f.send(sid, "tool", name=name, args=args, output=out)
            await asyncio.sleep(0.6)
    await asyncio.sleep(2)
    f.send(
        sid, "session_end", reason="caller hung up", cost_usd=round(random.uniform(0.05, 0.09), 4)
    )


async def _chat(f: Feed, script: list) -> None:
    sid = f"wa-demo{uuid.uuid4().hex[:5]}"
    f.send(
        sid,
        "session_start",
        channel="whatsapp",
        direction="inbound",
        persona="Clara",
        caller="xxxxxx4471",
    )
    for role, text in script:
        await asyncio.sleep(
            random.uniform(2.5, 5.0) if role == "user" else random.uniform(1.5, 3.0)
        )
        if role == "user":
            f.send(sid, "user_text", text=text)
        else:
            f.send(sid, "bot_text", text=text, latency={"ttfa": round(random.uniform(2.4, 3.6), 2)})
    await asyncio.sleep(3)
    f.send(sid, "session_end", reason="chat finished")


async def _loop(url: str) -> None:
    f = Feed(url)
    f.enabled = True
    await asyncio.sleep(1.5)
    while True:
        await asyncio.gather(
            _call(f, OUTBOUND, direction="outbound", persona="Clara", caller="xxxxxx2210"),
            _delayed(
                6,
                _call(f, INBOUND_ANGRY, direction="inbound", persona="Clara", caller="xxxxxx9087"),
            ),
            _delayed(3, _chat(f, WHATSAPP)),
            _delayed(9, _assist(f, ASSIST)),
        )
        await asyncio.sleep(25)


async def _delayed(s: float, coro) -> None:
    await asyncio.sleep(s)
    await coro


def seed_history(hub, n: int = 36) -> None:
    """A few weeks of made-up scorecards so History has something to show."""
    rnd = random.Random(7)
    now = time.time()
    out_steps = [i.id for i in OUTBOUND_SCRIPT.items if i.kind == "step"]
    in_steps = [i.id for i in INBOUND_SCRIPT.items if i.kind == "step"]
    for k in range(n):
        direction = rnd.choice(["inbound", "inbound", "outbound"])
        channel = "whatsapp" if direction == "inbound" and rnd.random() < 0.35 else "phone"
        steps = out_steps if direction == "outbound" else in_steps
        reach = rnd.choice([len(steps)] * 5 + list(range(3, len(steps))))
        status = {sid: ("done" if j < reach else "missed") for j, sid in enumerate(steps)}
        booked = reach >= steps.index("registered") + 1 if "registered" in steps else False
        mood = [round(rnd.uniform(-1.6, 1.8), 1) for _ in range(rnd.randint(3, 8))]
        started = now - (n - k) * rnd.uniform(5000, 9000)
        duration = round(rnd.uniform(70, 300), 1)
        done = sum(1 for v in status.values() if v == "done")
        hub.history.append(
            {
                "id": f"demo-past-{k}",
                "channel": channel,
                "direction": direction,
                "persona": "Clara" if direction == "outbound" else "Clara",
                "caller": f"xxxxxx{rnd.randint(1000, 9999)}",
                "started_at": started,
                "ended_at": started + duration,
                "duration_s": duration,
                "outcome": "Appointment request"
                if booked
                else rnd.choice(["Enquiry", "Left before booking", "Callback"]),
                "end_reason": "caller hung up",
                "sop": {
                    "score": round(done / len(steps), 3),
                    "done": done,
                    "total": len(steps),
                    "missed": [],
                    "broken": [],
                    "rules_kept": 3,
                    "rules": 3,
                },
                "steps": status,
                "mood": {
                    "avg": round(sum(mood) / len(mood), 2),
                    "min": min(mood),
                    "end": mood[-1],
                    "label": "satisfied" if mood[-1] > 0.5 else "neutral",
                },
                "turns": rnd.randint(6, 18),
                "barge_ins": rnd.randint(0, 4),
                "ttfa_p50": None if channel == "whatsapp" else round(rnd.uniform(0.7, 1.3), 2),
                "cost_usd": round(rnd.uniform(0.03, 0.1), 4),
                "analysis_usd": 0.0012,
                "flags": rnd.randint(0, 1),
            }
        )


def start_demo(url: str):
    async def _go(app) -> None:
        app["demo"] = asyncio.create_task(_loop(url))

    return _go
