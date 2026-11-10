"""The live dashboard: every call and WhatsApp chat as it happens, on this PC only.

    python src/dashboard_hub.py            # http://127.0.0.1:8767 (START-PHONE-TEST starts it)
    python src/dashboard_hub.py demo       # the same page, fed by sample calls and chats

The agent, the Exotel bridge and the WhatsApp bot POST small JSON events to /events
(dashboard_feed.py; fire-and-forget, so a stopped dashboard never slows a call). The hub
adds what the events don't carry by themselves and pushes it all to the browser over a
websocket:
  mood      every patient turn: Claude Haiku reads the words (any language), the bridge's
            loudness of that utterance against the caller's own normal says how it was said
  checklist the call script ticking as the bot says it (sop_checklists.py: patterns, tool
            calls, and Haiku for paraphrases, Hindi and the "never" rules)
  alerts    an upset patient, an emergency, a broken rule
  scorecard when a session ends: script score, mood, outcome, speed, cost, saved to
            logs/dashboard/sessions/ and listed in History
  assist    calls a person answers (Exotel Stream applet -> agent_assist.py): Claude suggests
            what to say next, the facts they need and the booking details heard (assist_brain)

It listens on 127.0.0.1 only and is never behind ngrok: it shows patient conversations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import statistics
import sys
import time
from pathlib import Path

from aiohttp import WSMsgType, web

sys.path.insert(0, str(Path(__file__).resolve().parent))

from call_cost import LLM_PRICES
from sop_checklists import Checklist, Item, checklist_for

logger = logging.getLogger("frontdesk.dashboard")
ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "dashboard_static"
LOG_DIR = Path(os.getenv("LOG_DIR", ROOT / "logs"))
HOST = "127.0.0.1"
PORT = int(os.getenv("DASHBOARD_PORT", "8767"))
JUDGE_MODEL = os.getenv("DASHBOARD_JUDGE_MODEL", "claude-haiku-5-5")
RAISED_DB = 6.0  # this much louder than the caller's own normal = a raised voice
# no events for this long: the session ended
QUIET_END_S = {"phone": 600.0, "whatsapp": 900.0, "assist": 600.0}
HISTORY_MAX = 500
# Agent Assist Mode also follows inbound calls the bot answers (what a person could say next).
# It costs one Claude reply per patient turn; ASSIST_ON_BOT_CALLS=0 limits it to assisted calls.
ASSIST_ON_BOT_CALLS = os.getenv("ASSIST_ON_BOT_CALLS", "1") != "0"


# -- mood --------------------------------------------------------------------------------------
LABELS = (
    "happy",
    "satisfied",
    "calm",
    "neutral",
    "confused",
    "anxious",
    "frustrated",
    "angry",
    "distressed",
)
_NEG = re.compile(
    r"\b(angry|worst|useless|rubbish|stupid|ridiculous|nonsense|frustrat\w*|fed up|wasting|"
    r"so many questions|why are you|not helping|bakwas|bekar|pagal|gussa|kya bakwas)\b|"
    r"बकवास|बेकार|गुस्सा|परेशान|हद है|पागल",
    re.IGNORECASE,
)
_ANX = re.compile(
    r"\b(pain|dard|worried|scared|urgent|emergency|bleeding|can'?t see|blurr?y?)\b|दर्द|डर|धुंधला",
    re.IGNORECASE,
)
_POS = re.compile(
    r"\b(thank\w*|great|perfect|wonderful|helpful|dhanyavaa?d|shukriya|bahut accha)\b|"
    r"धन्यवाद|शुक्रिया|बढ़िया|बहुत अच्छा",
    re.IGNORECASE,
)


def words_mood(text: str) -> dict:
    """No API key or Haiku down: a rough mood from the words alone."""
    if _NEG.search(text):
        return {"score": -1.5, "label": "frustrated", "why": "frustrated words", "by": "words"}
    if _ANX.search(text):
        return {"score": -0.5, "label": "anxious", "why": "mentions pain or worry", "by": "words"}
    if _POS.search(text):
        return {"score": 1.0, "label": "satisfied", "why": "thanks the bot", "by": "words"}
    return {"score": 0.0, "label": "neutral", "why": "plain request", "by": "words"}


def arousal(rel_db: float | None, barge_ins_recent: int = 0) -> float:
    """0 calm .. 1 agitated, from how loud they were against their own normal."""
    a = 0.0 if rel_db is None else max(0.0, min(1.0, (rel_db + 2.0) / 10.0))
    return round(min(1.0, a + 0.15 * barge_ins_recent), 3)


# -- Claude Haiku, in the background ---------------------------------------------------------------
_MOOD_SYS = (
    "You rate how a patient feels in a conversation with an eye hospital's front-desk "
    "assistant. They may write or speak English, Hindi or Hinglish; transcripts of calls "
    "can have recognition errors. Judge the latest patient message in its context. "
    'Answer JSON only: {"score": <-2..2, -2 angry or distressed, -1 frustrated or anxious, '
    '0 neutral, 1 satisfied, 2 happy>, "label": one of ' + ", ".join(LABELS) + ', "why": '
    '"<at most 8 words>"}'
)
_SCRIPT_SYS = (
    "You check an eye hospital front-desk assistant against its call script. Read the "
    "conversation and its latest assistant reply. Answer JSON only: "
    '{"done": [ids of the steps the assistant has now clearly completed, in this reply or '
    'earlier], "broken": [{"id": <rule id>, "why": "<at most 10 words>"}]} . Mark a rule '
    "broken only if the latest reply itself clearly breaks it. Hindi and paraphrases count."
)


def _json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


class Judge:
    def __init__(self) -> None:
        self.client = None
        if os.getenv("ANTHROPIC_API_KEY") and os.getenv("DASHBOARD_JUDGE", "1") != "0":
            try:
                import anthropic

                self.client = anthropic.AsyncAnthropic(max_retries=1, timeout=20.0)
            except Exception:
                logger.exception(
                    "Claude unavailable: mood from words only, checklist from patterns"
                )
        self.price = LLM_PRICES.get(JUDGE_MODEL, (0.0, 0.0, 0.0, 0.0))

    @property
    def ready(self) -> bool:
        return self.client is not None

    async def _ask(
        self, system: str, user: str, max_tokens: int = 220
    ) -> tuple[dict | None, float]:
        if not self.client:
            return None, 0.0
        try:
            r = await self.client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                # a short label, not a reasoning task: thinking would eat the token budget
                thinking={"type": "disabled"},
            )
        except Exception as e:
            logger.warning("dashboard judge failed: %s", str(e)[:200])
            return None, 0.0
        text = "".join(getattr(b, "text", "") for b in r.content)
        u = r.usage
        usd = u.input_tokens * self.price[0] + u.output_tokens * self.price[3]
        return _json(text), usd

    async def mood(self, context: str, text: str) -> tuple[dict | None, float]:
        out, usd = await self._ask(
            _MOOD_SYS, f"Conversation so far:\n{context}\n\nLatest patient message: {text}", 120
        )
        if not out:
            return None, usd
        try:
            score = max(-2.0, min(2.0, float(out.get("score", 0))))
        except (TypeError, ValueError):
            return None, usd
        label = str(out.get("label", "neutral")).lower()
        if label not in LABELS:
            label = "neutral"
        return {
            "score": score,
            "label": label,
            "why": str(out.get("why", ""))[:80],
            "by": "claude",
        }, usd

    async def script(
        self, context: str, reply: str, steps: list[Item], rules: list[Item]
    ) -> tuple[dict | None, float]:
        if not steps and not rules:
            return None, 0.0
        lines = [f"- {i.id}: {i.judge}" for i in steps]
        rule_lines = [f"- {i.id}: the assistant {i.judge}" for i in rules]
        user = (
            f"Conversation:\n{context}\n\nLatest assistant reply: {reply}\n\n"
            f"Script steps still open (id: what completes it):\n{chr(10).join(lines) or '- none'}"
            f"\n\nRules (id: what would break it):\n{chr(10).join(rule_lines) or '- none'}"
        )
        return await self._ask(_SCRIPT_SYS, user)


# -- one session ----------------------------------------------------------------------------------
class Session:
    def __init__(self, sid: str, meta: dict) -> None:
        self.id = sid
        self.channel = meta.get("channel", "phone")
        self.direction = meta.get("direction", "inbound")
        self.persona = meta.get("persona") or "Clara"
        self.caller = meta.get("caller", "")
        self.started_at = float(meta.get("at") or time.time())
        self.last_event = self.started_at
        self.ended_at: float | None = None
        self.end_reason = ""
        self.turns: list[dict] = []
        self.caption = ""
        self.bot_state = ""
        self.flags: list[dict] = []
        self.tools: list[dict] = []
        self.barge_ins: list[float] = []
        self.voice_level = -90.0
        self.baseline_db: float | None = None
        self._utt_levels: list[float] = []
        self._pending_utt: dict | None = None
        self.cost_usd: float | None = None
        self.analysis_usd = 0.0
        self.booking_flow = False
        self.outcome = ""
        self.last_alert = 0.0
        # agent assist: the current suggestion, the booking details heard, the after-call note
        self.assist: dict | None = None
        self.assist_form: dict = {}
        self.assist_wrap: dict | None = None
        self.speaker_roles: dict[str, str] = {}  # speaker id -> "bot" (agent) | "user"
        self.roles_fixed = (
            False  # the source says who is who (LiveKit tracks): Claude never guesses
        )
        self._assist_task: asyncio.Task | None = None
        self._assist_dirty = False
        self.checklist: Checklist = checklist_for(self.direction, self.channel)
        self.items: dict[str, dict] = self._fresh_items()
        self.lock = asyncio.Lock()
        self.tasks: set[asyncio.Task] = set()  # this session's mood / checklist analysis

    def _fresh_items(self) -> dict[str, dict]:
        return {
            i.id: {"status": "kept" if i.kind == "never" else "open", "at": None, "why": ""}
            for i in self.checklist.items
        }

    def set_direction(self, direction: str) -> None:
        self.direction = direction
        want = checklist_for(direction, self.channel)
        if want is not self.checklist:
            self.checklist = want
            self.items = self._fresh_items()

    @property
    def live(self) -> bool:
        return self.ended_at is None

    def item(self, iid: str) -> Item:
        return next(i for i in self.checklist.items if i.id == iid)

    def tick(self, iid: str, why: str = "", turn: str | None = None) -> bool:
        st = self.items.get(iid)
        if st is None or st["status"] != "open":
            return False
        st.update(status="done", at=time.time(), why=why, turn=turn)
        return True

    def context(self, n: int = 8) -> str:
        who = {"user": "Patient", "bot": self.persona}
        return "\n".join(f"{who[t['role']]}: {t['text']}" for t in self.turns[-n:])

    def mood_now(self) -> dict | None:
        for t in reversed(self.turns):
            if t["role"] == "user" and t.get("mood"):
                m = dict(t["mood"])
                recent = sum(1 for b in self.barge_ins if time.time() - b < 30)
                m["arousal"] = arousal((t.get("voice") or {}).get("rel_db"), recent)
                m["raised"] = bool((t.get("voice") or {}).get("raised"))
                return m
        return None

    def finalize(self, reason: str) -> None:
        if self.ended_at is not None:
            return
        self.ended_at = time.time()
        self.end_reason = reason or "ended"
        for i in self.checklist.items:
            st = self.items[i.id]
            if st["status"] != "open":
                continue
            if i.kind == "when" or (i.needs == "booking" and not self.booking_flow):
                st["status"] = "na"
            else:
                st["status"] = "missed"
        self.outcome = self._outcome()

    def _outcome(self) -> str:
        if self.assist_wrap and self.assist_wrap.get("outcome"):
            return self.assist_wrap["outcome"]
        names = {t["name"] for t in self.tools if not t.get("error")}
        if self.items.get("registered", {}).get("status") == "done":
            return "Appointment request"
        if "flag_emergency" in names:
            return "Emergency advice"
        if "transfer_to_department" in names:
            return "Transferred"
        if "mark_do_not_call" in names:
            return "Do not call"
        if "schedule_callback" in names:
            return "Callback"
        if not any(t["role"] == "user" for t in self.turns):
            return "No reply"
        if self.booking_flow:
            return "Left before booking"  # details started, no request registered
        return "Enquiry"

    def sop(self) -> dict:
        steps = [
            i
            for i in self.checklist.items
            if i.kind != "never" and self.items[i.id]["status"] != "na"
        ]
        done = [i for i in steps if self.items[i.id]["status"] == "done"]
        broken = [i for i in self.checklist.items if self.items[i.id]["status"] == "broken"]
        total = len(steps)
        return {
            "score": round(len(done) / total, 3) if total else None,
            "done": len(done),
            "total": total,
            "missed": [i.label for i in steps if self.items[i.id]["status"] == "missed"],
            "broken": [i.label for i in broken],
            "rules_kept": sum(
                1
                for i in self.checklist.items
                if i.kind == "never" and self.items[i.id]["status"] == "kept"
            ),
            "rules": sum(1 for i in self.checklist.items if i.kind == "never"),
        }

    def summary(self) -> dict:
        scores = [t["mood"]["score"] for t in self.turns if t["role"] == "user" and t.get("mood")]
        ttfa = [t["latency"]["ttfa"] for t in self.turns if (t.get("latency") or {}).get("ttfa")]
        end = self.ended_at or time.time()
        last = self.mood_now()
        return {
            "id": self.id,
            "channel": self.channel,
            "direction": self.direction,
            "persona": self.persona,
            "caller": self.caller,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_s": round(end - self.started_at, 1),
            "outcome": self.outcome,
            "end_reason": self.end_reason,
            "sop": self.sop(),
            "steps": {i.id: self.items[i.id]["status"] for i in self.checklist.items},
            "mood": {
                "avg": round(statistics.fmean(scores), 2) if scores else None,
                "min": min(scores) if scores else None,
                "end": scores[-1] if scores else None,
                "label": last["label"] if last else None,
            },
            "turns": len(self.turns),
            "barge_ins": len(self.barge_ins),
            "ttfa_p50": round(statistics.median(ttfa), 3) if ttfa else None,
            "cost_usd": self.cost_usd,
            "analysis_usd": round(self.analysis_usd, 6),
            "flags": len(self.flags),
            "note": (self.assist_wrap or {}).get("summary"),
        }

    def to_json(self) -> dict:
        return {
            **self.summary(),
            "live": self.live,
            "caption": self.caption,
            "bot_state": self.bot_state,
            "turns_detail": self.turns,
            "flags_detail": self.flags,
            "tools": self.tools[-30:],
            "assist": self.assist,
            "assist_form": self.assist_form,
            "assist_wrap": self.assist_wrap,
            "voice": {"baseline_db": self.baseline_db, "level": self.voice_level},
            "mood_now": self.mood_now(),
            "checklist": {
                "name": self.checklist.name,
                "items": [
                    {"id": i.id, "label": i.label, "kind": i.kind, **self.items[i.id]}
                    for i in self.checklist.items
                ],
            },
        }


# -- the hub ---------------------------------------------------------------------------------------
class Hub:
    def __init__(
        self,
        judge: Judge | None = None,
        store: Path | None = None,
        *,
        load: bool = True,
        brain=None,
    ) -> None:
        self.judge = judge or Judge()
        if brain is None:
            from assist_brain import AssistBrain

            brain = AssistBrain()
        self.brain = brain
        self.store = store or LOG_DIR / "dashboard" / "sessions"
        self.sessions: dict[str, Session] = {}
        self.history: list[dict] = []
        self.clients: set[web.WebSocketResponse] = set()
        self._dirty: set[str] = set()
        self._tasks: set[asyncio.Task] = set()
        if load:
            self._load_history()

    # -- persistence
    def _load_history(self) -> None:
        if not self.store.exists():
            return
        rows = []
        for p in sorted(self.store.glob("*.json"))[-HISTORY_MAX:]:
            try:
                rows.append(json.loads(p.read_text(encoding="utf-8"))["summary"])
            except Exception:
                continue
        self.history = sorted(rows, key=lambda r: r.get("started_at") or 0)

    def _save(self, s: Session) -> None:
        try:
            self.store.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(s.started_at))
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", s.id)[:60]
            (self.store / f"{stamp}-{safe}.json").write_text(
                json.dumps(
                    {"summary": s.summary(), "session": s.to_json()}, ensure_ascii=False, indent=1
                ),
                encoding="utf-8",
            )
        except OSError:
            logger.exception("could not save the scorecard")

    # -- pushing to browsers
    def _spawn(self, coro) -> None:
        t = asyncio.create_task(coro)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    async def _send_all(self, msg: dict) -> None:
        if not self.clients:
            return
        data = json.dumps(msg, ensure_ascii=False, default=str)
        for ws in list(self.clients):
            try:
                await ws.send_str(data)
            except Exception:
                self.clients.discard(ws)

    def _analyse(self, s: Session, coro) -> None:
        t = asyncio.create_task(coro)
        s.tasks.add(t)
        t.add_done_callback(s.tasks.discard)

    def touch(self, s: Session) -> None:
        self._dirty.add(s.id)

    async def flush_loop(self) -> None:
        while True:
            await asyncio.sleep(0.12)
            ids, self._dirty = self._dirty, set()
            for sid in ids:
                s = self.sessions.get(sid)
                if s:
                    await self._send_all({"type": "session", "session": s.to_json()})

    async def quiet_loop(self) -> None:
        while True:
            await asyncio.sleep(30)
            now = time.time()
            for s in list(self.sessions.values()):
                if s.live and now - s.last_event > QUIET_END_S.get(s.channel, 600):
                    await self.end(s, "no activity")

    def hello(self) -> dict:
        return {
            "type": "hello",
            "judge": self.judge.ready,
            "assist": self.brain.ready,
            "sessions": [s.to_json() for s in self.sessions.values()],
            "history": self.history,
            "checklists": {
                d: [{"id": i.id, "label": i.label, "kind": i.kind} for i in checklist_for(d).items]
                for d in ("inbound", "outbound")
            },
        }

    # -- events in
    def _session(self, ev: dict) -> Session:
        sid = str(ev.get("session") or "unknown")[:120]
        s = self.sessions.get(sid)
        if s is None or (not s.live and ev.get("type") == "session_start"):
            meta = dict(ev)
            if sid.startswith("wa-"):  # a chat that began before the dashboard did
                meta.setdefault("channel", "whatsapp")
            s = Session(sid, meta)
            self.sessions[sid] = s
            # keep the live list short: drop ended sessions after a while
            ended = [x for x in self.sessions.values() if not x.live]
            for old in sorted(ended, key=lambda x: x.ended_at or 0)[:-12]:
                self.sessions.pop(old.id, None)
        return s

    async def ingest(self, ev: dict) -> None:
        kind = ev.get("type")
        s = self._session(ev)
        at = float(ev.get("at") or time.time())
        s.last_event = max(s.last_event, at)
        if kind == "session_start":
            for k in ("channel", "persona", "caller"):
                if ev.get(k):
                    setattr(s, k, ev[k])
            if ev.get("direction"):
                s.set_direction(ev["direction"])
        elif kind == "caption":
            s.caption = str(ev.get("text", ""))[:300]
        elif kind == "bot_state":
            s.bot_state = str(ev.get("state", ""))
        elif kind == "user_text":
            self._user_text(s, ev, at)
        elif kind == "bot_text":
            self._bot_text(s, ev, at)
        elif kind == "tool":
            self._tool(s, ev, at)
        elif kind == "latency":
            self._latency(s, ev)
        elif kind == "utterance":
            self._utterance(s, ev, at)
        elif kind == "voice":
            s.voice_level = float(ev.get("level", -90))
            await self._send_all(
                {
                    "type": "voice",
                    "id": s.id,
                    "level": s.voice_level,
                    "speaking": bool(ev.get("speaking")),
                }
            )
            return
        elif kind == "assist_segment":
            self._assist_segment(s, ev, at)
        elif kind == "assist_error":
            self._alert(s, "serious", str(ev.get("text") or "Agent assist error")[:120], force=True)
        elif kind == "barge_in":
            s.barge_ins.append(at)
        elif kind == "session_end":
            if ev.get("cost_usd") is not None:
                s.cost_usd = float(ev["cost_usd"])
            self._spawn(self.end(s, str(ev.get("reason") or "ended")))  # never hold the sender
            return
        self.touch(s)

    def _user_text(self, s: Session, ev: dict, at: float) -> None:
        text = str(ev.get("text", "")).strip()
        if not text:
            return
        s.caption = ""
        turn = {"id": ev.get("id") or f"u{len(s.turns)}", "role": "user", "text": text, "at": at}
        utt = s._pending_utt
        if utt and abs(at - utt["at"]) < 8:
            turn["voice"] = utt
            s._pending_utt = None
        s.turns.append(turn)
        self._analyse(s, self._rate(s, turn))
        if ASSIST_ON_BOT_CALLS and s.channel == "phone" and s.direction != "outbound":
            self._assist_soon(s)

    def _bot_text(self, s: Session, ev: dict, at: float) -> None:
        text = str(ev.get("text", "")).strip()
        if not text:
            return
        turn = {
            "id": ev.get("id") or f"b{len(s.turns)}",
            "role": "bot",
            "text": text,
            "at": at,
            "interrupted": bool(ev.get("interrupted")),
        }
        if ev.get("latency"):
            turn["latency"] = ev["latency"]
        s.turns.append(turn)
        self._script_ticks(s, turn)

    def _script_ticks(self, s: Session, turn: dict) -> None:
        for i in s.checklist.items:
            if i.say and s.items[i.id]["status"] == "open" and i.says(turn["text"]):
                s.tick(i.id, "said it", turn["id"])
        self._analyse(s, self._check(s, turn))

    # -- agent assist: a person answers, Claude suggests ---------------------------------------
    def _assist_segment(self, s: Session, ev: dict, at: float) -> None:
        text = str(ev.get("text", "")).strip()
        if not text:
            return
        s.caption = ""
        spk = ev.get("speaker")
        spk = None if spk in (None, "") else str(spk)
        known = {"agent": "bot", "patient": "user"}.get(str(ev.get("role") or ""))
        if known and spk is not None:
            s.speaker_roles[spk] = known
            s.roles_fixed = True
        if spk is None:
            role = "user"  # no speaker from the recogniser: Claude labels the line
        elif spk in s.speaker_roles:
            role = s.speaker_roles[spk]
        else:
            # the stream starts when the agent picks up, so the first voice is the agent's
            role = "user" if "bot" in s.speaker_roles.values() else "bot"
            s.speaker_roles[spk] = role
        turn = {"id": ev.get("id") or f"s{len(s.turns)}", "role": role, "text": text, "at": at}
        if spk is not None:
            turn["speaker"] = spk
        s.turns.append(turn)
        self._assist_role(s, turn)
        self._assist_soon(s)

    def _assist_role(self, s: Session, turn: dict) -> None:
        """A line's speaker is known (or changed): mood for the patient, script for the agent."""
        if turn["role"] == "user":
            if not turn.get("mood"):
                self._analyse(s, self._rate(s, turn))
        else:
            turn.pop("mood", None)
            self._script_ticks(s, turn)

    def _assist_soon(self, s: Session) -> None:
        if not self.brain.ready:
            return
        if s._assist_task and not s._assist_task.done():
            s._assist_dirty = True  # rerun with the newest line as soon as this one is back
            return
        s._assist_task = asyncio.create_task(self._assist_loop(s))
        s.tasks.add(s._assist_task)
        s._assist_task.add_done_callback(s.tasks.discard)

    async def _assist_loop(self, s: Session) -> None:
        if s.channel == "assist":
            await asyncio.sleep(0.3)  # a sentence often arrives in two pieces
        while True:
            s._assist_dirty = False
            try:
                await self._assist_once(s)
            except Exception:
                logger.exception("agent assist suggestion crashed")
            if not s._assist_dirty or not s.live:
                return

    async def _assist_once(self, s: Session) -> None:
        recent = s.turns[-30:]
        if not recent:
            return
        lines = [{"id": t["id"], "speaker": t.get("speaker"), "text": t["text"]} for t in recent]
        if s.channel != "assist" or s.roles_fixed:  # who spoke is known, Claude need not guess
            for ln, t in zip(lines, recent, strict=True):
                ln["role"] = "agent" if t["role"] == "bot" else "patient"
        last = recent[-1]["id"]

        def early(part: dict) -> None:
            # the words (then why / kind / alert) arrive before the facts and the general
            # knowledge: the previous question's facts and knowledge must not stay under this one
            cur = s.assist or {}
            keep = cur if cur.get("for") == last else {**cur, "facts": [], "knowledge": None}
            part = dict(part)
            alert = part.pop("alert", None)
            s.assist = {**keep, **part, "for": last, "partial": True, "at": time.time()}
            if alert:
                self._assist_alert(s, alert)
            self.touch(s)

        today = time.strftime("%A %d %B %Y, %I:%M %p")
        out, usd = await self.brain.suggest(lines, s.assist_form, today, on_say=early)
        s.analysis_usd += usd
        if out:
            self._assist_apply(s, out, last)
        self.touch(s)

    def _assist_apply(self, s: Session, out: dict, last: str) -> None:
        to_role = {"agent": "bot", "patient": "user"}
        assisted = s.channel == "assist"  # only there are speakers guessed and lines unlabelled
        guess = assisted and not s.roles_fixed  # unless the source (LiveKit tracks) labelled them
        for spk, who in out.get("speakers", {}).items() if guess else ():
            role = to_role[who]
            if s.speaker_roles.get(spk, role) != role or spk not in s.speaker_roles:
                s.speaker_roles[spk] = role
                for t in s.turns:
                    if t.get("speaker") == spk and t["role"] != role:
                        t["role"] = role
                        self._assist_role(s, t)
        for lid, who in out.get("lines", {}).items() if guess else ():
            t = next((t for t in s.turns if t["id"] == lid and "speaker" not in t), None)
            if t and t["role"] != to_role[who]:
                t["role"] = to_role[who]
                self._assist_role(s, t)
        for k, v in (out.get("form") or {}).items():
            if v:
                s.assist_form[k] = v
        # form field -> the SOP step it completes (details to capture: name, age, number,
        # city, eye complaint)
        steps = {"name": "name", "age": "age", "city": "city", "concern": "complaint"}
        for key, item in steps.items() if assisted else ():
            if s.assist_form.get(key):
                s.booking_flow = True
                s.tick(item, "heard in the call", last)
        s.assist = {
            "say": out.get("say", ""),
            "why": out.get("why", ""),
            "kind": out.get("kind", "answer"),
            "facts": out.get("facts", []),
            "knowledge": out.get("knowledge"),
            "for": last,
            "partial": False,
            "at": time.time(),
        }
        if out.get("alert"):
            self._assist_alert(s, out["alert"])

    def _assist_alert(self, s: Session, a: dict) -> None:
        if not any(
            f["text"] == a["text"] or (f["level"] == a["level"] == "critical")
            for f in s.flags[-3:]
            if time.time() - f["at"] < 120
        ):
            self._alert(s, a["level"], a["text"], force=a["level"] == "critical")

    def _tool(self, s: Session, ev: dict, at: float) -> None:
        name = str(ev.get("name", ""))
        args = ev.get("args") or {}
        output = str(ev.get("output", ""))[:300]
        s.tools.append(
            {"name": name, "at": at, "error": bool(ev.get("error")), "output": output[:120]}
        )
        if name in ("update_patient_details", "request_appointment", "book_appointment"):
            s.booking_flow = True
        if not ev.get("error"):  # a failed tool call ticks nothing
            for i in s.checklist.items:
                if s.items[i.id]["status"] == "open" and i.done_by(name, args, output):
                    s.tick(i.id, name)
        if name == "flag_emergency":
            self._alert(s, "critical", "Emergency described by the patient", force=True)

    def _latency(self, s: Session, ev: dict) -> None:
        tid = ev.get("id")
        for t in reversed(s.turns):
            if t["role"] == "bot" and (tid is None or t["id"] == tid):
                t["latency"] = {k: ev.get(k) for k in ("ttfa", "llm", "tts", "eou")}
                return

    def _utterance(self, s: Session, ev: dict, at: float) -> None:
        mean = float(ev.get("mean_db", -90))
        s._utt_levels.append(mean)
        normal = s._utt_levels[:5]  # the caller's own usual level: their first few utterances
        s.baseline_db = round(statistics.median(normal), 1)
        rel = round(mean - s.baseline_db, 1) if len(normal) >= 2 else 0.0
        s._pending_utt = {
            "at": at,
            "dur_s": round(float(ev.get("dur_s", 0)), 2),
            "mean_db": round(mean, 1),
            "peak_db": round(float(ev.get("peak_db", -90)), 1),
            "rel_db": rel,
            "raised": rel >= RAISED_DB,
        }
        # the transcript may already be in: attach to the last patient turn without a voice
        for t in reversed(s.turns[-3:]):
            if t["role"] == "user" and "voice" not in t and abs(t["at"] - at) < 8:
                t["voice"] = s._pending_utt
                s._pending_utt = None
                self._maybe_upset(s, t)
                break

    # -- analysis (background)
    async def _rate(self, s: Session, turn: dict) -> None:
        context = s.context()
        mood, usd = await self.judge.mood(context, turn["text"])
        s.analysis_usd += usd
        turn["mood"] = mood or words_mood(turn["text"])
        self._maybe_upset(s, turn)
        self.touch(s)

    def _maybe_upset(self, s: Session, turn: dict) -> None:
        m = turn.get("mood")
        if not m:
            return
        raised = bool((turn.get("voice") or {}).get("raised"))
        if m["score"] <= -1.5:
            self._alert(s, "serious", f"Patient sounds {m['label']}: “{turn['text'][:60]}”")
        elif m["score"] <= -1 and raised:
            self._alert(s, "serious", "Patient is frustrated and raising their voice")

    async def _check(self, s: Session, turn: dict) -> None:
        async with s.lock:
            steps = [
                i
                for i in s.checklist.items
                if i.kind in ("step", "when") and i.judge and s.items[i.id]["status"] == "open"
            ]
            rules = [
                i
                for i in s.checklist.items
                if i.kind == "never" and s.items[i.id]["status"] == "kept"
            ]
            out, usd = await self.judge.script(s.context(10), turn["text"], steps, rules)
            s.analysis_usd += usd
            if not out:
                return
            open_ids = {i.id for i in steps}
            for iid in out.get("done") or []:
                if iid in open_ids:
                    s.tick(iid, "Claude: done", turn["id"])
            rule_ids = {i.id for i in rules}
            for b in out.get("broken") or []:
                iid = b.get("id") if isinstance(b, dict) else b
                if iid in rule_ids:
                    s.items[iid].update(
                        status="broken",
                        at=time.time(),
                        why=str(b.get("why", "") if isinstance(b, dict) else "")[:100],
                        turn=turn["id"],
                    )
                    self._alert(s, "critical", f"Rule broken: {s.item(iid).label}", force=True)
            self.touch(s)

    def _alert(self, s: Session, level: str, text: str, force: bool = False) -> None:
        now = time.time()
        if not force and now - s.last_alert < 60:
            return
        s.last_alert = now
        a = {"level": level, "text": text, "at": now}
        s.flags.append(a)
        self._spawn(self._send_all({"type": "alert", "id": s.id, "alert": a}))

    async def end(self, s: Session, reason: str) -> None:
        if not s.live:
            return
        # let the last mood / checklist answers land before scoring (bounded)
        pending = [t for t in s.tasks if not t.done()]
        if pending:
            await asyncio.wait(pending, timeout=12)
        if not s.live:
            return
        if s.channel == "assist" and self.brain.ready and s.turns:
            lines = [
                {"role": "agent" if t["role"] == "bot" else "patient", "text": t["text"]}
                for t in s.turns[-80:]
            ]
            s.assist_wrap, usd = await self.brain.wrap_up(lines)
            s.analysis_usd += usd
        if not s.live:
            return
        s.finalize(reason)
        summary = s.summary()
        self.history.append(summary)
        self.history = self.history[-HISTORY_MAX:]
        self._save(s)
        await self._send_all({"type": "session", "session": s.to_json()})
        await self._send_all({"type": "history", "summary": summary})
        logger.info(
            "session %s ended (%s): script %s, mood %s",
            s.id,
            reason,
            summary["sop"]["score"],
            summary["mood"]["avg"],
        )


# -- web ------------------------------------------------------------------------------------------
HUB = web.AppKey("hub", Hub)
LOOPS = web.AppKey("loops", list)


async def events(request: web.Request) -> web.Response:
    if request.remote not in ("127.0.0.1", "::1", None):
        return web.Response(status=403)
    hub = request.app[HUB]
    try:
        body = await request.json()
    except Exception:
        return web.Response(status=400, text="bad json")
    batch = body if isinstance(body, list) else [body]
    for ev in batch[:200]:
        if isinstance(ev, dict):
            try:
                await hub.ingest(ev)
            except Exception:
                logger.exception("bad dashboard event %s", str(ev)[:200])
    return web.Response(text="ok")


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    hub = request.app[HUB]
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    await ws.send_str(json.dumps(hub.hello(), ensure_ascii=False, default=str))
    hub.clients.add(ws)
    try:
        async for msg in ws:
            if msg.type == WSMsgType.ERROR:
                break
    finally:
        hub.clients.discard(ws)
    return ws


async def index(_: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


async def session_json(request: web.Request) -> web.Response:
    """A saved scorecard with its full transcript (History > open a row)."""
    sid = request.match_info["sid"]
    hub = request.app[HUB]
    s = hub.sessions.get(sid)
    if s:
        return web.json_response(s.to_json(), dumps=lambda o: json.dumps(o, ensure_ascii=False))
    for p in sorted(hub.store.glob("*.json"), reverse=True):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("summary", {}).get("id") == sid:
            return web.json_response(
                d["session"], dumps=lambda o: json.dumps(o, ensure_ascii=False)
            )
    return web.Response(status=404, text="not found")


def make_app(hub: Hub | None = None) -> web.Application:
    app = web.Application()
    app[HUB] = hub or Hub()
    app.router.add_post("/events", events)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/api/session/{sid}", session_json)
    app.router.add_get("/", index)
    app.router.add_static("/static/", STATIC, show_index=False)

    async def _no_cache(request: web.Request, response: web.StreamResponse) -> None:
        # the page's own files change with the code: never show an old copy after an update
        if request.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"

    app.on_response_prepare.append(_no_cache)

    async def _start(app: web.Application) -> None:
        h = app[HUB]
        app[LOOPS] = [asyncio.create_task(h.flush_loop()), asyncio.create_task(h.quiet_loop())]

    async def _stop(app: web.Application) -> None:
        for t in app[LOOPS]:
            t.cancel()

    app.on_startup.append(_start)
    app.on_cleanup.append(_stop)
    return app


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env.local")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    import log_masking

    log_masking.install()
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)  # one line per event is noise
    demo = len(sys.argv) > 1 and sys.argv[1] == "demo"
    hub = Hub(store=LOG_DIR / "dashboard" / "demo", load=False) if demo else Hub()
    if demo:
        from dashboard_demo import seed_history, start_demo

        seed_history(hub)
    app = make_app(hub)
    if demo:
        app.on_startup.append(start_demo(f"http://{HOST}:{PORT}/events"))
    logger.info(
        "dashboard on http://%s:%d%s (mood and checklist by %s)",
        HOST,
        PORT,
        " with sample calls" if demo else "",
        f"Claude {JUDGE_MODEL}"
        if hub.judge.ready
        else "word patterns only: ANTHROPIC_API_KEY not set",
    )
    web.run_app(app, host=HOST, port=PORT, print=None)


if __name__ == "__main__":
    main()
