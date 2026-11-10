// Front desk monitor: live calls and chats from the hub's websocket, rendered without a
// framework. Everything the page shows is also in text (the transcript is the table view).
import { Iris } from "./iris.js";

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};
const svgNS = "http://www.w3.org/2000/svg";
const s = (tag, attrs = {}) => {
  const n = document.createElementNS(svgNS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
};

const ICON = {
  done: "M4.5 8.5 7 11l4.5-6",
  missed: "M5 5l6 6M11 5l-6 6",
  broken: "M8 3.5v5.5M8 11.6v.4",
  na: "M4.5 8h7",
};
const ALERT_ICON = "M8 1.5 15 14H1L8 1.5Zm0 4v4.2m0 1.8v.6";
const PHONE = "M3 2.5 5.5 2l1.4 3.2-1.6 1.2a8 8 0 0 0 4.3 4.3l1.2-1.6L14 10.5 13.5 13c-6 .4-11-4.6-10.5-10.5Z";
const CHAT = "M2 3.5h12v7.5H7l-3.5 2.5V11H2z";
const HEADSET = "M8 2a6 6 0 0 0-6 6v4h3V8.5H3.5V8a4.5 4.5 0 0 1 9 0v.5H11V12h3V8a6 6 0 0 0-6-6Z";

const state = {
  sessions: new Map(),
  history: [],
  checklists: {},
  selected: null,
  pinned: false,
  judge: false,
  seen: new Map(), // session id -> {itemId: status}, to animate fresh ticks
  view: "live",
};

// -- mood colour -------------------------------------------------------------------------
function cssVar(n) {
  return getComputedStyle(document.documentElement).getPropertyValue(n).trim();
}
function hex(h) {
  const v = h.replace("#", "");
  return [0, 2, 4].map((i) => parseInt(v.slice(i, i + 2), 16));
}
function moodColor(score) {
  if (score == null) return cssVar("--line-strong");
  const t = Math.max(-1, Math.min(1, score / 2));
  const a = hex(cssVar("--mood-mid"));
  const b = hex(cssVar(t >= 0 ? "--mood-pos" : "--mood-neg"));
  const m = a.map((x, i) => Math.round(x + (b[i] - x) * Math.abs(t)));
  return `rgb(${m.join(",")})`;
}
const cap = (w) => (w ? w[0].toUpperCase() + w.slice(1) : "");

// -- formatting -----------------------------------------------------------------------------
const clock = (secs) => {
  secs = Math.max(0, Math.round(secs));
  return `${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, "0")}`;
};
const when = (ts) => {
  const d = new Date(ts * 1000);
  const today = new Date();
  const t = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  return d.toDateString() === today.toDateString()
    ? `Today ${t}`
    : `${d.toLocaleDateString([], { day: "numeric", month: "short" })}, ${t}`;
};
const INR_PER_USD = 88; // for display only; the scorecards keep exact USD
const rupees = (usd) => (usd == null ? "–" : `₹${(usd * INR_PER_USD).toFixed(2)}`);
const secsText = (v) => (v == null ? "–" : `${v.toFixed(1)} s`);
const kindLabel = (x) =>
  x.channel === "whatsapp"
    ? "WhatsApp chat"
    : x.channel === "assist"
      ? "Agent-assisted call"
      : x.direction === "outbound"
        ? "Outbound call"
        : "Inbound call";
const who = (x) => (x.channel === "assist" ? kindLabel(x) : `${kindLabel(x)} with ${x.persona}`);
/** the upper median, as the hub reports it; null for none */
const median = (a) => {
  const v = [...a].sort((x, y) => x - y);
  return v.length ? v[Math.floor(v.length / 2)] : null;
};

// -- websocket ------------------------------------------------------------------------------
let retry = 500;
function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onopen = () => {
    retry = 500;
    $("conn").textContent = "Live";
    $("conn").className = "conn on";
  };
  ws.onclose = () => {
    $("conn").textContent = "Reconnecting";
    $("conn").className = "conn off";
    setTimeout(connect, (retry = Math.min(8000, retry * 2)));
  };
  ws.onmessage = (m) => handle(JSON.parse(m.data));
}

function handle(msg) {
  if (msg.type === "hello") {
    state.judge = msg.judge;
    if (!msg.judge) {
      $("conn").textContent = "Live, mood from words only";
      $("conn").title = "No Anthropic key on this PC: mood and script checks use word patterns, not Claude.";
    }
    state.checklists = msg.checklists;
    state.sessions = new Map(msg.sessions.map((x) => [x.id, x]));
    state.history = msg.history;
    autoSelect();
    schedule();
  } else if (msg.type === "session") {
    state.sessions.set(msg.session.id, msg.session);
    autoSelect(msg.session);
    schedule();
  } else if (msg.type === "voice") {
    if (msg.id === state.selected) iris.setVoice(msg.level, msg.speaking);
  } else if (msg.type === "history") {
    state.history = state.history.filter((h) => h.id !== msg.summary.id).concat([msg.summary]);
    if (state.view === "history") renderHistory();
    renderToday();
  } else if (msg.type === "alert") {
    toast(msg.id, msg.alert);
  }
}

function autoSelect(changed) {
  const cur = state.sessions.get(state.selected);
  if (state.pinned && cur) return;
  const live = [...state.sessions.values()].filter((x) => x.live).sort((a, b) => b.started_at - a.started_at);
  if (changed && changed.live && (!cur || !cur.live)) state.selected = changed.id;
  else if (!cur) state.selected = live[0]?.id ?? [...state.sessions.values()].at(-1)?.id ?? null;
}

let queued = false;
function schedule() {
  if (queued) return;
  queued = true;
  requestAnimationFrame(() => {
    queued = false;
    renderRail();
    renderStage();
    renderSide();
    renderToday();
    if (state.view === "assist") renderAssistView();
  });
}

// -- rail -------------------------------------------------------------------------------------
// Buttons are kept and updated in place: rebuilding them on every event would swallow clicks.
const railItems = new Map(); // session id -> {li, button, name, time, chip, mood}

function railItem(x) {
  let r = railItems.get(x.id);
  if (!r) {
    const li = el("li");
    const b = el("button", "sess");
    b.type = "button";
    b.dataset.id = x.id;
    const name = el("span", "name");
    const time = el("span", "time");
    const meta = el("span", "meta");
    const chip = el("span", "chip");
    const mood = el("span");
    meta.append(chip, mood);
    b.append(name, time, meta);
    b.addEventListener("click", () => {
      state.selected = x.id;
      state.pinned = true;
      schedule();
    });
    li.append(b);
    r = { li, b, name, time, chip, mood, chipKey: "" };
    railItems.set(x.id, r);
  }
  const m = x.mood_now;
  if (x.id === state.selected) r.b.setAttribute("aria-current", "true");
  else r.b.removeAttribute("aria-current");
  r.b.style.setProperty("--tick", moodColor(m ? m.score : null));
  r.name.textContent = x.caller || (x.channel === "whatsapp" ? "WhatsApp patient" : "Caller");
  r.time.textContent = clock((x.ended_at || Date.now() / 1000) - x.started_at);
  const key = `${x.channel}/${x.direction}`;
  if (r.chipKey !== key) {
    const [glyph, word] =
      x.channel === "whatsapp" ? [CHAT, "Chat"] : x.channel === "assist" ? [HEADSET, "Assist"] : [PHONE, x.direction === "outbound" ? "Outbound" : "Inbound"];
    r.chip.replaceChildren(icon(glyph), document.createTextNode(word));
    r.chipKey = key;
  }
  r.mood.textContent = m ? cap(m.label) : x.live ? "Listening" : x.outcome || "Ended";
  return r.li;
}

function icon(d) {
  const v = s("svg", { viewBox: "0 0 16 16", "aria-hidden": "true" });
  v.append(s("path", { d }));
  return v;
}

function placeList(ul, nodes, emptyText) {
  if (!nodes.length) {
    if (!ul.querySelector(".empty")) ul.replaceChildren(el("li", "empty", emptyText));
    return;
  }
  ul.querySelector(".empty")?.remove();
  nodes.forEach((n, i) => {
    if (ul.children[i] !== n) ul.insertBefore(n, ul.children[i] || null);
  });
  while (ul.children.length > nodes.length) ul.lastElementChild.remove();
}

function renderRail() {
  const all = [...state.sessions.values()].sort((a, b) => b.started_at - a.started_at);
  for (const id of railItems.keys()) if (!state.sessions.has(id)) railItems.delete(id);
  placeList($("liveList"), all.filter((x) => x.live).map(railItem), "No calls or chats right now.");
  placeList($("endedList"), all.filter((x) => !x.live).map(railItem), "Ended sessions appear here.");
}

function tickClocks() {
  for (const [id, r] of railItems) {
    const x = state.sessions.get(id);
    if (x && x.live) r.time.textContent = clock(Date.now() / 1000 - x.started_at);
  }
}

// -- stage --------------------------------------------------------------------------------------
const iris = new Iris($("iris"));

function renderStage() {
  const x = state.sessions.get(state.selected);
  if (!x) {
    $("who").textContent = "No call or chat right now";
    $("moodWord").textContent = "Waiting";
    $("moodWhy").textContent =
      "When a patient calls or writes, their mood shows here: the iris turns blue when they are calm or happy and red when they are upset, and pulses as they speak.";
    $("facts").replaceChildren();
    $("transcript").replaceChildren();
    $("caption").textContent = "";
    $("botState").textContent = "";
    $("assistCard").hidden = true;
    $("stage").classList.remove("assist-mode");
    iris.setMood(null, 0);
    drawMood(null);
    return;
  }
  renderAssist(x);
  const mood = x.mood_now;
  $("who").textContent = `${who(x)}${x.live ? "" : " (ended)"}`;
  $("moodWord").textContent = mood ? cap(mood.label) : x.live ? "Listening" : "No mood read";
  $("moodWhy").textContent = mood
    ? `${mood.why ? cap(mood.why) + "." : ""}${mood.raised ? " Their voice is raised." : ""}`
    : "The mood appears after the patient's first message.";
  iris.setMood(mood ? mood.score : x.live ? 0 : null, mood ? mood.arousal || 0 : 0);
  if (!x.live) iris.setVoice(-90, false);

  const facts = [
    ["Caller", x.caller || "–"],
    [x.live ? "Going for" : "Lasted", clock(x.duration_s)],
    ["Turns", String(x.turns)],
  ];
  if (x.channel === "phone") {
    facts.push(["Talked over the bot", `${x.barge_ins} time${x.barge_ins === 1 ? "" : "s"}`]);
    const last = [...x.turns_detail].reverse().find((t) => t.role === "user" && t.voice);
    if (last) facts.push(["Last voice", last.voice.raised ? `Raised, +${last.voice.rel_db} dB` : "Usual level", last.voice.raised]);
  }
  if (!x.live) facts.push(["Outcome", x.outcome || "–"], ["Cost", rupees(x.cost_usd)]);
  $("facts").replaceChildren(
    ...facts.map(([k, v, warn]) => {
      const d = el("div");
      const dd = el("dd", warn ? "raised" : "", v);
      if (k === "Going for") dd.id = "durFact";
      d.append(el("dt", "", k), dd);
      return d;
    }),
  );

  const list = $("transcript");
  const nearBottom = list.getBoundingClientRect().bottom - innerHeight < 160;
  list.replaceChildren(...x.turns_detail.map((t) => turnItem(x, t)));
  $("caption").textContent = x.live ? x.caption || "" : "";
  const bs = $("botState");
  bs.textContent = x.live ? ({ speaking: `${x.persona} is speaking`, thinking: `${x.persona} is thinking`, listening: `${x.persona} is listening` }[x.bot_state] || "") : "";
  bs.className = `state ${x.bot_state || ""}`;
  if (nearBottom && x.live) list.lastElementChild?.scrollIntoView({ block: "nearest" });
  drawMood(x);
}

// -- agent assist card ---------------------------------------------------------------------------
const KIND_TEXT = { answer: "Answer", ask: "Next question", empathy: "Calm them first", emergency: "Emergency", closing: "Close the call" };
const FORM_TEXT = [
  ["patient_type", "Visit"], ["name", "Name"], ["age", "Age"], ["city", "City"], ["centre", "Centre"],
  ["doctor", "Doctor"], ["day", "Day"], ["time", "Time"], ["concern", "Concern"],
];

function renderAssist(x) {
  const card = $("assistCard");
  const on = x.channel === "assist";
  card.hidden = !on;
  $("stage").classList.toggle("assist-mode", on);
  if (!on) return;
  const a = x.assist;
  const wrap = !x.live && x.assist_wrap;
  card.className = `assist ${wrap ? "" : (a && a.kind) || ""}`;
  $("assistTitle").textContent = wrap ? "After-call note" : "Suggested reply";
  const kind = $("assistKind");
  kind.className = `chip ${!wrap && a && a.kind === "emergency" ? "emergency" : ""}`;
  kind.textContent = wrap ? wrap.outcome : a && a.kind ? KIND_TEXT[a.kind] || "" : "";
  kind.hidden = !kind.textContent;
  $("assistWhy").textContent = wrap ? "" : (a && a.why) || "";
  const say = $("assistSay");
  if (wrap) {
    say.textContent = wrap.summary + (wrap.follow_up ? ` Next: ${wrap.follow_up}` : "");
    say.className = "say wrap";
  } else {
    say.textContent = a && a.say ? a.say : x.live ? "Listening to the call. A suggestion appears after the patient speaks." : "The call has ended.";
    say.className = `say${a && a.partial ? " writing" : ""}${a && a.say ? "" : " wrap"}`;
  }
  const facts = (a && a.facts) || [];
  $("assistFacts").replaceChildren(
    ...(facts.length
      ? facts.map((f) => {
          const li = el("li");
          if (f.title) li.append(el("b", "", f.title));
          li.append(document.createTextNode(f.text));
          return li;
        })
      : [el("li", "empty", "None needed right now.")]),
  );
  const form = x.assist_form || {};
  const rows = FORM_TEXT.filter(([k]) => form[k]);
  $("assistForm").replaceChildren(
    ...(rows.length
      ? rows.flatMap(([k, label]) => [el("dt", "", label), el("dd", "", String(form[k]))])
      : [el("dd", "empty", "Nothing yet.")]),
  );
}

// -- agent assist mode: a full screen of what to say, for the person on the call ---------------------
/** the call to help with: a live assisted call, else a live inbound bot call, else the latest of either */
const assistable = (x) => x.channel === "assist" || (x.channel === "phone" && x.direction !== "outbound");
function assistSession() {
  const all = [...state.sessions.values()].filter(assistable).sort((a, b) => b.started_at - a.started_at);
  return (
    all.find((x) => x.live && x.channel === "assist") ||
    all.find((x) => x.live) ||
    all.find((x) => x.assist || x.assist_wrap) ||
    all[0] ||
    null
  );
}

/** the answer from general knowledge, when the patient asked what the clinic facts don't cover */
function renderKnowledge(a, wrap) {
  const kn = !wrap && a && a.knowledge;
  const looking = !wrap && !kn && a && a.partial;
  const box = $("avKnow");
  box.hidden = !kn && !looking;
  box.classList.toggle("pending", !!looking);
  if (!kn) {
    $("avKnowApprox").hidden = true;
    $("avKnowQ").textContent = looking ? "Checking whether this needs general knowledge..." : "";
    $("avKnowA").textContent = "";
    $("avKnowPoints").replaceChildren();
    $("avKnowConfirm").textContent = "";
    return;
  }
  $("avKnowApprox").hidden = !kn.approximate;
  $("avKnowQ").textContent = kn.question || "";
  $("avKnowA").textContent = kn.answer;
  $("avKnowPoints").replaceChildren(...(kn.points || []).map((p) => el("li", "", p)));
  $("avKnowConfirm").textContent = kn.confirm ? `Only the team can confirm: ${kn.confirm}` : "";
}

function renderAssistView() {
  const x = assistSession();
  const status = $("avStatus");
  const card = $("avSayCard");
  if (!x) {
    status.textContent = "Waiting for a call. Suggestions appear here as soon as a patient speaks on any call.";
    card.className = "av-say";
    $("avKind").hidden = true;
    $("avSay").textContent = "Nothing to answer yet.";
    $("avWhy").textContent = "";
    $("avAlert").hidden = true;
    $("avKnow").hidden = true;
    $("avAsked").textContent = "–";
    $("avFacts").replaceChildren(el("li", "empty", "None needed right now."));
    $("avForm").replaceChildren(el("dd", "empty", "Nothing yet."));
    $("avLines").replaceChildren();
    return;
  }
  const a = x.assist;
  const wrap = !x.live && x.assist_wrap;
  status.replaceChildren(
    document.createTextNode(
      `${x.channel === "phone" ? `${x.persona} (the bot) is answering this call. Suggestions are what a person could say. ` : ""}${x.live ? "Listening to the call, " : "Call ended, lasted "}`,
    ),
    Object.assign(el("span", "", clock(x.live ? Date.now() / 1000 - x.started_at : x.duration_s)), { id: "avClock" }),
    document.createTextNode(x.caller ? `  ·  ${x.caller}` : ""),
  );
  status.dataset.live = x.live ? "1" : "";

  const flag = [...(x.flags_detail || [])].reverse().find((f) => Date.now() / 1000 - f.at < 180);
  const alert = $("avAlert");
  alert.hidden = !flag || !x.live;
  if (flag) {
    alert.textContent = flag.text;
    alert.className = `av-alert ${flag.level}`;
  }

  card.className = `av-say ${wrap ? "" : (a && a.kind) || ""}`;
  $("avSayLabel").textContent = wrap ? "After-call note" : "Say this";
  const kind = $("avKind");
  kind.className = `chip ${!wrap && a && a.kind === "emergency" ? "emergency" : ""}`;
  kind.textContent = wrap ? wrap.outcome : a && a.kind ? KIND_TEXT[a.kind] || "" : "";
  kind.hidden = !kind.textContent;
  const say = $("avSay");
  if (wrap) {
    say.textContent = wrap.summary + (wrap.follow_up ? ` Next: ${wrap.follow_up}` : "");
    say.className = "av-say-text wrap";
  } else {
    say.textContent = a && a.say ? a.say : x.live ? "Listening. A suggestion appears after the patient speaks." : "The call has ended.";
    say.className = `av-say-text${a && a.partial ? " writing" : ""}${a && a.say ? "" : " wrap"}`;
  }
  $("avWhy").textContent = wrap ? "" : (a && a.why) || "";

  const lines = x.turns_detail || [];
  const lastPatient = [...lines].reverse().find((t) => t.role === "user");
  $("avAsked").textContent = lastPatient ? lastPatient.text : "–";

  renderKnowledge(a, wrap);
  const facts = (a && a.facts) || [];
  $("avFacts").replaceChildren(
    ...(facts.length
      ? facts.map((f) => {
          const li = el("li");
          if (f.title) li.append(el("b", "", f.title));
          li.append(document.createTextNode(f.text));
          return li;
        })
      : [el("li", "empty", "None needed right now.")]),
  );
  const form = x.assist_form || {};
  const rows = FORM_TEXT.filter(([k]) => form[k]);
  $("avForm").replaceChildren(
    ...(rows.length
      ? rows.flatMap(([k, label]) => [el("dt", "", label), el("dd", "", String(form[k]))])
      : [el("dd", "empty", "Nothing yet.")]),
  );
  $("avLines").replaceChildren(
    ...lines.slice(-6).map((t) => {
      const li = el("li", `av-line ${t.role}`);
      li.append(el("span", "speaker", t.role === "user" ? "Patient" : x.channel === "assist" ? "You" : x.persona), el("span", "text", t.text));
      return li;
    }),
  );
}

function turnItem(x, t) {
  const li = el("li", `turn ${t.role}${t.interrupted ? " cut" : ""}`);
  li.append(el("span", "speaker", t.role === "user" ? "Patient" : x.persona));
  const body = el("div");
  body.append(el("p", "text", t.text));
  const tags = el("div", "tags");
  if (t.mood) {
    const c = el("span", "chip tag-mood", `${cap(t.mood.label)} ${t.mood.score > 0 ? "+" : ""}${t.mood.score}`);
    c.style.setProperty("--c", moodColor(t.mood.score));
    c.title = t.mood.why || "";
    tags.append(c);
  } else if (t.role === "user") tags.append(el("span", "chip", "Reading mood"));
  if (t.voice && t.voice.raised) tags.append(el("span", "chip tag-loud", `Voice raised +${t.voice.rel_db} dB`));
  if (t.latency && t.latency.ttfa) tags.append(el("span", "chip", `Replied in ${t.latency.ttfa.toFixed(1)} s`));
  if (tags.childElementCount) body.append(tags);
  li.append(body);
  return li;
}

// -- charts --------------------------------------------------------------------------------------
const tip = $("tip");
function showTip(evt, html) {
  tip.innerHTML = html;
  tip.hidden = false;
  const r = (evt.target.getBoundingClientRect && evt.target.getBoundingClientRect()) || { left: evt.clientX, top: evt.clientY, width: 0 };
  const x = (evt.clientX ?? r.left + r.width / 2) + 14;
  const y = (evt.clientY ?? r.top) + 14;
  tip.style.left = `${Math.min(x, innerWidth - 340)}px`;
  tip.style.top = `${Math.min(y, innerHeight - 90)}px`;
}
const hideTip = () => (tip.hidden = true);
const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

function hoverable(node, html) {
  node.setAttribute("tabindex", "0");
  node.addEventListener("pointerenter", (e) => showTip(e, html));
  node.addEventListener("pointermove", (e) => showTip(e, html));
  node.addEventListener("pointerleave", hideTip);
  node.addEventListener("focus", (e) => showTip(e, html));
  node.addEventListener("blur", hideTip);
}

function drawMood(x) {
  const box = $("moodPlot");
  const W = Math.max(320, box.clientWidth || 600);
  const H = 150;
  const L = 64, R = 12, T = 10, B = 24;
  const v = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Patient mood for each message" });
  const y = (sc) => T + ((2 - sc) / 4) * (H - T - B);
  for (const [sc, label] of [[2, "Happy"], [0, "Neutral"], [-2, "Upset"]]) {
    v.append(s("line", { x1: L, x2: W - R, y1: y(sc), y2: y(sc), class: sc === 0 ? "axis" : "grid" }));
    const t = s("text", { x: L - 10, y: y(sc) + 4, "text-anchor": "end" });
    t.textContent = label;
    v.append(t);
  }
  const pts = x ? x.turns_detail.filter((t) => t.role === "user" && t.mood) : [];
  if (!pts.length) {
    const t = s("text", { x: L + 8, y: y(1), class: "empty" });
    t.textContent = x ? "Waiting for the patient's first message." : "No conversation selected.";
    v.append(t);
  } else {
    const n = Math.max(pts.length, 6);
    const px = (i) => L + 10 + (i / (n - 1)) * (W - L - R - 20);
    if (pts.length > 1) v.append(s("path", { class: "line", d: pts.map((p, i) => `${i ? "L" : "M"}${px(i)},${y(p.mood.score)}`).join("") }));
    pts.forEach((p, i) => {
      const cx = px(i), cy = y(p.mood.score);
      v.append(s("circle", { cx, cy, r: 6, class: "dot", fill: moodColor(p.mood.score) }));
      const hit = s("circle", { cx, cy, r: 14, class: "hit" });
      hoverable(hit, `<b>${esc(cap(p.mood.label))} (${p.mood.score > 0 ? "+" : ""}${p.mood.score})</b>${esc(p.text.slice(0, 120))}${p.voice && p.voice.raised ? "<br>Voice raised +" + p.voice.rel_db + " dB" : ""}`);
      v.append(hit);
    });
    const last = pts.at(-1);
    const lt = s("text", { x: Math.min(px(pts.length - 1) + 12, W - 70), y: y(last.mood.score) - 10, class: "lbl" });
    lt.textContent = cap(last.mood.label);
    v.append(lt);
  }
  box.replaceChildren(v);
}

function drawSpeed(x) {
  const box = $("speedPlot");
  const W = Math.max(280, box.clientWidth || 320);
  const H = 120;
  const L = 34, R = 8, T = 8, B = 20;
  const v = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Seconds the bot took to start each reply" });
  const turns = x ? x.turns_detail.filter((t) => t.role === "bot" && t.latency && t.latency.ttfa) : [];
  const phone = x && x.channel === "phone";
  const max = Math.max(phone ? 2 : 5, ...turns.map((t) => t.latency.ttfa));
  const y = (val) => T + (1 - val / max) * (H - T - B);
  for (const g of [0, max / 2, max]) {
    v.append(s("line", { x1: L, x2: W - R, y1: y(g), y2: y(g), class: g === 0 ? "axis" : "grid" }));
    const t = s("text", { x: L - 6, y: y(g) + 4, "text-anchor": "end" });
    t.textContent = `${g.toFixed(g < 2 && g % 1 ? 1 : 0)}s`;
    v.append(t);
  }
  if (!turns.length) {
    const t = s("text", { x: L + 8, y: y(max / 2) - 6, class: "empty" });
    t.textContent = "Appears after the bot's first reply.";
    v.append(t);
    $("speedNote").textContent = "";
  } else {
    const slot = (W - L - R) / Math.max(turns.length, 8);
    const bw = Math.min(18, slot - 4);
    turns.forEach((t, i) => {
      const val = t.latency.ttfa;
      const color = !phone ? cssVar("--brand") : val <= 0.9 ? cssVar("--good") : val <= 1.5 ? cssVar("--warning") : cssVar("--serious");
      const x0 = L + i * slot + (slot - bw) / 2;
      const top = y(val), base = y(0), r = Math.min(4, bw / 2, base - top);
      v.append(s("path", { d: `M${x0},${base}V${top + r}q0,-${r} ${r},-${r}h${bw - 2 * r}q${r},0 ${r},${r}V${base}Z`, fill: color }));
      const hit = s("rect", { x: x0 - 3, y: T, width: bw + 6, height: H - T - B, class: "hit" });
      hoverable(hit, `<b>${val.toFixed(2)} s to the first word</b>${esc(t.text.slice(0, 90))}`);
      v.append(hit);
    });
    if (phone) {
      v.append(s("line", { x1: L, x2: W - R, y1: y(0.9), y2: y(0.9), class: "axis" }));
      const tt = s("text", { x: W - R, y: y(0.9) - 4, "text-anchor": "end", class: "lbl" });
      tt.textContent = "target 0.9 s";
      v.append(tt);
      const ok = turns.filter((t) => t.latency.ttfa <= 0.9).length;
      $("speedNote").innerHTML = `<b>${ok}</b> of ${turns.length} on target`;
    } else {
      const med = median(turns.map((t) => t.latency.ttfa));
      $("speedNote").innerHTML = `typical <b>${med.toFixed(1)} s</b>`;
    }
  }
  box.replaceChildren(v);
}

// -- side: checklist, speed, alerts ------------------------------------------------------------------
const STATUS_TEXT = { done: "Done", open: "Not yet", missed: "Missed", na: "Not needed", broken: "Broken", kept: "Kept" };

function stepItem(sid, it) {
  const status = it.status;
  const seen = state.seen.get(sid) || {};
  const fresh = seen[it.id] && seen[it.id] !== status && (status === "done" || status === "broken");
  const li = el("li", `step ${status}${fresh ? " fresh" : ""}`);
  const m = el("span", "mark-s");
  const glyph = { done: ICON.done, kept: ICON.done, missed: ICON.missed, broken: ICON.broken, na: ICON.na }[status];
  if (glyph) {
    const v = s("svg", { viewBox: "0 0 16 16", "aria-hidden": "true" });
    v.append(s("path", { d: glyph, fill: "none", stroke: "currentColor", "stroke-width": status === "broken" ? 2.4 : 2, "stroke-linecap": "round", "stroke-linejoin": "round" }));
    v.style.color = status === "missed" ? "#1b0d06" : "#fff";
    m.append(v);
  }
  const txt = el("span", "lbl", it.label);
  txt.append(el("span", "ev", `${STATUS_TEXT[status]}${it.why && status === "broken" ? `: ${it.why}` : ""}`));
  li.append(m, txt);
  li.setAttribute("aria-label", `${it.label}: ${STATUS_TEXT[status]}`);
  return li;
}

function renderSide() {
  const x = state.sessions.get(state.selected);
  if (!x) {
    $("checklistName").textContent = "Call script";
    $("sopScore").textContent = "";
    for (const id of ["steps", "whens", "rules"]) $(id).replaceChildren();
    $("steps").append(el("li", "empty", "The script ticks here as the bot follows it."));
    $("speedPlot").closest("section").hidden = false;
    drawSpeed(null);
    $("alerts").replaceChildren(el("li", "empty", "Nothing needs attention."));
    return;
  }
  const items = x.checklist.items;
  $("checklistName").textContent = x.checklist.name;
  const sop = x.sop;
  if (x.live) {
    const steps = items.filter((i) => i.kind === "step");
    const done = steps.filter((i) => i.status === "done").length;
    $("sopScore").innerHTML = `<b>${done}</b> of ${steps.length} so far`;
  } else {
    $("sopScore").innerHTML = sop.score == null ? "" : `<b>${Math.round(sop.score * 100)}%</b> followed`;
  }
  $("steps").replaceChildren(...items.filter((i) => i.kind === "step").map((i) => stepItem(x.id, i)));
  $("whens").replaceChildren(...items.filter((i) => i.kind === "when").map((i) => stepItem(x.id, i)));
  $("rules").replaceChildren(...items.filter((i) => i.kind === "never").map((i) => stepItem(x.id, i)));
  state.seen.set(x.id, Object.fromEntries(items.map((i) => [i.id, i.status])));
  $("speedPlot").closest("section").hidden = x.channel === "assist"; // a person replies: no bot speed
  drawSpeed(x);
  const flags = x.flags_detail || [];
  $("alerts").replaceChildren(
    ...(flags.length
      ? [...flags].reverse().map((a) => {
          const li = el("li", `alert ${a.level}`);
          li.append(icon(ALERT_ICON));
          const body = el("div", "", a.text);
          const t = el("time", "", new Date(a.at * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }));
          body.append(t);
          li.append(body);
          return li;
        })
      : [el("li", "empty", "Nothing needs attention.")]),
  );
}

// -- toasts ---------------------------------------------------------------------------------------
function toast(sid, a) {
  const t = el("div", `toast ${a.level}`);
  const x = state.sessions.get(sid);
  t.append(el("b", "", a.text));
  const close = el("button", "", "Dismiss");
  close.addEventListener("click", () => t.remove());
  const open = el("button", "open", x ? `Open ${x.channel === "whatsapp" ? "chat" : "call"}` : "Open");
  open.addEventListener("click", () => {
    state.selected = sid;
    state.pinned = true;
    setView("live");
    schedule();
    t.remove();
  });
  t.append(close, open);
  $("toasts").prepend(t);
  setTimeout(() => t.remove(), 20000);
}

// -- today strip ---------------------------------------------------------------------------------------
function renderToday() {
  const start = new Date();
  start.setHours(0, 0, 0, 0);
  const t0 = start.getTime() / 1000;
  const live = [...state.sessions.values()].filter((x) => x.live).length;
  const today = state.history.filter((h) => h.started_at >= t0);
  const booked = today.filter((h) => h.outcome === "Appointment request").length;
  const box = $("today");
  box.replaceChildren();
  for (const [n, label] of [[live, "live now"], [today.length, "today"], [booked, "appointment requests today"]]) {
    const span = el("span");
    span.append(el("b", "", String(n)), document.createTextNode(label));
    box.append(span);
  }
}

// -- history ----------------------------------------------------------------------------------------
function filtered() {
  const d = $("hDirection").value, c = $("hChannel").value;
  return state.history.filter((h) => (d === "all" || h.direction === d) && (c === "all" || h.channel === c));
}

function renderHistory() {
  const rows = filtered();
  const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : null);
  const scores = rows.map((r) => r.sop.score).filter((v) => v != null);
  const moods = rows.map((r) => r.mood.avg).filter((v) => v != null);
  // a chat reply (seconds of typing) and a spoken reply are not comparable: calls when there are any
  const calls = rows.filter((r) => r.channel === "phone");
  const speed = median((calls.length ? calls : rows).map((r) => r.ttfa_p50).filter((v) => v != null));
  const booked = rows.filter((r) => r.outcome === "Appointment request").length;
  const upset = rows.filter((r) => r.mood.min != null && r.mood.min <= -1.5).length;
  const cost = mean(rows.map((r) => r.cost_usd).filter((v) => v != null));
  const stats = [
    ["Conversations", String(rows.length), `${rows.filter((r) => r.channel === "whatsapp").length} on WhatsApp`],
    ["Appointment requests", String(booked), rows.length ? `${Math.round((booked / rows.length) * 100)}% of conversations` : ""],
    ["Script followed", scores.length ? `${Math.round(mean(scores) * 100)}%` : "–", "average per conversation"],
    ["Patient mood", moods.length ? cap(moodWordFor(mean(moods))) : "–", `${upset} with an upset moment`],
    ["Reply speed", speed == null ? "–" : `${speed.toFixed(2)} s`, calls.length || !rows.length ? "typical call reply" : "typical chat reply"],
    ["Cost", cost == null ? "–" : rupees(cost), "average per conversation"],
  ];
  $("hStats").replaceChildren(
    ...stats.map(([k, v, d]) => {
      const n = el("div", "stat");
      n.append(el("div", "k", k), el("div", "v", v), el("div", "d", d));
      return n;
    }),
  );
  drawFunnel(rows);
  drawTrend(rows);
  const body = $("hTable").tBodies[0];
  body.replaceChildren(
    ...[...rows].reverse().map((r) => {
      const tr = el("tr");
      tr.tabIndex = 0;
      const moodCell = el("td");
      if (r.mood.avg != null) {
        const d = el("span", "dotc");
        d.style.setProperty("--c", moodColor(r.mood.avg));
        moodCell.append(d, document.createTextNode(cap(moodWordFor(r.mood.avg))));
      } else moodCell.textContent = "–";
      const cells = [
        el("td", "", when(r.started_at)),
        el("td", "", kindLabel(r)),
        el("td", "", r.caller || "–"),
        el("td", "", r.outcome || "–"),
        el("td", "num", r.sop.score == null ? "–" : `${Math.round(r.sop.score * 100)}%`),
        moodCell,
        el("td", "num", clock(r.duration_s)),
        el("td", "num", secsText(r.ttfa_p50)),
        el("td", "num", rupees(r.cost_usd)),
      ];
      tr.append(...cells);
      const open = () => openPast(r.id);
      tr.addEventListener("click", open);
      tr.addEventListener("keydown", (e) => e.key === "Enter" && open());
      return tr;
    }),
  );
  if (!rows.length) {
    const tr = el("tr");
    const td = el("td", "", "No conversations yet. They appear here when a call or chat ends.");
    td.colSpan = 9;
    tr.append(td);
    body.append(tr);
  }
}

function moodWordFor(v) {
  if (v == null) return "–";
  if (v >= 1.2) return "happy";
  if (v >= 0.4) return "satisfied";
  if (v > -0.4) return "neutral";
  if (v > -1.2) return "frustrated";
  return "angry";
}

function drawFunnel(rows) {
  const box = $("funnel");
  const dir = $("hDirection").value === "outbound" ? "outbound" : "inbound";
  $("funnelHint").textContent = `share of ${dir === "outbound" ? "outbound calls" : "inbound calls and chats"} that completed each step, in script order`;
  const steps = (state.checklists[dir] || []).filter((i) => i.kind === "step");
  const pool = rows.filter((r) => r.direction === dir);
  const W = Math.max(420, box.clientWidth || 520);
  const rowH = 26, L = 250, R = 56, T = 4;
  const H = T + steps.length * rowH + 4;
  const v = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Share of conversations completing each script step" });
  if (!pool.length) {
    const t = s("text", { x: 0, y: 20, class: "empty" });
    t.textContent = `No ${dir} conversations in this filter yet.`;
    v.append(t);
    box.replaceChildren(v);
    return;
  }
  const x = (p) => L + p * (W - L - R);
  v.append(s("line", { x1: L, x2: L, y1: T, y2: H - 4, class: "axis" }));
  steps.forEach((st, i) => {
    const appl = pool.filter((r) => r.steps && r.steps[st.id] && r.steps[st.id] !== "na");
    const done = appl.filter((r) => r.steps[st.id] === "done").length;
    const p = appl.length ? done / appl.length : 0;
    const yy = T + i * rowH;
    const label = s("text", { x: L - 10, y: yy + 17, "text-anchor": "end", class: "lbl" });
    label.textContent = st.label.length > 38 ? st.label.slice(0, 36) + "…" : st.label;
    v.append(label);
    const bw = Math.max(0, x(p) - L), bh = 14, r = Math.min(4, bw);
    if (bw > 0) v.append(s("path", { d: `M${L},${yy + 6}h${bw - r}q${r},0 ${r},${r}v${bh - 2 * r}q0,${r} -${r},${r}h-${bw - r}Z`, fill: cssVar("--brand") }));
    const val = s("text", { x: x(p) + 6, y: yy + 17, class: "lbl" });
    val.textContent = appl.length ? `${Math.round(p * 100)}%` : "–";
    v.append(val);
    const hit = s("rect", { x: 0, y: yy, width: W, height: rowH, class: "hit" });
    hoverable(hit, `<b>${esc(st.label)}</b>${done} of ${appl.length} conversations where it applied`);
    v.append(hit);
  });
  box.replaceChildren(v);
}

function drawTrend(rows) {
  const box = $("trend");
  const W = Math.max(420, box.clientWidth || 520);
  const panelH = 104, gap = 44, L = 64, R = 10, T = 26;
  const H = T + panelH * 2 + gap + 12;
  const v = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Script score and mood for each conversation" });
  const recent = rows.slice(-40);
  if (!recent.length) {
    const t = s("text", { x: 0, y: 20, class: "empty" });
    t.textContent = "Scorecards appear here as conversations end.";
    v.append(t);
    box.replaceChildren(v);
    return;
  }
  const n = Math.max(recent.length, 10);
  const px = (i) => L + 8 + (i / (n - 1)) * (W - L - R - 16);
  const panel = (top, title, lo, hi, ticks, value, color) => {
    const y = (val) => top + (1 - (val - lo) / (hi - lo)) * panelH;
    for (const [tv, tl] of ticks) {
      v.append(s("line", { x1: L, x2: W - R, y1: y(tv), y2: y(tv), class: tv === lo ? "axis" : "grid" }));
      const t = s("text", { x: L - 10, y: y(tv) + 4, "text-anchor": "end" });
      t.textContent = tl;
      v.append(t);
    }
    const ttl = s("text", { x: L, y: top - 14, class: "lbl" });
    ttl.textContent = title === "Script" ? "Script followed" : "Average mood";
    v.append(ttl);
    recent.forEach((r, i) => {
      const val = value(r);
      if (val == null) return;
      v.append(s("circle", { cx: px(i), cy: y(val), r: 5, class: "dot", fill: color(val) }));
      const hit = s("circle", { cx: px(i), cy: y(val), r: 12, class: "hit" });
      hoverable(hit, `<b>${esc(title)}: ${title === "Script" ? Math.round(val * 100) + "%" : cap(moodWordFor(val)) + " (" + val + ")"}</b>${esc(when(r.started_at))}, ${esc(r.outcome || "")}`);
      hit.addEventListener("click", () => openPast(r.id));
      v.append(hit);
    });
  };
  panel(T, "Script", 0, 1, [[0, "0%"], [0.5, "50%"], [1, "100%"]], (r) => r.sop.score, () => cssVar("--brand"));
  panel(T + panelH + gap, "Mood", -2, 2, [[-2, "Upset"], [0, "Neutral"], [2, "Happy"]], (r) => r.mood.avg, (val) => moodColor(val));
  box.replaceChildren(v);
}

async function openPast(id) {
  let x = state.sessions.get(id);
  if (!x) {
    try {
      const r = await fetch(`/api/session/${encodeURIComponent(id)}`);
      if (r.ok) x = await r.json();
    } catch {}
  }
  if (!x) return;
  state.sessions.set(x.id, x);
  state.selected = x.id;
  state.pinned = true;
  setView("live");
  schedule();
}

// -- view switching, theme ----------------------------------------------------------------------------
function setView(v) {
  state.view = v;
  for (const b of document.querySelectorAll(".tab")) b.setAttribute("aria-pressed", String(b.dataset.view === v));
  $("live").hidden = v !== "live";
  $("history").hidden = v !== "history";
  $("assistView").hidden = v !== "assist";
  const on = v === "assist";
  $("assistMode").setAttribute("aria-pressed", String(on));
  $("assistMode").textContent = on ? "Close Agent Assist Mode" : "Open Agent Assist Mode";
  if (v === "history") renderHistory();
  else schedule();
}
$("assistMode").addEventListener("click", () => setView(state.view === "assist" ? "live" : "assist"));
for (const b of document.querySelectorAll(".tab")) b.addEventListener("click", () => setView(b.dataset.view));
for (const id of ["hDirection", "hChannel"]) $(id).addEventListener("change", renderHistory);

function applyTheme(t) {
  if (t) document.documentElement.dataset.theme = t;
  iris.themeChanged();
  schedule();
  if (state.view === "history") renderHistory();
}
$("theme").addEventListener("click", () => {
  const dark = (document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark")) === "dark";
  const next = dark ? "light" : "dark";
  try { localStorage.setItem("theme", next); } catch {}
  applyTheme(next);
});
try {
  const saved = localStorage.getItem("theme");
  if (saved) applyTheme(saved);
} catch {}
addEventListener("resize", () => (state.view === "history" ? renderHistory() : schedule()));
setInterval(() => {
  // the clocks only: a full redraw every second would pull charts out from under the pointer
  if (state.view === "assist") {
    const c = $("avClock");
    const x = assistSession();
    if (c && x && x.live) c.textContent = clock(Date.now() / 1000 - x.started_at);
    return;
  }
  if (state.view !== "live") return;
  tickClocks();
  const x = state.sessions.get(state.selected);
  const d = $("durFact");
  if (x && x.live && d) d.textContent = clock(Date.now() / 1000 - x.started_at);
}, 1000);
connect();
