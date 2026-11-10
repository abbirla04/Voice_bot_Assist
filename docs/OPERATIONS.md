# Operations guide: setup, phone, WhatsApp, dashboard, agent assist

A phone voice agent built on **LiveKit Agents 1.8** from the clinic's four call scripts:
opening (inbound), outbound, hold, and transfer. It takes the patient's details gently,
checks doctor availability, books, gives the scripted confirmation (doctor, location, time,
preparation, fee, dilation warning, SMS), and closes with the
scripted line. It follows the hold SOP (announce, music, check-ins, never more than a
minute, then callback) and the transfer SOP (warm transfer: brief the colleague first, connect only if
they pick up, otherwise callback). It speaks English, Hindi and Hinglish, says it is a virtual
assistant, never diagnoses, and treats eye emergencies as emergencies.

See **BENCHMARK.md** for the targets, current results and what failed during the build.

## Stack (and why)
| Piece | Choice | Why |
|---|---|---|
| Framework | LiveKit Agents 1.8 | best published barge-in + end-of-turn models, warm transfer, SIP, built-in test + simulation tools |
| Turn-taking | LiveKit Turn Detector v1 (audio) + adaptive interruption | ignores "hmm/haan ji", stops fast on real interruptions; v1-mini runs on-device as fallback |
| STT | Deepgram Nova-3 multilingual (via LiveKit Inference) -> ElevenLabs Scribe | Hindi-English code-switching; no separate Deepgram key |
| LLM | Claude Sonnet 5.5 (effort low) -> Claude Haiku 5.5 -> GPT-4.1 mini (via LiveKit) | automatic failover; Sonnet: 100% on the clinic evals vs Haiku 94%; Gemini dropped: slowest from India and kept failing |
| TTS | ElevenLabs Flash v2.5 -> Cartesia Sonic-3 (Hindi) | low latency, Indian voices; failover |
| Phone | Exotel Voicebot applet -> `src/exotel_bridge.py` (works on the trial account today); Exotel <-> LiveKit SIP trunk for production | your existing provider; with SIP, warm transfer happens inside LiveKit, so Exotel's lack of live-call redirect doesn't matter |

Every provider is a line in `.env.local` (`STT_CHAIN`, `LLM_CHAIN`, `TTS_CHAIN`), primary first.

## Setup (Windows / PowerShell)
```powershell
winget install astral-sh.uv            # once
cd eye-clinic-voice-agent
uv sync
copy .env.example .env.local           # then fill in the keys
uv run python src/agent.py preflight   # says READY or exactly which keys are missing
uv run pytest                          # 125 offline tests (uv run pytest -m live = LLM evals)
```

## Talk to it
```powershell
uv run python src/agent.py console --text       # type to Clara (needs only an LLM key)
uv run python src/agent.py console              # speak through your mic
uv run python src/agent.py dev                  # LiveKit Cloud: web console / phone
```
## Clinic data
`config/clinic.yaml` holds everything the bot may say about the clinic: locations with
addresses, landmarks and phone numbers, doctors with their specialities, departments,
emergency red flags, contacts and visit facts. The file in this repo is a **fictional demo
clinic** (Demo Eye and Retina Clinic, Dr. Arjun Mehta); replace it with your clinic's data.
When a clinic has not published **OPD timings** or **department phone numbers**:
- booking runs in `booking.mode: request`: Clara takes the patient, centre, doctor or speciality
  and preferred day/time, registers an **appointment request**, and says the team
  will call back to confirm the exact time. Every request goes to `logs/notifications.jsonl`
  (`appointment_request`) and to `NOTIFY_WEBHOOK_URL` if set; also `logs/bookings.jsonl`.
- fees: only what is in the file (`consultation_fee_inr`, `booking.fee_note`). The bot never
  invents an amount.
- transfers end in a callback until a department number is put in `sip_call_to`.
When the appointment system or the doctors' schedules are available, switch to
`booking.mode: slots` (give each doctor `days` + `slots`, each centre `consultation_fee_inr`).
A sample clinic with slot timings is kept for the tests in
`tests/fixtures/clinic_placeholder.yaml`; `dev`/`start` refuse any file with `placeholder_data: true`
unless `ALLOW_PLACEHOLDER_DATA=1`.

## Phone calls (Exotel), today: Voicebot applet + bridge
Works on the Exotel trial account (no SIP needed). Your `Voice_bot` flow already sends calls on
your ExoPhone numbers to `wss://<your-ngrok-domain>/exotel/stream`.
**One click, one window:** double-click `START-PHONE-TEST.bat`. It installs what's missing,
starts the voice bot, the bridge, the WhatsApp bot, the live dashboard and ngrok in the
background, places a fake Exotel call locally and through ngrok, prints READY with the numbers
to dial and opens the dashboard. That window then shows if any part stops; keys D (dashboard),
L (logs), Q (stop). Closing the window stops everything (a Windows job object ties the parts to
it); `STOP-PHONE-TEST.bat` does the same from outside.
Each run keeps its logs in `logs\runs\<date-time>\` (agent, bridge, ngrok, voice speed, fake
calls); `logs\calls\*.json` has one report per call. Latency lines to look for:
`reply N: TTFA ... = end-of-turn + LLM first token + TTS first audio` (agent.log) and
`TIMING reply N: TTFA ... at the phone line` (bridge.log).

Or by hand, three terminals in this folder:
```powershell
uv run python src/agent.py dev                                        # 1: the agent
uv run python src/exotel_bridge.py                                    # 2: bridge on :8765
ngrok http --url=<your-ngrok-domain> 8765         # 3: the public URL
uv run python bench/exotel_sim.py                                     # optional: fake call, no phone
uv run python bench/exotel_sim.py --url wss://<your-ngrok-domain>/exotel/stream
```
Then call the ExoPhone. Every call becomes a LiveKit room `exotel-<CallSid>`; the bridge log shows
the call, the agent joining, barge-in clears and why the call ended. Barge-in: the agent tells the
bridge, the bridge sends Exotel `clear`, the phone stops at once. Hang-up: the goodbye finishes
playing (Exotel `mark`), then the socket closes and Exotel ends the call.

Limits of the applet route: no warm transfer to staff and no outbound (Clara) calls; those need
SIP (below). Recommended: add `?sample-rate=16000` to the applet URL in the flow: better Hindi/English
recognition and ~100 ms faster replies (Exotel's smallest audio chunk is 200 ms at 8 kHz, 100 ms at 16 kHz).

One bot answers every number (your ExoPhone numbers): Deepgram speech recognition -> Claude Sonnet
-> ElevenLabs voice. Word-for-word scripts, recorded greeting, hold lines.

## Phone calls (Exotel), production: SIP trunk
Needs ExoTrunk (SIP) enabled on the Exotel account; the trial account doesn't have it.
1. LiveKit Cloud -> Telephony: copy your SIP URI; create an **inbound trunk** and a
   **dispatch rule** that sends calls to agent `eye-front-desk`.
2. Exotel: create an ExoTrunk pointing at that SIP URI (or ask hello@exotel.com to map your DID),
   following Exotel's "LiveKit SIP Trunking" guide.
3. For transfers and Clara's outbound calls, create a LiveKit **outbound trunk** with your
   ExoTrunk credentials and set `LIVEKIT_SIP_OUTBOUND_TRUNK` (+ `LIVEKIT_SIP_NUMBER`).
4. Put real department numbers in `config/clinic.yaml -> departments[].sip_call_to`.
5. Outbound call to a lead:
   ```powershell
   lk dispatch create --agent-name eye-front-desk --new-room `
     --metadata '{"phone_number":"+9198xxxxxxxx","lead_source":"WhatsApp","name":"Meena"}'
   ```
   Clara dials, detects answering machines (hangs up without a message), stays quiet until
   the person answers, then greets. Calls outside 9 AM-9 PM are refused.

## WhatsApp (same front desk, in writing)
`src/whatsapp_bot.py` runs the phone bot's agent (clinic data, scripts, appointment requests,
emergency handling, safety rules, Claude Sonnet) as a WhatsApp chat: written style, replies in
the script the patient types (English, Devanagari or Hinglish), remembers each number's chat
for 24 hours (`logs/whatsapp/chats/`), cost of every reply in `logs/whatsapp/costs.jsonl`.
- **Inactivity**: 120 s after the bot's last reply with no message back, the chat closes
  (a polite closing message if the bot had asked something; quietly after a goodbye or
  emergency advice). The patient's next message starts the whole flow again: greeting,
  details, booking. That holds even if the bot or laptop was off when the time ran out: any
  message more than 120 s after the last one starts over. `WHATSAPP_IDLE_CLOSE_S` changes the
  time (0 = never close).
- Try it without WhatsApp: `WHATSAPP-TEST-CHAT.bat`.
- Live: START-PHONE-TEST.bat starts it (port 8766, log `whatsapp.log`). Meta calls
  `https://<your-ngrok-domain>/whatsapp/webhook`; the Exotel bridge passes it on.
- `.env.local`: `WHATSAPP_TOKEN`, `WHATSAPP_PHONE_NUMBER_ID`, `WHATSAPP_APP_SECRET`, `WHATSAPP_VERIFY_TOKEN`.

## Live dashboard
`http://127.0.0.1:8767`: START-PHONE-TEST starts it and opens it (D in its window opens it again);
`DASHBOARD.bat` opens it again. `DASHBOARD-DEMO.bat` shows it with sample calls and chats (port 8768).
- **Live**: every call and WhatsApp chat as it happens. A 3D iris (Three.js) shows the patient's
  mood: blue calm or happy, grey neutral, red upset; it pulses with their voice. Mood per patient
  turn comes from Claude Haiku reading the words (any language) plus the bridge's loudness of that
  utterance against the caller's own normal (a raised voice shows as "+7 dB").
- **Call script**: Clara's inbound or Clara's outbound checklist ticks as the bot follows it
  (`src/sop_checklists.py`: fixed lines by pattern, details/requests by tool call, paraphrases,
  Hindi and the "never" rules by Haiku). Items that never came up are "not needed".
- **Reply speed** per turn against the 0.9 s target, **alerts** (upset patient, emergency, rule broken).
- **History**: a scorecard for every conversation (`logs/dashboard/sessions/`): script %, mood,
  outcome, length, speed, cost; where conversations stop following the script; trends.
- Only on this PC (127.0.0.1, never behind ngrok): it shows patient conversations. Phone numbers
  are masked. The bot never waits for it: if it isn't running, events are dropped. About ₹0.1-0.2
  of Haiku per call. `DASHBOARD=0` in `.env.local` turns the feed off.

## Agent assist: a person answers, Claude suggests
For calls a person at the clinic takes. Exotel flow **Agent_assist** on your ExoPhone: a Stream applet sends a copy of the call audio to the bridge at
`wss://<ngrok>/exotel/assist` once the agent picks up, and a Connect applet rings the agent.
Nothing we do can be heard on the call.
- `src/agent_assist.py`: the audio (caller and agent mixed) -> Deepgram Nova-3 through LiveKit
  Inference, Hindi and English, with speakers told apart (diarization), one line per sentence.
- `src/assist_brain.py`: Claude (Sonnet, effort low; `ASSIST_MODEL=claude-haiku-5-5` is faster)
  reads the transcript with the clinic facts and the inbound script, and writes: the suggested
  reply in the patient's language, why, up to 2 facts the agent needs, the booking details
  heard, an alert for emergencies or an upset patient, and who is the agent. The words show
  about 2.5 s after the patient stops (streamed); the rest follows. After the call: a short note,
  the outcome and the follow-up.
- Dashboard: the call shows as **Assist**, with the suggestion card on top (it stays in view),
  the agent's script checklist (ticked by Claude from their own words), patient mood and alerts.
  `DASHBOARD-DEMO.bat` includes a sample assisted call.
- Same safety rules as Clara: no diagnosis, no medicine advice, no invented fees or timings.
  Tell callers the call is AI-assisted before going live.

## Tone: the bot must never sound rude or loud
Fixed after test calls sounded sharp (18 Oct). Each layer is tested:
- **Voice**: ElevenLabs runs with calm settings (stability 0.75, style 0, speed 0.95, speaker
  boost off; `ELEVEN_*` in `.env.example`). The voice's own defaults let short lines come out sharp.
- **Loudness**: the bridge turns the bot down 6 dB and peak-limits it at -4 dBFS (calls measured
  -15 dBFS with peaks at full scale, ~10 dB hotter than a normal phone voice).
- **Words**: polite requests only ("May I know...", "क्या आप ... बता सकते हैं?", आप forms),
  no "Hey", no Sir/Ma'am guessing; "!" is never spoken. The silence check is gentle
  ("I'm still here on the line. Am I audible?") and is never repeated back to back.
- **Never pushy**: a question is not asked twice in a row, and the bot cannot hang up while the
  caller is about to answer.
- **Evals**: every scenario has a deterministic polite-wording check and a judged tone check;
  S18 (irritated caller) and S19 (terse Hindi caller) stress it.

## Going live checklist
- [ ] The clinic's real centres, doctors, contacts and insurers in `config/clinic.yaml` (the repo ships a fictional demo clinic).
- [ ] From the clinic: OPD timings or appointment-system access (then `booking.mode: slots`), fees per centre, department numbers for transfers, a 24x7 emergency number for `emergency.contact_line`.
- [ ] A clinic doctor approves `emergency.advice` and the red-flag list.
- [ ] Swap `MockScheduler` (`src/scheduling.py`) for the hospital's appointment system (3 methods).
- [ ] `NOTIFY_WEBHOOK_URL` -> n8n/backend that sends the DLT-approved Exotel SMS and queues callbacks, do-not-call, emergency alerts.
- [ ] Pick Clara's voice (`ELEVEN_VOICE_ID`) and listen to `bench/results/audio/*.wav`.
- [ ] All benchmark targets green (../BENCHMARK.md), then a 1-week shadow pilot.
- [ ] Data protection: call reports in `logs/calls/` hold transcripts (phone numbers masked); set retention.

## Layout
```
config/clinic.yaml      everything the bot may say about the clinic (validated on start)
src/agent.py            entrypoint: session, turn-taking, noise cancellation, outbound dialing
src/front_desk.py       Clara agent + tools (intake, availability, booking, callback,
                        do-not-call, emergency flag, warm transfer, end call)
src/prompts.py          system prompt built from the four scripts
src/lines.py            fixed EN/HI lines (greeting, hold, silence, failure) - no LLM latency
src/hold.py             hold SOP around slow lookups
src/guards.py           silence, max call length, provider-outage failsafe, clean hang-up
src/providers.py        STT/LLM/TTS fallback chains from env
src/scheduling.py       scheduling backend interface + mock
src/telemetry.py        per-call latency + audit report (logs/calls/*.json)
src/notify.py           SMS / callback / alert webhook
src/exotel_bridge.py    Exotel Voicebot applet <-> LiveKit room (calls without SIP)
src/exotel_signals.py   agent side of it: barge-in -> Exotel `clear`
src/speech.py           fixed lines (greeting, silence prompts) spoken word for word
bench/                  benchmark scripts (see BENCHMARK.md)
tests/                  offline tests + live LLM-judged evals
scenarios.yaml          LiveKit Cloud simulations (generated from bench/scenarios.py)
```

## Known limits
- The live briefing with a colleague during a warm transfer has no time cap (the ring time does: 25 s).
- Spoken-out phone numbers ("nine eight seven...") are not masked in transcripts; digit forms are.
- Applet route: transfers fall back to a callback (no SIP to dial staff); keypad digits are ignored.
- Adaptive interruption and Turn Detector v1 run on LiveKit Cloud; self-hosted falls back to VAD interruption and v1-mini.
