# Voicebot benchmark (eye-clinic front desk)

There is no single industry benchmark that covers a phone voicebot end to end, so this one
combines the measures that matter to a caller, with targets taken from published production
guidance (Hamming, LiveKit eot-bench, tau-Voice, Full-Duplex-Bench) and tightened where a
hospital line needs it. Every number below is produced by a script in `bench/`.

## 1. Targets

### Speed (what the caller feels)
| Metric | What it measures | Target p50 | Target p95 | Script |
|---|---|---|---|---|
| **TTFA** | caller stops speaking -> first bot audio | <= 900 ms | <= 1400 ms | `bench/voice_call.py`, live calls |
| **TTFT** | LLM time to first token (real prompt + tools) | <= 450 ms | <= 900 ms | `bench/latency.py`, `bench/evaluate.py` |
| TTS TTFB | text sent -> first audio byte | <= 250 ms | <= 450 ms | `bench/latency.py` |
| STT final | speech end -> final transcript | <= 300 ms | <= 600 ms | `bench/latency.py` |
| End-of-turn | speech end -> "caller is done" decision | <= 350 ms | <= 700 ms | live call reports |
| Greeting | call connected -> first word | <= 1500 ms | | `bench/voice_call.py` |
| Turn detector / VAD compute | on-device model time per decision | <= 50 ms / 2 ms | <= 100 ms / 5 ms | `bench/latency.py --only local` |

WebRTC numbers exclude the phone network: a PSTN call through Exotel adds roughly 150-300 ms.
On the Voicebot-applet route (trial account, `src/exotel_bridge.py`) add up to one Exotel audio
chunk on top: 200 ms at 8 kHz, 100 ms at 16 kHz (`?sample-rate=16000` on the applet URL).

### Barge-in and turn-taking
| Metric | Target | Script |
|---|---|---|
| Stop latency: caller talks over bot -> bot audio stops | p95 <= 400 ms | `bench/voice_call.py` |
| Missed interruptions (bot keeps talking > 2 s) | <= 5% | `bench/voice_call.py` |
| Backchannel false stops ("hmm", "haan ji" stops the bot > 1 s) | <= 10% | `bench/voice_call.py` |
| Regression gate on real call recordings | no regressions | `uvx hotato run --stereo <call.wav>` |

### Doing the job (SOP + safety, LLM-judged, 3 runs per scenario)
| Metric | Target | Script |
|---|---|---|
| All 13 scenarios (booking, hold, transfer, Hindi, outbound...) | >= 95% pass | `bench/evaluate.py` |
| Safety scenarios (emergency, no medicine advice, AI disclosure, no invented fees/doctors, off-topic) | **100%** | `bench/evaluate.py` |
| STT accuracy on phone-quality audio (8 kHz, G.711, line noise) | WER <= 15% EN, CER <= 20% HI | `bench/latency.py` |
| Replies are voice-friendly (no lists/markdown, <= 60 words) | 100% | `bench/evaluate.py` |

### Never breaks (deterministic, offline)
| Behaviour | Target | Where |
|---|---|---|
| Primary LLM down or slow -> backup answers | 100% | `tests/test_flows.py` |
| Every LLM down -> apology, then human callback + clean hang-up (no dead air) | 100% | `tests/test_guards.py` |
| Slow appointment system -> hold SOP, never > 1 minute, callback | 100% | `tests/test_flows.py` |
| Transfer: no answer / closed / not configured -> back to caller + callback | 100% | `tests/test_flows.py` |
| Silent caller -> 2 prompts -> polite hang-up; never during hold/transfer | 100% | `tests/test_guards.py` |
| Max call length -> callback + polite close | 100% | `tests/test_guards.py` |
| Bad input (age 250, wrong phone, unknown doctor, past date) -> asks again | 100% | `tests/test_flows.py` |
| Placeholder clinic data never reaches real callers | 100% | `tests/test_review_fixes.py` |

## 2. Results (18 Oct 2026)

### Live: SOP + safety evals with real models
Bot: Claude Haiku 5.5. Judge: Claude Sonnet 5.5. Run from the cloud workspace, which can only reach Anthropic.

| Round | What changed | Scenario pass | Safety pass |
|---|---|---|---|
| 1 (39 calls) | first live run | 49% | 72% |
| 2 (39) | token cap, silent-turn guard, judge fixes | 67% | 83% |
| 3 (39) | hold lines out of LLM context, scripted confirmation in code | 74% | 78% |
| 4 (39) | Hindi times pre-computed, routing fix, rule fixes | 79% | 83% |
| 5 (39) | medical-question rule, judge saw full tool results | **100%** | **100%** |
| 6 (65) | 5 runs per scenario | 92% | 100% |
| 7 (65) | bracket filter, names verbatim, no re-asking | **98%** | **100%** |

The one remaining miss (1 of 65) is cosmetic: Hindi "धन्यवाद" said twice.

### Live: speed
| Metric | p50 | p95 | Target | Verdict |
|---|---|---|---|---|
| Claude Haiku 5.5 TTFT, direct, warm cache | 548 ms | 1073 ms | 450 / 900 | FAIL |
| Claude Haiku 5.5 TTFT, in conversation (115 turns) | 1106 ms | 2006 ms | 450 / 900 | FAIL |
| Turn detector v1-mini (on-device) | 33 ms | 44 ms | 50 / 100 | PASS |
| VAD | 0.02 ms | 0.03 ms | 2 / 5 | PASS |

Claude was measured from a US cloud box, so re-measure from India. Tool turns are slow:
Claude emits about 600 output tokens before a tool call fires.

### Still blocked: run from your PC
These need network access the cloud workspace doesn't have: Gemini (primary LLM), ElevenLabs
(voice, backup STT), LiveKit (Deepgram STT, TTFA, barge-in, backchannel, greeting).

### What failed while building, and is now fixed
Found by starting the agent:
1. **Every call would have crashed before the greeting**: the noise-cancellation plugin was imported inside the call ("Plugins must be registered on the main thread").
2. Importing provider plugins inside the call stalled the event loop (~80 ms+ freeze on the first call).
3. With every LLM down the caller got **dead air**; now: apology, then human callback + clean hang-up.
4. Console caller-ID handling crashed the end-of-call report.

Found by the test suite:

5. Hold lines were spoken in English to Hindi callers in some paths.
6. The benchmark counted "bot finished its answer" as "bot stopped for the interruption" (would have hidden missed barge-ins), and scored the Hindi full stop as an STT error.

Found by an independent review against the four scripts:

7. **Emergency alert could be skipped** (transfer ran first and ended the turn), and with no emergency line configured the bot offered a *callback* for a chemical splash. Now staff are alerted first, and the bot always insists on going to emergency.
8. **Outbound double greeting**: the callee's "Hello?" got an LLM reply before Clara's scripted greeting.
9. Closing lines didn't match the scripts (booking: "Thank you for considering..."; otherwise "Thank you for calling...").
10. A caller talking over the goodbye could leave the call open with all safety guards dead.
11. A slow booking was cancelled while the hospital system might still save it (double-booking risk).
12. SMS was promised even when it wasn't sent; SMS lacked the centre address ("with landmark").
13. "Do not call me" was filed as a callback request.
14. Picking a specialty from symptoms is an implied diagnosis; now it searches by location unless the caller asks.
15. Full phone numbers in local logs; outbound lead form text could inject instructions into the prompt.
16. Benchmark realism: agent hold-music track mixed into the voice timing, greeting measured too early, TTS trailing silence flattered TTFA, LLM test ran without the real tool schemas, STT test used clean audio.

Found checking for your Windows 11 setup:

17. It would not have run on Windows at all: Linux-only date formatting (`%-d`), no timezone database (`tzdata`), and Hindi text written with the default cp1252 encoding.

Found by the live evals with real models:

18. **Claude rejected every request**: an optional enum in a tool schema is invalid for Anthropic (400 on every turn).
19. **Claude Haiku 5.5 rejects `temperature`** (400). It is now sent only when set.
20. **Silent turns (dead air)**: a 400-token cap cut off tool calls, so nothing was said. Fixed with a higher cap plus a guard that speaks "Sorry, could you say that once more?" whenever a model turn comes back empty. Gemini 2.5 has a known empty-response bug with the same symptom.
21. **Every hold crashed on Claude**: hold announcements were added to the LLM context as a trailing assistant turn, which Claude refuses.
22. The silent-turn guard fired after the goodbye. Now skipped once the call is ending.
23. **Wrong time to a patient**: the model turned 10:30 into "सवा दस" (10:15). Hindi day and time words are now pre-computed in code.
24. **The model "thought aloud" into speech** ("(The caller said …) Correction: …"). Bracketed text is now filtered before it is spoken.
25. It changed a caller's name (Rohit became राहुल). Names are now used exactly as given.
26. Config routed LASIK cost questions to billing; surgery costs belong to surgery counselling.
27. The booking confirmation ran to 122-164 words and dropped SOP items. It is now built in code from the script; the model only translates it.
28. It asked medical questions, didn't offer a booking when declining medical advice, asked two questions per reply, re-asked details already given, switched to English mid-call, and didn't apologise when a department was closed. All fixed with prompt or tool changes.
29. Benchmark bugs: the judge was truncated at 400 tokens, broke on quotes in its reasons, saw only 300 characters of tool results, and the eval transcript was missing the scripted greeting.

Found building the Exotel bridge (offline tests in `tests/test_exotel_bridge.py`):

30. **The caller could stop hearing the bot for the rest of the call**: LiveKit's audio mixer
    times out a quiet track by cancelling its generator, which ends it for good. A bot that took
    a moment to speak was never heard. Tracks now go through a cancel-safe queue.
31. A silent hold-music track would have slowed the bot's audio below real time at 16 kHz, so it
    lagged further behind with every sentence. The mixer now skips a quiet track after half a chunk.
32. Caller audio that arrived while the room was connecting would have been queued and played
    late, delaying every turn of the call by ~1-2 s. It is now dropped, and the backlog is capped.
33. Hanging up straight after the goodbye could clip its last words, on SIP calls too. The agent
    now waits 0.6 s, and the bridge waits for Exotel to confirm the goodbye played.
34. If the agent isn't running, a caller would have heard dead air forever. The bridge now ends
    the call after 15 s and logs exactly what to start.


## Model choice on real clinic data (18 Oct 2026)

Same 17 scenarios, 34-51 runs per setup; speed = first word, one call at a time, warm cache.

| Setup | Plain reply | Reply after saving details | Evals | Safety | Cost per booking call |
|---|---|---|---|---|---|
| Haiku 5.5 (before) | 1.42 s | 3.08 s | 94% | 100% | ~Rs 0.2 |
| Haiku, thinking off | 0.74 s | 1.41 s | 79% | 75% | ~Rs 0.2 |
| Sonnet 5.5, default | 1.87 s | 2.95 s | 100% | 100% | ~Rs 4.7 |
| **Sonnet 5.5, effort low + one-step saving (now)** | **~1.0 s** | **0.8-1.5 s** | **100%** | **100%** | **~Rs 1.9** |

Optimisations behind the last row: detail-saving turns speak and save in one LLM request
(no second round trip); English appointment confirmations are spoken by code; the system
prompt is the same all day (clock time moved to the per-turn note) so the prompt cache is
shared across calls; the cache is warmed while the greeting plays. 10 -> 7 LLM requests per
booking call.

## 3. How to run everything

```powershell
uv run pytest                                  # offline: 125 tests, ~45 s
uv run python bench/latency.py                 # component speed + STT accuracy (keys)
uv run python bench/evaluate.py --runs 3       # 13 SOP/safety scenarios, LLM-judged (keys)
uv run python bench/evaluate.py --llm google/gemini-3.5-flash --llm anthropic/claude-haiku-5-5   # compare LLMs
# voice, end to end (two terminals):
uv run python src/agent.py dev
uv run python bench/voice_call.py --trials 6
# LiveKit Cloud simulations with improvising simulated callers:
lk agent simulate text --scenarios scenarios.yaml
# the phone route (bridge must be running, see README):
uv run python bench/exotel_sim.py              # fake Exotel call: greeting latency + chunk checks
# after real calls:
uv run python bench/summarize_calls.py
```
