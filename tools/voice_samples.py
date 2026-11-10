"""Hear calmer voices for Clara before switching: VOICE-SAMPLES.bat.

Finds soft, calm Indian female voices in ElevenLabs' voice library, has each one say the
same two lines the bot really says (English and Hindi, same model and calm settings as on
calls), and saves them to voice-samples\\ next to the current voice, with a list of names,
voice IDs and how loud each one is. Pick by number; the voice ID goes into ELEVEN_VOICE_ID.

Uses ELEVEN_API_KEY from .env.local. About 3,000 characters of ElevenLabs credit.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "voice-samples"
API = "https://api.elevenlabs.io/v1"
MODEL = "eleven_flash_v2_5"  # the model calls use
RATE = 22050
# what to look for, and what does not suit a receptionist on the phone
SEARCHES = ["calm", "soft", "helpline", "customer support", "receptionist", "warm", "gentle"]
SKIP = re.compile(
    r"meditat|asmr|whisper|narrat|story|audiobook|influencer|seductive|sexy|news|"
    r"podcast|kid|child|old|elderly|dramatic|energetic|excited",
    re.IGNORECASE,
)
HOW_MANY = 7  # plus the current voice
LINES = {
    "en": "Hello, welcome to Demo Eye and Retina Clinic. I'm Clara, your virtual assistant. "
    "How may I help you today? ... Thank you, Saurab ji. May I know the patient's age, please?",
    "hi": "जी ज़रूर, मुझे आपकी मदद करके ख़ुशी होगी। क्या आप पहली बार डेमो आई एंड रेटिना क्लिनिक आ रहे हैं, या पहले भी आ चुके हैं?",
}
CALM = {  # the bot's call settings (src/providers.py ELEVEN_CALM)
    "stability": 0.75,
    "similarity_boost": 0.75,
    "style": 0.0,
    "speed": 0.95,
    "use_speaker_boost": False,
}


def _env(path: Path) -> dict[str, str]:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*([A-Z0-9_]+)\s*=\s*(.*)$", line)
            if m:
                out[m.group(1)] = m.group(2).split(" #")[0].strip().strip('"').strip("'")
    return out


def _get(url: str, key: str) -> dict:
    req = urllib.request.Request(url, headers={"xi-api-key": key})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def find_voices(key: str) -> list[dict]:
    """Library voices that sound like a calm Indian receptionist, most used first."""
    found: dict[str, dict] = {}
    for term in SEARCHES:
        q = {"gender": "female", "language": "hi", "search": term, "page_size": "30"}
        try:
            data = _get(f"{API}/shared-voices?{urllib.parse.urlencode(q)}", key)
        except urllib.error.HTTPError as e:
            print(f"   search '{term}' failed: HTTP {e.code}")
            continue
        for v in data.get("voices", []):
            text = " ".join(
                str(v.get(k) or "") for k in ("name", "description", "descriptive", "use_case")
            )
            if v.get("voice_id") and not SKIP.search(text):
                found.setdefault(v["voice_id"], v)
    ranked = sorted(found.values(), key=lambda v: v.get("cloned_by_count") or 0, reverse=True)
    return ranked[:HOW_MANY]


def synth(key: str, voice_id: str, text: str) -> bytes:
    url = f"{API}/text-to-speech/{voice_id}?output_format=pcm_{RATE}"
    body = json.dumps(
        {"text": text, "model_id": MODEL, "voice_settings": CALM}, ensure_ascii=False
    ).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"xi-api-key": key, "Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def loudness(pcm: bytes) -> tuple[float, float]:
    """Average loudness of the speech (dBFS, silence left out) and the loudest peak."""
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if not a.size:
        return -90.0, -90.0
    win = max(1, RATE // 50)  # 20 ms
    frames = a[: len(a) // win * win].reshape(-1, win)
    rms = np.sqrt((frames**2).mean(axis=1))
    speech = rms[rms > 300]  # leave out the pauses
    level = float(np.sqrt((speech**2).mean())) if speech.size else 1.0
    db = lambda x: round(20 * float(np.log10(max(x, 1.0) / 32768)), 1)  # noqa: E731
    return db(level), db(float(np.abs(a).max()))


def save_wav(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)


def _safe(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]+', "", name).strip()[:60] or "voice"


def main() -> int:
    env = {**_env(ROOT / ".env.local"), **os.environ}
    key = env.get("ELEVEN_API_KEY") or env.get("ELEVENLABS_API_KEY")
    if not key:
        print("ELEVEN_API_KEY is missing from .env.local")
        return 1
    current = env.get("ELEVEN_VOICE_ID", "").strip()
    OUT.mkdir(exist_ok=True)

    print("Looking for calm Indian female voices in the ElevenLabs library...")
    voices = find_voices(key)
    if current:
        voices = [
            {"voice_id": current, "name": "CURRENT VOICE", "description": "what calls use now"}
        ] + [v for v in voices if v["voice_id"] != current]
    if len(voices) <= 1:
        print("No library voices found (check the key, or the network).")

    rows = []
    for n, v in enumerate(voices):
        label = f"{n:02d} {_safe(v.get('name', ''))}"
        about = v.get("description") or v.get("descriptive") or ""
        print(f"\n{label}  ({v['voice_id']})")
        levels = []
        for lang, text in LINES.items():
            try:
                pcm = synth(key, v["voice_id"], text)
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:160]
                print(f"   {lang}: could not make a sample (HTTP {e.code}) {detail}")
                continue
            save_wav(OUT / f"{label} - {lang}.wav", pcm)
            levels.append(loudness(pcm))
            print(f"   {lang}: saved, loudness {levels[-1][0]} dBFS, peak {levels[-1][1]} dBFS")
        if levels:
            avg = round(sum(x[0] for x in levels) / len(levels), 1)
            peak = max(x[1] for x in levels)
            rows.append((label, v["voice_id"], avg, peak, str(about)[:90]))
        elif v.get("preview_url"):
            # the library would not voice our lines with it: its own short demo instead
            try:
                with urllib.request.urlopen(v["preview_url"], timeout=30) as r:
                    (OUT / f"{label} - demo.mp3").write_bytes(r.read())
                print("   saved the voice's own demo instead")
                rows.append((label, v["voice_id"], None, None, str(about)[:90]))
            except OSError as e:
                print(f"   no demo either: {e}")

    with open(OUT / "voices.txt", "w", encoding="utf-8") as f:
        f.write("Pick a number and tell Claude, or put its voice ID in .env.local:\n")
        f.write("   ELEVEN_VOICE_ID=<voice ID>\n")
        f.write("Loudness: average while speaking. Lower (more negative) = softer.\n\n")
        for label, vid, avg, peak, about in rows:
            loud = "demo only" if avg is None else f"loudness {avg} dBFS, peak {peak} dBFS"
            f.write(f"{label}\n   voice ID {vid}\n   {loud}\n")
            if about:
                f.write(f"   {about}\n")
            f.write("\n")
    print(f"\nDone: {len(rows)} voices in {OUT}  (list with voice IDs: voices.txt)")
    if sys.platform == "win32":
        # not os.startfile(OUT): next to the folder voice-samples sits VOICE-SAMPLES.bat, and
        # Windows runs that file instead of opening the folder, which ran this tool again forever
        subprocess.Popen(["explorer.exe", str(OUT)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
