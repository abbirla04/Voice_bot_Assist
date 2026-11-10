"""Fixed lines the bot speaks without the LLM (greeting, hold, silence).

Fixed text is deliberate: these moments must be fast and must match the SOP
word for word, so they never depend on model output.
"""

from __future__ import annotations

from datetime import datetime

from call_state import Lang
from config import get_config

_CFG = get_config().clinic
CLINIC, CLINIC_HI = _CFG.name, _CFG.spoken_name_hi

LINES: dict[str, dict[Lang, str]] = {
    "hold_start": {
        "en": "Allow me to place your call on hold for a minute while I check this with the "
        "concerned department. I will not keep you on hold for more than a minute.",
        "hi": "कृपया एक मिनट के लिए लाइन पर बने रहिए, मैं संबंधित विभाग से जानकारी ले रही हूँ। "
        "मैं आपको एक मिनट से ज़्यादा होल्ड पर नहीं रखूँगी।",
    },
    "hold_checkin_1": {
        "en": "Thank you for holding. I am still speaking with the concerned department, "
        "please remain on the line.",
        "hi": "होल्ड करने के लिए धन्यवाद। मैं अभी भी संबंधित विभाग से बात कर रही हूँ, कृपया लाइन पर बने रहिए।",
    },
    "hold_checkin_2": {
        "en": "I am still trying to connect with the concerned department, it is taking a little "
        "more time. Please remain on the line, thank you for your patience.",
        "hi": "मैं अभी भी संबंधित विभाग से संपर्क कर रही हूँ, थोड़ा और समय लग रहा है। कृपया लाइन पर "
        "बने रहिए, आपके धैर्य के लिए धन्यवाद।",
    },
    "not_caught": {
        "en": "Sorry, I could not catch that clearly. Could you please say it once more?",
        "hi": "माफ़ कीजिए, मैं ठीक से सुन नहीं पाई। क्या आप एक बार फिर बता सकते हैं?",
    },
    "not_caught_2": {
        "en": "Sorry, the line is a little unclear. Could you please say it slowly, once more?",
        "hi": "माफ़ कीजिए, लाइन थोड़ी साफ़ नहीं है। क्या आप धीरे-धीरे एक बार फिर बता सकते हैं?",
    },
    "not_caught_3": {
        "en": "I am having trouble hearing you on this line. Please try once more, "
        "or I can arrange for our team to call you back.",
        "hi": "इस लाइन पर मुझे आपकी आवाज़ ठीक से नहीं आ रही है। कृपया एक बार और कोशिश कीजिए, "
        "या मैं हमारी टीम से आपको वापस कॉल करवा सकती हूँ।",
    },
    # Said only after 15 s of silence on both sides. Gentle on purpose: a sharp "Hello?" at
    # a caller who is thinking or searching for a document sounds impatient.
    "silence_1": {
        "en": "I'm still here on the line. Am I audible?",
        "hi": "मैं लाइन पर ही हूँ। क्या आपको मेरी आवाज़ आ रही है?",
    },
    "silence_2": {
        "en": "I'm sorry, I'm unable to hear you. Please take your time, I'm listening.",
        "hi": "माफ़ कीजिए, मुझे आपकी आवाज़ नहीं आ रही है। आप आराम से बताइए, मैं सुन रही हूँ।",
    },
    "silence_hangup": {
        "en": "I'm sorry, I'm still unable to hear you, so I'll end this call for now. "
        f"Please feel free to call us back anytime. Thank you for calling {CLINIC}.",
        "hi": "माफ़ कीजिए, मुझे अब भी आपकी आवाज़ नहीं आ रही है, इसलिए मैं अभी यह कॉल समाप्त कर रही हूँ। "
        f"आप हमें कभी भी दोबारा कॉल कर सकते हैं। {CLINIC_HI} को कॉल करने के लिए धन्यवाद।",
    },
    "max_duration": {
        "en": "I am sorry to keep you this long. So that you get the right help, "
        f"our team will call you back shortly. Thank you for calling {CLINIC}.",
        "hi": "आपको इतनी देर रोकने के लिए माफ़ कीजिए। सही मदद के लिए हमारी टीम आपको जल्दी ही "
        f"वापस कॉल करेगी। {CLINIC_HI} को कॉल करने के लिए धन्यवाद।",
    },
    "empty_turn": {
        "en": "Sorry, could you please say that once more?",
        "hi": "माफ़ कीजिए, क्या आप एक बार फिर से बता सकते हैं?",
    },
    "llm_retry": {
        "en": "I am sorry, there was a small issue on my side. Could you please repeat that?",
        "hi": "माफ़ कीजिए, मेरी तरफ़ थोड़ी दिक्कत आ गई। क्या आप एक बार फिर से बता सकते हैं?",
    },
    "technical_issue": {
        "en": "I am sorry, I am facing a technical issue on my side. Our team will call you back "
        f"shortly. Thank you for calling {CLINIC}.",
        "hi": "माफ़ कीजिए, मेरी तरफ़ कुछ तकनीकी समस्या आ रही है। हमारी टीम आपको जल्दी ही वापस "
        f"कॉल करेगी। {CLINIC_HI} को कॉल करने के लिए धन्यवाद।",
    },
}


def fixed_lines() -> list[str]:
    """Every fixed line in both languages: played from recordings (speech.use_recordings)."""
    return [t for variants in LINES.values() for t in variants.values()]


def line(key: str, lang: Lang) -> str:
    return LINES[key].get(lang) or LINES[key]["en"]


def time_greeting(now: datetime) -> str:
    if now.hour < 12:
        return "Good morning"
    if now.hour < 17:
        return "Good afternoon"
    return "Good evening"


def _who(persona: str, disclose_ai: bool, team: str) -> str:
    """SOP: "This is [Name] speaking. I am part of Dr. Arjun Mehta's team." """
    if team:
        if disclose_ai:
            return f"This is {persona} speaking, a virtual assistant and part of {team}."
        return f"This is {persona} speaking. I am part of {team}."
    return (
        f"This is {persona} speaking, a virtual assistant."
        if disclose_ai
        else (f"This is {persona} speaking.")
    )


def inbound_greeting(
    *, clinic: str, persona: str, now: datetime, disclose_ai: bool, team: str = ""
) -> str:
    """SOP opening: "Namaste/Good morning, Demo Eye & Retina. This is [Name] speaking.
    I am part of Dr. Arjun Mehta's team. How may I help you today?" Namaste, not good
    morning: the same words all day, so one recording is always ready to play.
    `now` is kept for callers and the outbound greeting."""
    return f"Namaste, {clinic}. {_who(persona, disclose_ai, team)} How may I help you today?"


def outbound_greeting(
    *,
    clinic: str,
    persona: str,
    now: datetime,
    disclose_ai: bool,
    lead_source: str | None,
    team: str = "",
) -> str:
    source = lead_source or "website"
    return (
        f"Namaste, {time_greeting(now).lower()}. This call is from {clinic}. "
        f"{_who(persona, disclose_ai, team)} We received your enquiry through our {source}, "
        "so how may I help you?"
    )


def chat_greeting(*, clinic: str, persona: str, disclose_ai: bool, team: str = "") -> str:
    """WhatsApp: the same welcome the chat prompt asks the model for (step 1), in writing."""
    who = f"I'm {persona}"
    if disclose_ai:
        who += f", a virtual assistant from {team}" if team else ", your virtual assistant"
    elif team:
        who += f", from {team}"
    return f"Namaste, welcome to {clinic}. {who}. How may I help you today?"
