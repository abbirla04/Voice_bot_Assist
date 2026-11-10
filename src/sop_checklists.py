"""The call scripts as checklists the dashboard ticks while a call or chat runs.

Each item is checked in up to three ways, cheapest first:
  say    the bot's own words match a pattern (scripted lines are fixed text: exact and instant)
  tool   a tool call proves it (details saved, request registered, callback booked...)
  judge  Claude Haiku reads the reply in context (paraphrases, Hindi, "verified the person")
A "never" item is a rule: it is fine until a reply breaks it, then it turns red.

kinds:  step   in the order the script asks for (the History funnel uses this order)
        when   only if the situation came up (a bad time, a transfer...); n/a otherwise
        never  a rule for the whole call
needs:  "booking": n/a when the conversation never got to an appointment
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Item:
    id: str
    label: str
    kind: str = "step"  # step | when | never
    say: tuple[str, ...] = ()  # regexes on the bot's reply (case-insensitive)
    tool: str | None = None  # tool name(s) that complete it, "a|b"
    tool_arg: str | None = None  # ...only when this argument is set
    tool_ok: str | None = None  # ...and its output starts with this, "A|B"
    judge: str | None = None  # what Haiku should check, in plain words
    needs: str | None = None

    def says(self, text: str) -> bool:
        return any(re.search(p, text, re.IGNORECASE) for p in self.say)

    def done_by(self, name: str, args: dict, output: str) -> bool:
        if not self.tool or name not in self.tool.split("|"):
            return False
        if self.tool_arg and args.get(self.tool_arg) in (None, "", []):
            return False
        return not self.tool_ok or output.startswith(tuple(self.tool_ok.split("|")))


@dataclass(frozen=True)
class Checklist:
    name: str
    items: tuple[Item, ...] = field(default_factory=tuple)


# Demo Clinic Front Desk SOP, "Key rules at a glance": details to capture are name, age,
# contact number, city/location and eye complaint.
_DETAILS = (
    Item(
        "name",
        "Takes the patient's name",
        tool="update_patient_details",
        tool_arg="name",
    ),
    Item(
        "age",
        "Takes the patient's age",
        tool="update_patient_details",
        tool_arg="age",
        needs="booking",
    ),
    Item(
        "city",
        "Takes the city or area the patient is coming from",
        tool="update_patient_details",
        tool_arg="city",
        judge="asked which city or area the patient lives in or is coming from",
    ),
    Item(
        "complaint",
        "Records the eye complaint",
        tool="update_patient_details",
        tool_arg="concern",
        judge="understood and noted the patient's eye complaint",
    ),
    Item(
        "contact",
        "Takes a contact number (the calling number is fine)",
        tool="update_patient_details|request_appointment|book_appointment|schedule_callback",
        judge="took or confirmed a contact number for the patient",
    ),
    Item(
        "offer_booking",
        "Asks: “Can I book an appointment with the doctor for you?”",
        say=(r"book an appointment with the doctor", r"book a consultation with our doctor"),
        judge="offered to book an appointment with the doctor",
    ),
    Item(
        "registered",
        "Registers the appointment request",
        tool="request_appointment|book_appointment",
        tool_ok="REQUEST REGISTERED|BOOKED",
        needs="booking",
    ),
    Item(
        "confirmation",
        "Repeats the details back, with preparation: reports, arrive early, dilation",
        say=(r"previous eye reports", r"fifteen minutes early"),
        judge="repeated the appointment details back and gave the preparation instructions: "
        "bring previous reports, come a little early, and the dilation advice (bring someone along)",
        needs="booking",
    ),
)

_ANYTHING_ELSE = Item(
    "anything_else",
    "Asks if there is anything else",
    say=(r"anything else", r"और कुछ", r"aur kuch", r"कुछ और"),
    judge="asked whether there is anything else it can help with",
)

_RULES = (
    Item(
        "no_diagnosis",
        "Never gives medical advice or diagnoses",
        kind="never",
        judge="gave medical advice, diagnosed, named a likely condition, or suggested a medicine",
    ),
    Item(
        "price_rule",
        "Never quotes a price before understanding the problem; no surgery or test prices",
        kind="never",
        judge="quoted any price before understanding the eye problem, stated a price other than "
        "the consultation charge, or stated a doctor's OPD day or time",
    ),
    Item(
        "no_human_claim",
        "Never claims to be a human",
        kind="never",
        judge="claimed or implied that it is a human",
    ),
    Item(
        "tone",
        "Clarity, control, care: reassuring, never dismissive",
        kind="never",
        judge="sounded dismissive, impatient or cold, or left the caller without a clear next step",
    ),
)

INBOUND = Checklist(
    "Inbound call (Demo Clinic front desk SOP)",
    (
        Item(
            "greeting",
            "Greets: “Namaste, Demo Eye and Retina Clinic”",
            say=(r"namaste,? demo", r"welcome to demo", r"डेमो"),
        ),
        Item(
            "intro",
            "Introduces herself as Clara, part of Dr. Arjun Mehta's team",
            say=(r"this is clara", r"i'?m clara", r"part of dr\.? arjun mehta'?s team", r"क्लारा"),
        ),
        Item(
            "how_help",
            "Asks how she may help",
            say=(r"how may i help", r"how can i help", r"कैसे मदद", r"kaise madad"),
        ),
        Item(
            "request_type",
            "Understands the request: appointment, query, emergency or billing",
            judge="understood what the caller needed (an appointment, a general query, an "
            "emergency or billing) before moving on",
        ),
        *_DETAILS,
        _ANYTHING_ELSE,
        Item(
            "closing",
            "Closes: repeats the details and “Thank you for calling”",
            say=(r"thank you for calling demo", r"धन्यवाद"),
            judge="repeated what was noted and closed with thank you for calling",
        ),
        Item(
            "emergency",
            "Emergency: gives the urgent advice and the clinic's number",
            kind="when",
            tool="flag_emergency",
        ),
        Item(
            "billing",
            "Billing: passes it to the billing desk, takes no payment details",
            kind="when",
            tool="transfer_to_department",
            tool_arg="department_id",
        ),
        Item(
            "callback",
            "Takes a message and assures a callback when someone is unavailable",
            kind="when",
            tool="schedule_callback",
        ),
        *_RULES,
    ),
)

OUTBOUND = Checklist(
    "Outbound call (Demo Clinic)",
    (
        Item(
            "greeting",
            "Greets with Namaste and the time of day",
            say=(r"namaste,? good (morning|afternoon|evening)",),
        ),
        Item(
            "call_from",
            "Says the call is from Demo Eye and Retina Clinic",
            say=(r"this call is from demo",),
        ),
        Item(
            "intro",
            "Introduces herself as Clara, part of Dr. Arjun Mehta's team",
            say=(r"this is clara", r"clara.{0,40}virtual assistant"),
        ),
        Item(
            "source",
            "Names where the enquiry came from",
            say=(r"received your enquiry through", r"enquiry (through|on|via)"),
        ),
        Item("how_help", "Asks how she may help", say=(r"how may i help", r"how can i help")),
        Item(
            "verify",
            "Confirms she's speaking with the person who enquired",
            judge="checked that it is speaking with the person who made the enquiry (or a family "
            "member handling it) before discussing medical details",
        ),
        *_DETAILS,
        _ANYTHING_ELSE,
        Item(
            "closing",
            "Closes with the outbound closing line",
            say=(r"you were speaking with clara", r"thank you for your time"),
            judge="thanked them for their time and said they were speaking with Clara",
        ),
        Item("bad_time", "Bad time: offers to call back", kind="when", tool="schedule_callback"),
        Item(
            "do_not_call",
            "Asked not to call again: apologises and notes it",
            kind="when",
            tool="mark_do_not_call",
        ),
        *_RULES,
    ),
)


# A person answering with agent assist: the same SOP, in their own words, so every step is
# checked by Claude (no tool calls to tick it) plus a few fixed phrases.
AGENT = Checklist(
    "Inbound call (Demo Clinic front desk SOP, agent)",
    (
        Item(
            "greeting",
            "Greets with Demo Clinic and Dr. Arjun Mehta's team",
            say=(r"demo",),
            judge="greeted the caller, named Demo Clinic and said they are part of Dr. Arjun "
            "Mehta's team",
        ),
        Item(
            "how_help",
            "Asks how they may help",
            say=(r"how may i help", r"how can i help", r"कैसे मदद", r"kaise madad"),
            judge="asked how they can help",
        ),
        Item("name", "Takes the patient's name", judge="asked for the patient's name"),
        Item(
            "age", "Takes the patient's age", judge="asked for the patient's age", needs="booking"
        ),
        Item(
            "contact",
            "Takes a contact number",
            judge="took or confirmed a contact number for the patient",
        ),
        Item(
            "city",
            "Takes the city or area",
            judge="asked which city or area the patient is coming from",
        ),
        Item(
            "complaint",
            "Understands the eye complaint",
            judge="understood the patient's eye complaint",
        ),
        Item(
            "offer_booking",
            "Offers to book an appointment with the doctor",
            judge="offered to book an appointment with the doctor",
        ),
        Item(
            "confirmation",
            "Repeats the details; reports, arrive early, dilation",
            judge="repeated the appointment details and told the caller to bring previous "
            "reports, come a little early, and bring someone along in case the eyes are dilated",
            needs="booking",
        ),
        _ANYTHING_ELSE,
        Item(
            "closing",
            "Thanks the caller and closes politely",
            judge="thanked the caller and closed the call politely",
        ),
        Item(
            "emergency",
            "Emergency: gives the urgent advice",
            kind="when",
            judge="told the caller to go to the nearest eye or hospital emergency right away",
        ),
        Item(
            "callback",
            "Takes a message and offers a callback when the answer has to wait",
            kind="when",
            judge="took a message and offered a callback from the team",
        ),
        *_RULES[:2],
    ),
)


def checklist_for(direction: str, channel: str = "phone") -> Checklist:
    if channel == "assist":
        return AGENT
    return OUTBOUND if direction == "outbound" else INBOUND
