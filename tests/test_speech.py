"""Fixed lines are spoken word for word, and the repeated "not caught" prompts differ."""

from __future__ import annotations

from types import SimpleNamespace

from speech import speak


def test_fixed_lines_are_spoken_word_for_word():
    calls = []
    session = SimpleNamespace(tts=object(), say=lambda text, **kw: calls.append((text, kw)))
    speak(session, "Hello, am I audible?", allow_interruptions=False)
    assert calls == [
        ("Hello, am I audible?", {"add_to_chat_ctx": True, "allow_interruptions": False})
    ]


def test_text_only_sessions_still_say_lines():
    calls = []
    text_only = SimpleNamespace(tts=None, llm=object(), say=lambda text, **kw: calls.append(text))
    speak(text_only, "Hello")
    assert calls == ["Hello"]


def test_a_recording_can_be_played_as_the_line():
    calls = []
    session = SimpleNamespace(tts=None, say=lambda text, **kw: calls.append(kw))
    speak(session, "hi", audio="frames")
    assert calls == [{"add_to_chat_ctx": True, "audio": "frames"}]


def test_repeated_not_caught_prompts_change_wording():
    from lines import LINES

    assert len({LINES[k]["en"] for k in ("not_caught", "not_caught_2", "not_caught_3")}) == 3
