"""Deterministic LLM doubles so the full AgentSession + tools run offline (no API keys).

ScriptedLLM plays back a fixed list of steps, one per LLM call:
    ("say", "text")                      -> assistant text
    ("call", "tool_name", {...args})     -> a tool call
    ("calls", [("tool", {...}), ...])    -> several tool calls in one response
    ("say_call", "text", "tool", {...})  -> text, then a tool call, in one response
FailingLLM raises a connection error on every call (for fallback tests).
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections import deque
from typing import Any

from livekit.agents import APIConnectionError, APITimeoutError, llm
from livekit.agents.llm import ChatChunk, ChoiceDelta, FunctionToolCall
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions


class _Stream(llm.LLMStream):
    def __init__(self, owner: llm.LLM, step: Any, delay: float, **kw: Any) -> None:
        super().__init__(owner, **kw)
        self._step = step
        self._delay = delay

    async def _run(self) -> None:
        if self._delay:
            # behave like a real plugin: the HTTP timeout (conn_options.timeout) bounds the wait
            if self._delay >= self._conn_options.timeout:
                await asyncio.sleep(self._conn_options.timeout)
                raise APITimeoutError("simulated slow provider")
            await asyncio.sleep(self._delay)
        rid = "fake-" + uuid.uuid4().hex[:8]
        step = self._step
        if isinstance(step, Exception):
            raise step
        kind = step[0]
        if kind == "say_call":  # text first, then a tool call, in one response (like Claude)
            _, text, name, args = step
            self._step = ("say", text)
            await self._run()
            self._step = ("call", name, args)
            await self._run()
            return
        if kind == "say":
            for i, word in enumerate(step[1].split(" ")):
                self._event_ch.send_nowait(
                    ChatChunk(
                        id=rid,
                        delta=ChoiceDelta(role="assistant", content=(" " if i else "") + word),
                    )
                )
        elif kind in ("call", "calls"):
            calls = [(step[1], step[2])] if kind == "call" else step[1]
            self._event_ch.send_nowait(
                ChatChunk(
                    id=rid,
                    delta=ChoiceDelta(
                        role="assistant",
                        tool_calls=[
                            FunctionToolCall(
                                name=name,
                                arguments=json.dumps(args),
                                call_id="call_" + uuid.uuid4().hex[:8],
                            )
                            for name, args in calls
                        ],
                    ),
                )
            )
        else:  # pragma: no cover
            raise ValueError(step)


class ScriptedLLM(llm.LLM):
    def __init__(
        self, steps: list[Any], *, delay: float = 0.0, default: Any = ("say", "Okay.")
    ) -> None:
        super().__init__()
        self.steps = deque(steps)
        self.delay = delay
        self.default = default
        self.calls: list[llm.ChatContext] = []
        self.tools_seen: list[list[str]] = []

    @property
    def model(self) -> str:
        return "scripted"

    @property
    def provider(self) -> str:
        return "test"

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls=NOT_GIVEN,
        tool_choice=NOT_GIVEN,
        extra_kwargs=NOT_GIVEN,
    ) -> llm.LLMStream:
        self.calls.append(chat_ctx.copy())
        self.tools_seen.append([getattr(t, "id", getattr(t, "name", "?")) for t in (tools or [])])
        step = self.steps.popleft() if self.steps else self.default
        return _Stream(
            self, step, self.delay, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options
        )


class FailingLLM(ScriptedLLM):
    def __init__(self) -> None:
        super().__init__([])
        self.attempts = 0

    @property
    def model(self) -> str:
        return "always-down"

    def chat(self, **kw: Any) -> llm.LLMStream:  # type: ignore[override]
        self.attempts += 1
        self.steps.append(APIConnectionError("simulated provider outage"))
        return super().chat(**kw)
