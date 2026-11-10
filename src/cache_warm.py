"""Write the LLM prompt cache while the greeting plays, so the caller's first reply is fast.

The system prompt (the front desk SOP + clinic facts: about 10k tokens) is the
same all day, but Anthropic keeps a cache entry for only 5 minutes: after a quiet spell the
call's first request was processed from scratch, about 0.7 s slower than the replies after
it. One throwaway request with the same tools and instructions, sent at call start, writes
(or refreshes) the cache. Its reply is capped at a few tokens: only the input matters.
"""

from __future__ import annotations

import contextlib
import logging
import time

from livekit.agents import Agent, llm

logger = logging.getLogger("frontdesk.latency")
WARMUP_MAX_TOKENS = 32  # the cache is written from the input; the reply itself is thrown away


def _warmup_twin(model: llm.LLM) -> llm.LLM | None:
    """A separate client for the call's first Anthropic model: its request must not count in
    the call's latency metrics, and a failed warm-up must not make the fallback adapter mark
    the provider unavailable."""
    instances = getattr(model, "_llm_instances", None)
    primary = instances[0] if instances else model
    from livekit.plugins import anthropic

    if not isinstance(primary, anthropic.LLM):
        return None  # only Anthropic's cache needs writing up front
    from providers import ChainItem, _make_llm

    return _make_llm(ChainItem("anthropic", primary.model), max_tokens=WARMUP_MAX_TOKENS)


async def warm_prompt_cache(model: llm.LLM | None, agent: Agent, on_usage=None) -> bool:
    """on_usage(model_name, CompletionUsage): the warm-up's tokens, for the call's cost."""
    if model is None:
        return False
    try:
        twin = _warmup_twin(model)
    except Exception as e:
        logger.info("prompt cache warm-up skipped: %s", e)
        return False
    if twin is None:
        return False
    ctx = llm.ChatContext.empty()
    ctx.add_message(role="system", content=agent.instructions)
    ctx.add_message(role="user", content="Hello")
    started = time.perf_counter()
    usage = None
    try:
        async with twin.chat(chat_ctx=ctx, tools=agent.tools) as stream:
            # read to the end (capped reply): the last chunk carries the exact token usage
            async for chunk in stream:
                if chunk.usage is not None:
                    usage = chunk.usage
    except Exception as e:  # warm-up is an optimisation, never a failure
        logger.info("prompt cache warm-up failed: %s", e)
        return False
    finally:
        with contextlib.suppress(Exception):
            await twin.aclose()
    logger.info(
        "prompt cache warmed in %.0f ms (%s cached, %s written)",
        (time.perf_counter() - started) * 1000,
        getattr(usage, "cache_read_tokens", "?"),
        getattr(usage, "cache_creation_tokens", "?"),
    )
    if usage is not None and on_usage is not None:
        try:
            on_usage(f"api.anthropic.com/{twin.model}", usage)
        except Exception:
            logger.exception("could not record warm-up usage")
    return True
