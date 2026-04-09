"""Anthropic async client with model-routed API wrappers and prompt caching."""

import anthropic

SONNET = "claude-sonnet-4-6"
OPUS = "claude-opus-4-6"

_client: anthropic.AsyncAnthropic | None = None


def get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic()
    return _client


async def sonnet_call(
    system: str,
    user_content: str | list[dict],
    max_tokens: int = 256,
) -> str:
    """Call Sonnet for structured/simple outputs (location choice, rating, memory)."""
    client = get_client()
    msg = await client.messages.create(
        model=SONNET,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_content}],
    )
    return msg.content[0].text.strip()


async def opus_call(
    system: str,
    user_content: list[dict],
    max_tokens: int = 512,
) -> str:
    """Call Opus for creative speech generation."""
    client = get_client()
    msg = await client.messages.create(
        model=OPUS,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_content}],
    )
    return msg.content[0].text.strip()


def cached_user_blocks(shared: str, character: str) -> list[dict]:
    """Build user content blocks with cache breakpoint on the shared prefix."""
    return [
        {"type": "text", "text": shared, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": character},
    ]
