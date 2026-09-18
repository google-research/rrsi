# Copyright 2026 The rrsi Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tool result processing for handling large outputs.

Simple head_tail truncation - reliable and predictable.
"""

from typing import Any

from litellm import token_counter
from loguru import logger

from runner.agents.models import LitellmInputMessage

# Defaults for head_tail truncation
# With ReSum context summarization, we can afford larger results
MAX_RESULT_TOKENS = 24000  # ~24k tokens before truncation
HEAD_CHARS = 20000  # Keep first 20k chars
TAIL_CHARS = 5000  # Keep last 5k chars

# Absolute maximum - if result exceeds this even after truncation, return error
# This prevents absurdly large results from ever being added to context
ABSOLUTE_MAX_CHARS = 100000


def _estimate_tokens(model: str, text: str) -> int:
    """Estimate token count for text content."""
    try:
        return token_counter(model=model, text=text)
    except Exception:
        return len(text) // 4


def _truncate_text(text: str, model: str) -> str | None:
    """
    Truncate text if it exceeds limits. Returns truncated text or None if no change needed.

    Uses head_tail strategy: keep first HEAD_CHARS and last TAIL_CHARS.
    """
    # Check for absurdly large results first
    if len(text) > ABSOLUTE_MAX_CHARS * 2:
        logger.bind(message_type="tool_result").error(
            f"Tool result is extremely large ({len(text):,} chars), truncating"
        )
        return (
            f"Error: Tool returned extremely large output ({len(text):,} characters). "
            f"This exceeds the maximum allowed size. "
            f"Try a more specific query or break down the request."
        )

    tokens = _estimate_tokens(model, text)
    if tokens <= MAX_RESULT_TOKENS:
        return None  # No truncation needed

    logger.bind(message_type="tool_result").warning(
        f"Tool result is too large ({tokens} tokens > {MAX_RESULT_TOKENS}), truncating"
    )

    # Head-tail truncation
    if len(text) <= HEAD_CHARS + TAIL_CHARS:
        return None  # Content is fine as-is

    head = text[:HEAD_CHARS]
    tail = text[-TAIL_CHARS:]
    omitted = len(text) - HEAD_CHARS - TAIL_CHARS
    processed = (
        f"{head}\n\n"
        f"[... {omitted:,} characters omitted. "
        f"Use more specific queries to access full data. ...]\n\n"
        f"{tail}"
    )

    # Final safety check - ensure truncated result is within absolute max
    if len(processed) > ABSOLUTE_MAX_CHARS:
        logger.bind(message_type="tool_result").error(
            f"Truncated result still too large ({len(processed):,} chars)"
        )
        return (
            f"Error: Tool output too large even after truncation. "
            f"Original: {len(text):,} chars. Try a more specific query."
        )

    return processed


def _truncate_content_list(content: list[Any], model: str) -> None:
    """Truncate text blocks within a content list. Mutates in place."""
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "text":
            continue  # Preserve images and other types

        text = item.get("text", "")
        if not isinstance(text, str):
            continue

        truncated = _truncate_text(text, model)
        if truncated is not None:
            item["text"] = truncated


def truncate_tool_messages(
    messages: list[LitellmInputMessage],
    model: str,
) -> None:
    """
    Truncate text content in tool messages, preserving everything else (images, etc).

    Handles both:
    - Dict messages with list content: [{"type": "text", "text": "..."}]
    - Dict messages with string content (legacy)

    Mutates messages in place. Uses head_tail strategy for truncation.
    """
    for msg in messages:
        content = msg.get("content")

        if isinstance(content, list):
            # New format: content is array of blocks
            _truncate_content_list(content, model)

        elif isinstance(content, str):
            # Legacy format: content is string
            truncated = _truncate_text(content, model)
            if truncated is not None:
                msg["content"] = truncated
