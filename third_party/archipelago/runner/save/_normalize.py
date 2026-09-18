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

"""Message normalization for trajectory reporting.

LiteLLM's ``ChatCompletionToolMessage`` schema types ``content`` as
``str | Iterable[ChatCompletionTextObject]`` — i.e. tool messages may only hold
plain text blocks. In practice, models like ``claude-opus-4-7`` can produce
tool results containing ``image_url`` blocks (e.g. from ``code_exec`` returning
a matplotlib PNG). When the agent runner then calls ``model_dump(mode="json")``
on its ``AgentTrajectoryOutput``, Pydantic iterates the content iterator and
rejects image blocks with ``ValidationError: Input should be 'text'``, which
manifests as a ``PydanticSerializationError`` in the webhook.

This module collapses any non-text blocks inside tool-message content into a
text placeholder so the trajectory can still be reported. We preserve the fact
that an image was present — the grader only needs the agent's final text
answer, and keeping a human-readable marker leaves the trajectory auditable.
Drop this shim once LiteLLM's tool-message schema accepts image/audio parts.
"""

from __future__ import annotations

import copy
from typing import Any

from litellm.types.utils import Message

# The message union type (``LitellmAnyMessage``) lives in ``runner.agents.models``,
# which imports this module for its ``AgentTrajectoryOutput.messages``
# field_validator. Avoiding the reverse import — even under ``TYPE_CHECKING`` —
# keeps the module dependency graph acyclic, which some type checkers flag on.
# Annotations use ``Any`` below; runtime dispatch already handles both shapes
# (dict-form TypedDict messages and Pydantic ``Message`` objects).


def _block_placeholder(block: dict[str, Any]) -> dict[str, Any]:
    """Convert a non-text content block into a text placeholder."""
    block_type = block.get("type") or "unknown"
    label = block_type

    # For the common OpenAI image_url shape, note the format if we can infer it.
    if block_type == "image_url":
        image_url = block.get("image_url")
        if isinstance(image_url, dict):
            url = image_url.get("url") or ""
        elif isinstance(image_url, str):
            url = image_url
        else:
            url = ""
        if url.startswith("data:") and ";" in url:
            mime = url.split(":", 1)[1].split(";", 1)[0]
            label = f"image_url {mime}"

    return {"type": "text", "text": f"[{label} content elided from trajectory report]"}


def _normalize_tool_content(content: Any) -> Any:
    """Replace any non-text blocks in tool-message content with text placeholders."""
    if not isinstance(content, list):
        return content

    normalized: list[Any] = []
    touched = False
    for block in content:
        if (
            isinstance(block, dict)
            and block.get("type")
            and block.get("type") != "text"
        ):
            normalized.append(_block_placeholder(block))
            touched = True
        else:
            normalized.append(block)
    return normalized if touched else content


def normalize_message_for_report(msg: Any) -> Any:
    """Return a copy of ``msg`` with tool-message content stripped of non-text blocks.

    Non-tool messages and tool messages that already contain only text are
    returned unchanged (and un-copied). ``msg`` is typed ``Any`` to accept any
    of LiteLLM's message shapes (``AllMessageValues`` TypedDicts, Pydantic
    ``Message`` objects) without a direct import of the union type (see
    module-level note on the import cycle).
    """
    if isinstance(msg, Message):
        if msg.role != "tool":
            return msg
        normalized_content = _normalize_tool_content(msg.content)
        if normalized_content is msg.content:
            return msg
        # Message is a Pydantic model; use model_copy to avoid in-place mutation.
        return msg.model_copy(update={"content": normalized_content})

    # Dict-shaped messages (LiteLLM TypedDicts).
    if not isinstance(msg, dict):
        return msg
    if msg.get("role") != "tool":
        return msg
    normalized_content = _normalize_tool_content(msg.get("content"))
    if normalized_content is msg.get("content"):
        return msg
    copied = copy.copy(msg)
    copied["content"] = normalized_content
    return copied


def normalize_messages_for_report(messages: list[Any]) -> list[Any]:
    """Apply :func:`normalize_message_for_report` to each message in a list."""
    return [normalize_message_for_report(m) for m in messages]
