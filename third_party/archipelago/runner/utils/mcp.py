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

"""MCP client helpers for agents using LiteLLM."""

import asyncio
from typing import Any

from loguru import logger
from mcp.types import ContentBlock, ImageContent, TextContent

from runner.agents.models import LitellmInputMessage

# Grace period (seconds) to wait for a shielded MCP tool call to finish
# after the primary timeout expires, before forcibly cancelling it.
SHIELDED_TASK_GRACE_SECONDS = 5.0


async def drain_shielded_task(task: asyncio.Task[Any]) -> None:
    """Wait for a shielded MCP task to finish, cancelling if it takes too long.

    After an ``asyncio.wait_for`` timeout, the shielded inner task is still
    running and holds the MCP session open.  Attempting a new tool call on the
    same session while the old one is in-flight causes a
    ``RuntimeError("Client is not connected")`` from the streamable-http
    transport.

    This helper gives the task a short grace period to complete naturally.  If
    it doesn't finish in time, the task is cancelled so the session is released
    before the next call.
    """
    if task.done():
        return
    try:
        await asyncio.wait_for(task, timeout=SHIELDED_TASK_GRACE_SECONDS)
    except (TimeoutError, Exception):
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


def build_mcp_gateway_schema(
    mcp_gateway_url: str,
    mcp_gateway_auth_token: str | None,
    mcp_gateway_actor_id: str | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Build the MCP client config schema for connecting to the environment's MCP gateway.

    The gateway is a single HTTP endpoint that proxies to all configured MCP servers
    in the environment sandbox.

    Args:
        mcp_gateway_url: URL of the MCP gateway (e.g. "http://localhost:8000/mcp/")
        mcp_gateway_auth_token: Real bearer token for authenticated gateway runtimes.
        mcp_gateway_actor_id: Runtime actor ID for user tenancy.

    Returns:
        The standard schema expected by the MCP client.
    """
    gateway_config: dict[str, Any] = {
        "transport": "streamable-http",
        "url": mcp_gateway_url,
    }

    # Authorization is only set when the runner explicitly provides a token or actor ID.
    auth_value = mcp_gateway_actor_id or mcp_gateway_auth_token
    if auth_value:
        gateway_config["headers"] = {"Authorization": f"Bearer {auth_value}"}

    return {
        "mcpServers": {
            "gateway": gateway_config,
        }
    }


def content_blocks_to_messages(
    content_blocks: list[ContentBlock],
    tool_call_id: str,
    name: str,
    model: str,
    deferred_image_messages: list[LitellmInputMessage],
) -> list[LitellmInputMessage]:
    """
    Convert MCP content blocks to a single LiteLLM tool message.

    Each tool_use must have exactly one tool_result. This function combines all
    content blocks into a single tool message to satisfy API requirements for
    Anthropic, OpenAI, and other providers.

    For non-Anthropic models, images cannot be embedded in tool results, so they
    are appended to deferred_image_messages as user messages. The caller is
    responsible for adding them to self.messages after all tool responses.
    This list is mutated in place.

    Args:
        content_blocks: MCP content blocks from tool result
        tool_call_id: The tool call ID to associate with the result
        name: The tool name
        model: The model being used
        deferred_image_messages: Mutable list that image user messages are
            appended to (mutated in place). Callers should extend self.messages
            with this list after all tool responses are added.

    Returns:
        List containing exactly one tool message.
    """
    # Anthropic supports images directly in tool results
    supports_image_tool_results = model.startswith("anthropic/")

    text_contents: list[str] = []
    image_data_uris: list[str] = []

    for content_block in content_blocks:
        match content_block:
            case TextContent():
                block = TextContent.model_validate(content_block)
                text_contents.append(block.text)

            case ImageContent():
                block = ImageContent.model_validate(content_block)
                data_uri = f"data:{block.mimeType};base64,{block.data}"
                image_data_uris.append(data_uri)

            case _:
                logger.warning(f"Content block type {content_block.type} not supported")
                text_contents.append("Unable to parse tool call response")

    messages: list[LitellmInputMessage] = []

    if supports_image_tool_results:
        content: list[dict[str, Any]] = []
        for text in text_contents:
            content.append({"type": "text", "text": text})
        for data_uri in image_data_uris:
            content.append({"type": "image_url", "image_url": {"url": data_uri}})

        tool_message: LitellmInputMessage = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": content if content else [{"type": "text", "text": ""}],
        }  # pyright: ignore[reportAssignmentType]
        messages.append(tool_message)
    else:
        content = [{"type": "text", "text": text} for text in text_contents]

        if image_data_uris and not content:
            content.append(
                {"type": "text", "text": f"Image(s) returned by {name} tool"}
            )

        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": content if content else [{"type": "text", "text": ""}],
        }  # pyright: ignore[reportAssignmentType]
        messages.append(tool_message)

        # Image workaround: non-Anthropic models don't support images in tool results,
        # so we append them to deferred_image_messages for the caller to add after all tool responses.
        for data_uri in image_data_uris:
            deferred_image_messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            )

    return messages
