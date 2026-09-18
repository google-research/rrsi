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

"""Shared tool output truncation utilities.

Used by agents that receive large MCP tool responses (e.g. web pages)
and need to truncate them to avoid filling the context window.
"""

from typing import Any

from runner.agents.models import LitellmAnyMessage


def truncate_tool_output(
    content: str, max_lines: int = 200, max_chars: int = 32768
) -> str:
    """Truncate content at whichever limit is hit first: max_lines or max_chars."""
    if not content:
        return content

    lines = content.split("\n")
    total_lines = len(lines)

    # Check if we need to truncate by lines
    if total_lines > max_lines:
        truncated_content = "\n".join(lines[:max_lines])
        # Also check char limit on the line-truncated content
        if len(truncated_content) > max_chars:
            truncated_content = truncated_content[:max_chars]
            kept_lines = truncated_content.count("\n") + 1
            hidden_lines = total_lines - kept_lines
        else:
            hidden_lines = total_lines - max_lines
        return truncated_content + f"\n... (truncated {hidden_lines} lines) ..."

    # Check if we need to truncate by chars
    if len(content) > max_chars:
        truncated_content = content[:max_chars]
        kept_lines = truncated_content.count("\n") + 1
        hidden_lines = total_lines - kept_lines

        if hidden_lines > 0:
            suffix = f"\n... (truncated {hidden_lines} lines) ..."
        else:
            hidden_chars = len(content) - len(truncated_content)
            suffix = f"\n... (truncated {hidden_chars} chars) ..."

        return truncated_content + suffix

    return content


def truncate_tool_message(
    msg: LitellmAnyMessage,
    max_lines: int = 200,
    max_chars: int = 32768,
) -> LitellmAnyMessage:
    """Apply truncation to a tool message's content.

    Handles both plain string content and structured content blocks
    (list[dict] with {"type": "text", "text": "..."} entries).
    """
    if not isinstance(msg, dict) or msg.get("role") != "tool":
        return msg

    content: Any = msg.get("content")
    if isinstance(content, str):
        msg["content"] = truncate_tool_output(content, max_lines, max_chars)
    elif isinstance(content, list):
        truncated_parts: list[dict[str, str]] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                truncated_text = truncate_tool_output(
                    str(part.get("text", "")), max_lines, max_chars
                )
                truncated_parts.append({"type": "text", "text": truncated_text})
            else:
                truncated_parts.append(part)  # pyright: ignore[reportArgumentType]
        msg["content"] = truncated_parts  # pyright: ignore[reportGeneralTypeIssues]
    return msg
