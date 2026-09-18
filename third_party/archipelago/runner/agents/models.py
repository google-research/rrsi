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
Models for agent definitions and execution.
"""

from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any

from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Message
from openai.types.responses.easy_input_message_param import EasyInputMessageParam
from pydantic import BaseModel, SerializeAsAny, field_validator

from runner.models import TaskFieldSchema
from runner.save._normalize import normalize_messages_for_report

# LiteLLM message types for agent execution:
# - InputMessage (AllMessageValues): TypedDict for Chat Completions API requests
# - ResponsesInputMessage (EasyInputMessageParam): TypedDict for Responses API requests
# - OutputMessage (Message): Pydantic model from API responses, used for new messages
# - AnyMessage: Union of all three, used for trajectory output (includes input + generated)
LitellmInputMessage = AllMessageValues
LitellmResponsesInputMessage = EasyInputMessageParam
LitellmOutputMessage = Message
LitellmAnyMessage = (
    LitellmInputMessage | LitellmResponsesInputMessage | LitellmOutputMessage
)

def get_msg_role(msg: LitellmAnyMessage) -> str:
    """Get role from either TypedDict or Pydantic Message."""
    if isinstance(msg, Message):
        return msg.role
    return msg["role"]

def get_msg_content(msg: LitellmAnyMessage) -> Any:
    """Get content from either TypedDict or Pydantic Message."""
    if isinstance(msg, Message):
        return msg.content
    return msg.get("content")

def get_msg_attr(msg: LitellmAnyMessage, key: str, default: Any = None) -> Any:
    """Get arbitrary attribute from either TypedDict or Pydantic Message."""
    if isinstance(msg, Message):
        return getattr(msg, key, default)
    return msg.get(key, default)

def content_to_str(content: Any) -> str:
    """Normalize message content to a string.

    Some providers (e.g. Anthropic) return content as a list of blocks
    like [{'type': 'text', 'text': '...'}]. Use this when you need a plain string.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text", "")
                if isinstance(text, str) and text:
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts) if parts else ""
    return str(content)

class AgentConfigIds(StrEnum):
    """Registry of available agent implementation IDs (e.g., 'loop_agent')."""

    LOOP_AGENT = "loop_agent"
    REACT_TOOLBELT_AGENT = "react_toolbelt_agent"

class AgentStatus(StrEnum):
    """Status of an agent run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    ERROR = "error"

class AgentRunInput(BaseModel):
    """Input to an agent implementation."""

    trajectory_id: str

    # The "actual" type of this is list[LitellmInputMessage] but given the way
    # pydantic works it makes sense to lazily handle this as just list[Any].
    # See also https://github.com/pydantic/pydantic/issues/9541.
    initial_messages: list[Any]

    mcp_gateway_url: str | None
    # Gateway calls can either populate `Authorization: Bearer <token>` with...
    # - mcp_gateway_auth_token = A real bearer token for secure AuthN
    # - mcp_gateway_actor_id = An Actor ID for user tenancy
    # - None = No authentication
    mcp_gateway_auth_token: str | None
    mcp_gateway_actor_id: str | None = None
    orchestrator_model: str
    orchestrator_extra_args: dict[str, Any] | None
    agent_config_values: dict[str, Any]

    # Parent trajectory output (for continuation trajectories used during multi-turn, None otherwise)
    parent_trajectory_output: dict[str, Any] | None = None

    # Arbitrary per-trajectory metadata from the orchestration request
    custom_args: dict[str, Any] | None = None

    # The task's custom_fields dict, passed through to all agents.
    # Most agents ignore this. Currently used by harness agents (e.g.
    # configurable_user_sim_agent) for task-level overrides of prompts/models.
    task_custom_fields: dict[str, Any] | None = None

    # Resolved inner agent config (for harness agents that delegate to an agent instance)
    inner_agent_config: dict[str, Any] | None = None

class AgentTrajectoryOutput(BaseModel):
    """Output from an agent run"""

    # Litellm Message is a Pydantic model; SerializeAsAny avoids union mismatch warnings
    # and keeps tool_calls in model_dump_json() for integration test assertions.
    messages: SerializeAsAny[list[LitellmAnyMessage]]
    output: dict[str, Any] | None = None
    status: AgentStatus
    time_elapsed: float
    usage: dict[str, Any] | None = None

    @field_validator("messages", mode="before")
    @classmethod
    def _normalize_messages(cls, messages: Any) -> Any:
        """Normalize tool-message image content at construction time.

        LiteLLM's ``ChatCompletionToolMessage`` schema restricts tool content to
        text-only blocks. Tools like ``code_exec`` can return ``image_url``
        blocks (matplotlib PNGs, screenshots, etc.) that would otherwise break
        any downstream ``model_dump()`` / ``model_dump_json()`` with a
        ``PydanticSerializationError``. Running normalization here — rather than
        at a single call site like ``webhook.py`` — means every serialization of
        a trajectory output (webhook POST, local file write in ``runner/main.py``,
        per-agent logging, integration-test assertions) sees the same
        spec-compliant shape.

        Must be ``mode="before"``: by ``mode="after"`` Pydantic has already
        wrapped tool-message ``content`` in a ``ValidatorIterator`` for the
        ``str | Iterable`` union, at which point ``isinstance(content, list)``
        is false and the normalizer can't inspect the blocks.

        Drop once LiteLLM's tool-message schema accepts image/audio parts.
        """
        if not isinstance(messages, list):
            return messages
        return normalize_messages_for_report(messages)

AgentImpl = Callable[[AgentRunInput], Awaitable[AgentTrajectoryOutput]]

class AgentDefn(BaseModel):
    """Definition of an agent implementation in the registry."""

    agent_config_id: AgentConfigIds
    agent_impl: AgentImpl | None = None  # Optional - server doesn't need implementation
    run_custom_args_schema: list[TaskFieldSchema] = []  # Per-run custom args schema
    agent_config_fields: list[TaskFieldSchema]  # Configurable fields for this agent

    class Config:
        arbitrary_types_allowed = True
