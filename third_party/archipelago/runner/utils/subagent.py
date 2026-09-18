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

"""Utility for spawning blocking subagents from within a parent agent run."""

from typing import Any

from runner.agents.models import AgentImpl, AgentRunInput, AgentTrajectoryOutput


async def spawn_subagent(
    parent_run_input: AgentRunInput,
    *,
    agent_impl: AgentImpl,
    system_prompt: str,
    messages: list[Any],
    agent_config_values: dict[str, Any] | None = None,
    max_steps: int = 50,
) -> AgentTrajectoryOutput:
    """Launch a blocking subagent with isolated context and a targeted system prompt.

    Inherits MCP gateway, model, and extra_args from the parent. The caller is
    responsible for resolving agent_impl (e.g. via get_agent_impl) and for
    choosing which messages to pass — typically a focused subset, not the full
    parent history.

    Args:
        parent_run_input: The parent agent's run input, used to inherit shared
            resources (MCP gateway, model, auth token, extra_args).
        agent_impl: Resolved agent implementation to run.
        system_prompt: System prompt for the subagent, prepended as the first
            message in initial_messages.
        messages: Conversation messages to seed the subagent with.
        agent_config_values: Additional agent config overrides. max_steps is
            set from the max_steps param; anything here takes precedence.
        max_steps: Step budget for the subagent. Defaults to 50.

    Returns:
        AgentTrajectoryOutput from the subagent run. Final answer is in the last
        assistant message; status and usage are populated as normal.
    """
    initial_messages: list[Any] = [
        {"role": "system", "content": system_prompt},
        *messages,
    ]
    run_input = AgentRunInput(
        trajectory_id=parent_run_input.trajectory_id,
        initial_messages=initial_messages,
        mcp_gateway_url=parent_run_input.mcp_gateway_url,
        mcp_gateway_auth_token=parent_run_input.mcp_gateway_auth_token,
        mcp_gateway_actor_id=parent_run_input.mcp_gateway_actor_id,
        orchestrator_model=parent_run_input.orchestrator_model,
        orchestrator_extra_args=parent_run_input.orchestrator_extra_args,
        agent_config_values={"max_steps": max_steps, **(agent_config_values or {})},
    )
    return await agent_impl(run_input)
