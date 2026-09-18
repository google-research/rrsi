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
Agent registry mapping agent IDs to their implementations and config schemas.
"""

from typing import Any

from runner.agents.models import (
    AgentConfigIds,
    AgentDefn,
    AgentImpl,
    AgentRunInput,
    AgentTrajectoryOutput,
)
import importlib
import os


def react_toolbelt_agent_run(*args, **kwargs):
    """RRSI local modification: the evolvable harness package is named by the
    RRSI_HARNESS_MODULE environment variable (e.g. harness_workspace.main) instead of
    being vendored under runner/agents/, so one engine serves every domain."""
    mod = importlib.import_module(os.environ.get("RRSI_HARNESS_MODULE", "harness_workspace.main"))
    return mod.run(*args, **kwargs)
from runner.models import TaskFieldSchema, TaskFieldType

AGENT_REGISTRY: dict[AgentConfigIds, AgentDefn] = {
    AgentConfigIds.REACT_TOOLBELT_AGENT: AgentDefn(
        agent_config_id=AgentConfigIds.REACT_TOOLBELT_AGENT,
        agent_impl=react_toolbelt_agent_run,
        agent_config_fields=[
            TaskFieldSchema(
                field_id="timeout",
                field_type=TaskFieldType.NUMBER,
                label="Timeout (seconds)",
                description="Maximum time for agent execution",
                default_value=10800,  # 3 hours
                min_value=300,  # 5 minutes
                max_value=28800,  # 8 hours
            ),
            TaskFieldSchema(
                field_id="max_steps",
                field_type=TaskFieldType.NUMBER,
                label="Max Steps",
                description="Maximum number of LLM calls before stopping",
                default_value=250,
                min_value=1,
                max_value=1000,
            ),
            TaskFieldSchema(
                field_id="llm_response_timeout",
                field_type=TaskFieldType.NUMBER,
                label="LLM Response Timeout (seconds)",
                description="Timeout for a single LLM API call",
                default_value=600,
                min_value=30,
                max_value=10800,
            ),
        ],
    ),
}

def get_agent_impl(agent_config_id: str) -> AgentImpl:
    """
    Get the agent implementation function for the given agent config ID.

    Args:
        agent_config_id: The agent config ID to look up (e.g., "loop_agent")

    Returns:
        The agent implementation function

    Raises:
        ValueError: If the agent config ID is not found in the registry
    """
    try:
        config_id_enum = AgentConfigIds(agent_config_id)
    except ValueError as e:
        raise ValueError(f"Unknown agent config ID: {agent_config_id}") from e

    defn = AGENT_REGISTRY.get(config_id_enum)
    if defn is None:
        raise ValueError(f"Unknown agent config ID: {agent_config_id}")

    if defn.agent_impl is None:
        raise ValueError(
            f"Agent '{agent_config_id}' is registered but has no implementation"
        )

    return defn.agent_impl

def get_agent_defn(agent_config_id: str) -> AgentDefn:
    """
    Get the full agent definition for the given agent config ID.

    Args:
        agent_config_id: The agent config ID to look up (e.g., "loop_agent")

    Returns:
        The agent definition including config fields

    Raises:
        ValueError: If the agent config ID is not found in the registry
    """
    try:
        config_id_enum = AgentConfigIds(agent_config_id)
    except ValueError as e:
        raise ValueError(f"Unknown agent config ID: {agent_config_id}") from e

    defn = AGENT_REGISTRY.get(config_id_enum)
    if defn is None:
        raise ValueError(f"Unknown agent config ID: {agent_config_id}")

    return defn
