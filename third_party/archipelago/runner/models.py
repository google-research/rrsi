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
Shared models for agent runner.
"""

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class TaskFieldType(StrEnum):
    """Types of custom fields that can be defined for agent config."""

    TEXT = "text"  # Single-line text input
    TEXTAREA = "textarea"  # Multi-line text input
    NUMBER = "number"  # Numeric input
    BOOLEAN = "boolean"  # Checkbox
    DATE = "date"  # Date picker
    DATETIME = "datetime"  # Date and time picker
    SELECT = "select"  # Single choice dropdown
    MULTISELECT = "multiselect"  # Multiple choice dropdown
    URL = "url"  # URL input with validation
    EMAIL = "email"  # Email input with validation
    ARTIFACT_MULTISELECT = (
        "artifact_multiselect"  # Multi-select file picker from snapshots
    )
    ARTIFACT_MULTISELECT_TRANSFORMED = (
        "artifact_multiselect_transformed"  # Multi-select with transformation options
    )
    LIKERT_SCALE = "likert_scale"  # Sliding integer scale with endpoint labels
    FILE = "file"  # File upload field, stores S3 keys
    SUBSCHEMA_LIST = "subschema_list"  # Nested list of field groups
    JSON = "json"  # Arbitrary JSON data
    ORCHESTRATOR_SELECT = "orchestrator_select"  # Dropdown populated with orchestrators
    AGENT_SELECT = "agent_select"  # Dropdown populated with agents
    PLATFORM_SELECT = "platform_select"  # Dropdown populated with platforms


class TaskFieldSchema(BaseModel):
    """Schema definition for a single agent config field."""

    field_id: str = Field(
        ...,
        description="Identifier for this field (e.g., 'timeout', 'max_steps').",
    )
    field_type: TaskFieldType = Field(
        ...,
        description="Type of field determines UI component and validation",
    )
    label: str = Field(
        ...,
        description="Human-readable label shown in UI",
    )
    required: bool = Field(
        default=False,
        description="Whether this field is required",
    )
    description: str | None = Field(
        default=None,
        description="Help text shown to users",
    )
    default_value: Any | None = Field(
        default=None,
        description="Default value when creating new configs",
    )
    options: list[str] | None = Field(
        default=None,
        description="Available options for select fields",
    )
    min_value: float | None = Field(
        default=None,
        description="Minimum value for number fields",
    )
    max_value: float | None = Field(
        default=None,
        description="Maximum value for number fields",
    )
    display_width: Literal["full", "half", "third"] = Field(
        default="full",
        description="Width in form layout",
    )

    # File field configuration
    max_files: int | None = Field(
        default=None,
        description="Maximum number of files allowed for file fields",
    )

    # Likert scale display labels
    display_min_explanation: str | None = Field(
        default=None,
        description="Label shown at the min end of a likert scale (e.g., 'Strongly Disagree')",
    )
    display_max_explanation: str | None = Field(
        default=None,
        description="Label shown at the max end of a likert scale (e.g., 'Strongly Agree')",
    )
    subschema: list["TaskFieldSchema"] | None = Field(
        default=None,
        description="Schema for items when field_type is subschema_list.",
    )
    conditional_render_filter: list[Any] | None = Field(
        default=None,
        description="Filters controlling when this field is shown in the UI.",
    )


# @apg_agent_config_model:start
class AgentConfig(BaseModel):
    """Agent configuration"""

    agent_config_id: str  # Which agent implementation (e.g., "loop_agent")
    agent_name: str  # Human-readable name (e.g., "Fast Loop Agent")
    agent_config_values: dict[str, Any]  # Agent-specific configuration values


# @apg_agent_config_model:end


TaskFieldSchema.model_rebuild()
