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

"""Token usage tracking for agent LLM calls."""

from typing import Any

from litellm.files.main import ModelResponse


def _coerce_int(value: Any) -> int:
    """Defensively coerce a usage attribute to a non-negative int.

    LiteLLM occasionally leaves details fields as None, strings, or floats
    depending on the upstream provider's shape, so we normalize here.
    """
    if value is None:
        return 0
    try:
        return int(value) if int(value) > 0 else 0
    except (TypeError, ValueError):
        return 0


def _get_attr_or_item(obj: Any, key: str) -> Any:
    """Read a key from either a pydantic model / dataclass or a plain dict."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


class UsageTracker:
    """Accumulates token usage across multiple LLM calls during agent execution."""

    def __init__(self) -> None:
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.final_answer_tokens: int = 0
        self.max_prompt_tokens: int = 0
        self.compaction_count: int = 0
        # Provider-specific detail fields. LiteLLM normalizes these onto
        # `usage.completion_tokens_details` / `usage.prompt_tokens_details`,
        # so we extract them once here and let downstream consumers persist
        # them alongside the basic counts.
        self.reasoning_tokens: int = 0
        self.cached_tokens: int = 0
        self.cache_creation_tokens: int = 0
        # Per-call log: one entry per LLM call in order of execution.
        self.call_log: list[dict[str, int]] = []

    def track_compaction(self) -> None:
        """Increment compaction count when a context summarization LLM call fires."""
        self.compaction_count += 1

    def track(self, response: ModelResponse) -> None:
        """Extract and accumulate usage from a ModelResponse."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return

        call_prompt_tokens = _coerce_int(getattr(usage, "prompt_tokens", 0))
        call_completion_tokens = _coerce_int(getattr(usage, "completion_tokens", 0))

        self.prompt_tokens += call_prompt_tokens
        self.completion_tokens += call_completion_tokens
        self.final_answer_tokens = call_completion_tokens
        self.max_prompt_tokens = max(self.max_prompt_tokens, call_prompt_tokens)

        completion_details = getattr(usage, "completion_tokens_details", None)
        call_reasoning_tokens = _coerce_int(
            _get_attr_or_item(completion_details, "reasoning_tokens")
        )
        self.reasoning_tokens += call_reasoning_tokens

        prompt_details = getattr(usage, "prompt_tokens_details", None)
        # LiteLLM normalizes Anthropic's `cache_read_input_tokens` →
        # `prompt_tokens_details.cached_tokens` and
        # `cache_creation_input_tokens` →
        # `prompt_tokens_details.cache_creation_tokens`. We also fall back to
        # the flat Anthropic-shaped attributes in case a backend hasn't been
        # routed through the normalizer.
        cached = _get_attr_or_item(prompt_details, "cached_tokens")
        if not cached:
            cached = getattr(usage, "cache_read_input_tokens", None)
        call_cached_tokens = _coerce_int(cached)
        self.cached_tokens += call_cached_tokens

        cache_creation = _get_attr_or_item(prompt_details, "cache_creation_tokens")
        if not cache_creation:
            cache_creation = getattr(usage, "cache_creation_input_tokens", None)
        call_cache_creation_tokens = _coerce_int(cache_creation)
        self.cache_creation_tokens += call_cache_creation_tokens

        self.call_log.append(
            {
                "prompt_tokens": call_prompt_tokens,
                "completion_tokens": call_completion_tokens,
                "total_tokens": call_prompt_tokens + call_completion_tokens,
                "reasoning_tokens": call_reasoning_tokens,
                "cached_tokens": call_cached_tokens,
                "cache_creation_tokens": call_cache_creation_tokens,
            }
        )

    def track_from_dict(self, response_dict: dict[str, Any]) -> None:
        """Extract and accumulate usage from a response dictionary (e.g., Responses API).

        Handles both OpenAI Responses API format and standard completion format.
        """
        usage = response_dict.get("usage")
        if usage is None:
            return

        prompt_tokens = _coerce_int(
            usage.get("prompt_tokens") or usage.get("input_tokens")
        )
        completion_tokens = _coerce_int(
            usage.get("completion_tokens") or usage.get("output_tokens")
        )

        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.final_answer_tokens = completion_tokens
        self.max_prompt_tokens = max(self.max_prompt_tokens, prompt_tokens)

        completion_details = usage.get("completion_tokens_details") or {}
        # Responses API exposes reasoning tokens as `output_tokens_details.reasoning_tokens`.
        output_details = usage.get("output_tokens_details") or {}
        call_reasoning_tokens = _coerce_int(
            completion_details.get("reasoning_tokens")
            or output_details.get("reasoning_tokens")
        )
        self.reasoning_tokens += call_reasoning_tokens

        prompt_details = usage.get("prompt_tokens_details") or {}
        input_details = usage.get("input_tokens_details") or {}
        cached = (
            prompt_details.get("cached_tokens")
            or input_details.get("cached_tokens")
            or usage.get("cache_read_input_tokens")
        )
        call_cached_tokens = _coerce_int(cached)
        self.cached_tokens += call_cached_tokens

        cache_creation = prompt_details.get("cache_creation_tokens") or usage.get(
            "cache_creation_input_tokens"
        )
        call_cache_creation_tokens = _coerce_int(cache_creation)
        self.cache_creation_tokens += call_cache_creation_tokens

        self.call_log.append(
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "reasoning_tokens": call_reasoning_tokens,
                "cached_tokens": call_cached_tokens,
                "cache_creation_tokens": call_cache_creation_tokens,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Return accumulated usage as a dict for AgentTrajectoryOutput."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "final_answer_tokens": self.final_answer_tokens,
            "max_prompt_tokens": self.max_prompt_tokens,
            "compaction_count": self.compaction_count,
            "reasoning_tokens": self.reasoning_tokens,
            "cached_tokens": self.cached_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "call_log": self.call_log,
        }
