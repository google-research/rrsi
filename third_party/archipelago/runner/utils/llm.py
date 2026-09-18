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

"""LLM utilities for agents using LiteLLM."""

from enum import StrEnum
from typing import Any

import litellm
from litellm import acompletion, aresponses
from litellm.exceptions import (
    APIConnectionError,
    BadGatewayError,
    BadRequestError,
    ContextWindowExceededError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from litellm.files.main import ModelResponse
from litellm.types.utils import Message
from loguru import logger
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

import runner.utils.litellm_patches  # noqa: F401
from runner.agents.models import LitellmAnyMessage
from runner.utils.decorators import llm_attempt_ctx, llm_call_id_ctx, with_retry
from runner.utils.settings import get_settings

settings = get_settings()

# Configure LiteLLM proxy routing if configured
if settings.LITELLM_PROXY_API_BASE and settings.LITELLM_PROXY_API_KEY:
    litellm.use_litellm_proxy = True


# Params that must be sent through extra_body so LiteLLM's proxy client does not
# drop them. LiteLLMProxyChatConfig._map_openai_params silently filters out any
# top-level kwarg that is not in OPENAI_CHAT_COMPLETION_PARAMS, which strips
# vendor-specific flags like `chat_template_kwargs` (Qwen/Nemotron thinking
# controls) before they can reach the proxy.
_EXTRA_BODY_PASSTHROUGH_KEYS = frozenset({"chat_template_kwargs"})


def _extract_psf_reasoning_delta(chunk: Any) -> str:
    """Pull intermediate-output text out of a chunk's `provider_specific_fields`.

    Deliberately narrow: only reads `provider_specific_fields["reasoning_content"]`,
    never `delta.reasoning_content`. This scopes per-delta streaming logs to
    external-agent-harness providers — today just Gemini Deep Research, where
    `GenericStreamingChunk` has no first-class reasoning field so the harness
    surfaces its progress narrations via `provider_specific_fields` instead.

    The resulting logs are tagged `message_type="intermediate_output"` (not
    `"reasoning"`) because these aren't model-internal chain-of-thought —
    they're progress updates the harness emits while it's working (search
    plans, section headings, milestone narrations). The UI renders them
    with a distinct badge so viewers can tell them apart from genuine
    chain-of-thought reasoning from R1 / o-series / Anthropic thinking
    (which arrive on `delta.reasoning_content` and are logged once as
    `"reasoning"` at end-of-call by singleshot_agent).

    The field is kept as `reasoning_content` on the wire because that's
    litellm's generic channel for "non-final text"; only the downstream
    `message_type` tag distinguishes intermediate outputs from reasoning.
    """
    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return ""
    delta = getattr(choices[0], "delta", None)
    if delta is None:
        return ""
    psf = getattr(delta, "provider_specific_fields", None)
    if isinstance(psf, dict):
        reasoning = psf.get("reasoning_content")
        if isinstance(reasoning, str):
            return reasoning
    return ""


def responses_args_to_completions(extra_args: dict[str, Any]) -> dict[str, Any]:
    """Convert Responses API extra_args to Chat Completions API equivalents.

    The Responses API uses ``{"reasoning": {"effort": "high", ...}}`` while
    Chat Completions uses ``{"reasoning_effort": "high"}`` as a top-level
    param. Sending the Responses-API shape to a Chat-Completions endpoint
    yields ``Unknown parameter: 'reasoning'`` 400s.
    """
    result = {k: v for k, v in extra_args.items() if k != "reasoning"}
    reasoning = extra_args.get("reasoning")
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
        if effort and "reasoning_effort" not in result:
            result["reasoning_effort"] = effort
    return result


def _split_extra_args(
    extra_args: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split extra_args into (top_level, extra_body) for proxy-safe transport."""
    top_level: dict[str, Any] = {}
    extra_body: dict[str, Any] = {}
    for k, v in extra_args.items():
        if k in _EXTRA_BODY_PASSTHROUGH_KEYS:
            extra_body[k] = v
        else:
            top_level[k] = v
    return top_level, extra_body


class CacheControlAllowlist(StrEnum):
    """Targets where we explicitly attach Anthropic-style ``cache_control``.

    Anthropic-family models *require* an explicit ``cache_control`` field to
    enable prompt caching. Every other major provider Studio routes to
    (OpenAI, Gemini direct, Vertex Gemini, OpenRouter) caches input prompts
    automatically — adding ``cache_control`` for those paths would be
    noise without benefit, so we don't bother.

    Members may be either:
    - A LiteLLM provider prefix (matched against the ``provider/`` segment
      of the model arg): enables caching for every model routed through
      that provider.
    - A full ``provider/model`` string (matched verbatim against the
      ``model`` argument): enables caching for that specific model only.
    """

    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"


_CACHE_CONTROL_ALLOWLIST: frozenset[str] = frozenset(
    member.value for member in CacheControlAllowlist
)
# Explicit 5-minute TTL. ``{"type": "ephemeral"}`` alone also defaults to 5m,
# but pinning ``ttl="5m"`` keeps us from silently inheriting any future
# change Anthropic makes to the default and makes the choice auditable
# alongside the (more expensive) ``ttl="1h"`` alternative.
_EPHEMERAL_CACHE: dict[str, str] = {"type": "ephemeral", "ttl": "5m"}


def _is_cache_control_allowed(model: str) -> bool:
    """Whether ``model`` accepts Anthropic-style ``cache_control`` markers.

    The provider is read from the model prefix (``"anthropic"`` from
    ``"anthropic/claude-opus-4-7"``) rather than via
    ``litellm.get_llm_provider`` — that helper short-circuits to
    ``"litellm_proxy"`` whenever ``litellm.use_litellm_proxy`` is set
    (always true in Studio prod), losing the underlying provider info.
    """
    provider = model.split("/", 1)[0] if "/" in model else None
    return provider in _CACHE_CONTROL_ALLOWLIST or model in _CACHE_CONTROL_ALLOWLIST


def _with_cached_system_prompt(
    messages: list[LitellmAnyMessage], model: str
) -> list[LitellmAnyMessage]:
    """Return ``messages`` with the system prompt marked ephemerally cacheable.

    No-op unless the model's provider prefix or the full model string is in
    :class:`CacheControlAllowlist`. On a match, attaches
    ``cache_control={"type": "ephemeral", "ttl": "5m"}`` to the final
    content block of the first system message. Idempotent: a no-op if
    there is no system message, if the system message isn't a dict (system
    messages are dicts in agent contexts), or if the last block already
    carries ``cache_control``.
    """
    if not _is_cache_control_allowed(model):
        return messages

    # The constructed system messages carry a ``cache_control`` field that
    # OpenAI's TypedDicts (the backbone of LitellmAnyMessage) don't model,
    # even though LiteLLM passes it through to Anthropic. Suppress the
    # return-type complaint at the construction sites only.
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") != "system":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content:
            block = {"type": "text", "text": content, "cache_control": _EPHEMERAL_CACHE}
            new_msg: LitellmAnyMessage = {"role": "system", "content": [block]}  # pyright: ignore[reportAssignmentType]
            return [*messages[:i], new_msg, *messages[i + 1 :]]
        if isinstance(content, list) and content:
            last = content[-1]
            if isinstance(last, dict) and "cache_control" not in last:
                cached_last = {**last, "cache_control": _EPHEMERAL_CACHE}
                new_msg: LitellmAnyMessage = {  # pyright: ignore[reportAssignmentType]
                    **msg,
                    "content": [*content[:-1], cached_last],
                }
                return [*messages[:i], new_msg, *messages[i + 1 :]]
        return messages
    return messages


def _with_cached_tools(
    tools: list[ChatCompletionToolParam], model: str
) -> list[ChatCompletionToolParam]:
    """Return ``tools`` with the last tool marked ephemerally cacheable.

    Anthropic's prompt cache extends from the start of the request through
    the *last* ``cache_control`` marker; placing one on the final tool
    definition caches the entire system+tools prefix together. With Stirrup
    loads of 7+ MCP tools and Opus tool schemas of a few KB each, this is
    typically the largest still-uncached chunk of the per-turn input.

    The marker sits at the top level of the tool dict (alongside ``"type"``
    and ``"function"``), where LiteLLM's Anthropic transformer picks it up
    when converting OpenAI-format tools into Anthropic's native shape.

    No-op unless the provider/model is allowlisted, the list is empty, or
    the last tool already carries ``cache_control``. Idempotent.
    """
    if not _is_cache_control_allowed(model):
        return tools
    if not tools:
        return tools
    last = tools[-1]
    if not isinstance(last, dict) or last.get("cache_control") is not None:
        return tools
    cached_last = {**last, "cache_control": _EPHEMERAL_CACHE}
    return [*tools[:-1], cached_last]  # pyright: ignore[reportReturnType]


def _with_cached_last_message(
    messages: list[LitellmAnyMessage], model: str
) -> list[LitellmAnyMessage]:
    """Return ``messages`` with the most recent message marked ephemerally cacheable.

    Conversation history is append-only across an agent loop, so marking the
    *last* message extends the cached prefix through every prior message.
    On the next turn — when one or two more messages have been appended —
    Anthropic's cache lookup falls back to this marker as the longest
    matching prefix and processes only the new suffix at full price.

    Operates on the last message regardless of role (assistant tool_use,
    tool result, user follow-up). String content is wrapped into a single
    text block; list content has its final block annotated. Pydantic
    ``Message`` instances are skipped — their content is typed without a
    ``cache_control`` field, and the system + tools breakpoints still
    cover the static prefix even when the rolling marker no-ops.

    Idempotent: skips when the target block already carries
    ``cache_control``.
    """
    if not _is_cache_control_allowed(model):
        return messages
    if not messages:
        return messages
    last = messages[-1]
    if not isinstance(last, dict):
        return messages

    content = last.get("content")

    if isinstance(content, str):
        if not content:
            return messages
        block = {"type": "text", "text": content, "cache_control": _EPHEMERAL_CACHE}
        new_msg: LitellmAnyMessage = {**last, "content": [block]}  # pyright: ignore[reportAssignmentType]
        return [*messages[:-1], new_msg]

    if isinstance(content, list) and content:
        last_block = content[-1]
        if isinstance(last_block, dict) and "cache_control" not in last_block:
            cached_block = {**last_block, "cache_control": _EPHEMERAL_CACHE}
            new_msg_list: LitellmAnyMessage = {  # pyright: ignore[reportAssignmentType]
                **last,
                "content": [*content[:-1], cached_block],
            }
            return [*messages[:-1], new_msg_list]

    return messages


def normalize_assistant_tool_call_content(
    messages: list[LitellmAnyMessage],
) -> list[LitellmAnyMessage]:
    """Ensure assistant messages with tool_calls have non-empty content.

    Some model endpoints (e.g. alabaster/100) return HTTP 500 when an assistant
    message has null or empty string content alongside tool_calls. Replacing it
    with a single space satisfies those APIs without affecting model behaviour.
    """
    normalized = []
    for msg in messages:
        if isinstance(msg, Message):
            if msg.role == "assistant" and msg.tool_calls and not msg.content:
                msg = msg.model_copy(update={"content": " "})
        elif (
            isinstance(msg, dict)
            and msg.get("role") == "assistant"
            and msg.get("tool_calls")
            and not msg.get("content")
        ):
            msg = {**msg, "content": " "}
        normalized.append(msg)
    return normalized


def _is_context_window_error(e: Exception) -> bool:
    """
    Detect context window exceeded errors that LiteLLM doesn't properly classify.

    Some providers (notably Gemini) return context window errors as BadRequestError
    instead of ContextWindowExceededError. This predicate catches those cases
    by checking the error message content.

    Known error patterns:
    - Gemini: "input token count exceeds the maximum number of tokens allowed"
    - OpenAI: "context_length_exceeded" (usually caught as ContextWindowExceededError)
    - Anthropic: "prompt is too long" (usually caught as ContextWindowExceededError)
    """
    error_str = str(e).lower()

    # Common patterns indicating context/token limit exceeded
    context_patterns = [
        "token count exceeds",
        "context_length_exceeded",
        "context length exceeded",
        "maximum context length",
        "maximum number of tokens",
        "prompt is too long",
        "input too long",
        "exceeds the model's maximum context",
        "exceeds the context window",
    ]

    return any(pattern in error_str for pattern in context_patterns)


def _is_non_retriable_bad_request(e: Exception) -> bool:
    """
    Detect BadRequestErrors that are deterministic and should NOT be retried.

    These are configuration/validation errors that will always fail regardless
    of retry attempts. Retrying wastes time and resources.

    Note: Patterns must be specific enough to avoid matching transient errors
    like rate limits (e.g., "maximum of 100 requests" should NOT match).
    """
    error_str = str(e).lower()

    non_retriable_patterns = [
        # Tool count errors - be specific to avoid matching rate limits
        "tools are supported",  # "Maximum of 128 tools are supported"
        "too many tools",
        # Model/auth errors
        "model not found",
        "does not exist",
        "invalid api key",
        "authentication failed",
        "unauthorized",
        "unsupported parameter",
        "unsupported value",
        # OpenAI emits "Unknown parameter: 'foo'" for fields the model
        # endpoint doesn't accept (e.g. Responses-API `reasoning` sent to a
        # Chat-Completions-only model). These are config errors, not
        # transient — retrying just burns the worker for ~12 minutes.
        "unknown parameter",
        # Model capability mismatch
        "does not support multimodal",
        "is not a multimodal model",
    ]

    return any(pattern in error_str for pattern in non_retriable_patterns)


def _should_skip_retry(e: Exception) -> bool:
    """Combined check for all non-retriable errors."""
    return _is_context_window_error(e) or _is_non_retriable_bad_request(e)


@with_retry(
    max_retries=10,
    base_backoff=5,
    jitter=5,
    retry_on=(
        RateLimitError,
        Timeout,
        BadRequestError,
        ServiceUnavailableError,
        APIConnectionError,
        InternalServerError,
        BadGatewayError,
    ),
    skip_on=(ContextWindowExceededError,),
    skip_if=_should_skip_retry,
)
async def generate_response(
    model: str,
    messages: list[LitellmAnyMessage],
    tools: list[ChatCompletionToolParam],
    llm_response_timeout: int,
    extra_args: dict[str, Any],
    trajectory_id: str | None = None,
    stream: bool = False,
) -> ModelResponse:
    """
    Generate a response from the LLM with retry logic.

    Args:
        model: The model identifier to use
        messages: The conversation messages (input AllMessageValues or output Message)
        tools: Available tools for the model to call
        llm_response_timeout: Timeout in seconds for the LLM response
        extra_args: Additional arguments to pass to the completion call
        trajectory_id: Optional trajectory ID for tracking/tagging

    Returns:
        The model response
    """
    top_level_extra, extra_body = _split_extra_args(
        responses_args_to_completions(extra_args)
    )
    cached_messages = _with_cached_last_message(
        _with_cached_system_prompt(messages, model), model
    )
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": cached_messages,
        "timeout": llm_response_timeout,
        # Outer @with_retry owns retries — pin num_retries=0 so the LiteLLM
        # proxy doesn't retry on top, compounding caller × proxy attempts.
        # Caller's top_level_extra wins via the spread (later key overrides).
        "num_retries": 0,
        **top_level_extra,
    }

    if tools:
        kwargs["tools"] = _with_cached_tools(tools, model)

    # If LiteLLM proxy is configured, route completions through it and add tags.
    # Mirrors call_responses_api: rely on explicit api_base/api_key — some SDK /
    # model paths ignore litellm.use_litellm_proxy for acompletion(), which would
    # send requests direct-to-provider (no proxy logging / spend attribution).
    #
    # Tags are sent two ways for redundancy:
    #   1. `metadata.tags` — the documented path. Lands in StandardLoggingPayload
    #      iff the proxy key has admin metadata `allow_client_tags: true`.
    #      In litellm 1.83.10+ a default-deny strip wipes this otherwise
    #      (BerriAI/litellm@0e62add).
    #   2. `extra_headers` — harvested by the proxy via
    #      `litellm_settings.extra_spend_tag_headers` in config.yaml. Bypasses
    #      the strip because header tags are appended in `_get_request_tags`
    #      AFTER the strip runs (litellm_logging.py:5249-5272). This is what
    #      actually makes tags flow today.
    # Docs: https://docs.litellm.ai/docs/proxy/cost_tracking
    if settings.LITELLM_PROXY_API_BASE and settings.LITELLM_PROXY_API_KEY:
        kwargs.setdefault("api_base", settings.LITELLM_PROXY_API_BASE)
        kwargs.setdefault("api_key", settings.LITELLM_PROXY_API_KEY)
        # Tags ship via HTTP headers only — litellm 1.83.10 added a
        # default-deny strip that wipes body-supplied `metadata.tags` unless
        # the key has admin metadata `allow_client_tags: true` (we don't).
        # Header-derived tags bypass the strip via the proxy's
        # `extra_spend_tag_headers` allowlist (see rl-studio/infra/litellm/config.yaml).
        # call_id is stable across `with_retry` attempts of the same logical
        # call; attempt is 1-indexed and increments per retry. Together they
        # let Datadog distinguish unique logical calls from retry attempts
        # (e.g. unique_count(@call_id) vs count(*) for the 429 rate).
        call_id = llm_call_id_ctx.get()
        attempt_num = llm_attempt_ctx.get() if call_id else None
        hdrs = dict(kwargs.get("extra_headers") or {})
        hdrs.setdefault("service", "trajectory")
        if trajectory_id:
            hdrs.setdefault("trajectory_id", trajectory_id)
        if call_id:
            hdrs.setdefault("call_id", call_id)
            hdrs.setdefault("attempt", str(attempt_num))
        kwargs["extra_headers"] = hdrs

    if extra_body:
        kwargs["extra_body"] = extra_body

    if stream:
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
        stream_iter: Any = await acompletion(**kwargs)
        chunks: list[ModelResponse] = []

        async for chunk in stream_iter:
            chunks.append(chunk)
            # Per-delta progress logs only for external-agent-harness
            # providers (reasoning surfaced via `provider_specific_fields`).
            # These aren't model-internal chain-of-thought — they're
            # narrations the harness emits while it works (e.g. Gemini Deep
            # Research's thought summaries: "Origins and Architectural
            # Foundations — I am beginning by analyzing..."). Tag them as
            # `intermediate_output` so the UI can render them distinctly
            # from genuine `reasoning` logs from models like DeepSeek-R1
            # or the o-series. Native-reasoning streams go through
            # `delta.reasoning_content`, which stream_chunk_builder
            # aggregates and singleshot_agent logs once at end-of-call as
            # `reasoning` — no per-delta duplication from this path.
            intermediate_output = _extract_psf_reasoning_delta(chunk)
            if intermediate_output:
                logger.bind(message_type="intermediate_output").info(
                    intermediate_output
                )

        rebuilt = litellm.stream_chunk_builder(chunks, messages=messages)
        if rebuilt is None:
            raise RuntimeError("stream_chunk_builder returned None — empty stream")
        return ModelResponse.model_validate(rebuilt)

    response = await acompletion(**kwargs)
    return ModelResponse.model_validate(response)


@with_retry(
    max_retries=10,
    base_backoff=5,
    jitter=5,
    retry_on=(
        RateLimitError,
        Timeout,
        BadRequestError,
        ServiceUnavailableError,
        APIConnectionError,
        InternalServerError,
        BadGatewayError,
    ),
    skip_on=(ContextWindowExceededError,),
    skip_if=_should_skip_retry,
)
async def call_responses_api(
    model: str,
    messages: list[LitellmAnyMessage],
    tools: list[dict[str, Any]],
    llm_response_timeout: int,
    extra_args: dict[str, Any],
    trajectory_id: str | None = None,
    stream: bool = False,
) -> Any:
    """
    Generate a response using a provider's Responses API (e.g., web search) with retry logic.

    Uses litellm.aresponses() which is the native async version.

    Args:
        model: The model identifier to use (e.g., 'openai/gpt-4o')
        messages: The conversation messages
        tools: Tools for web search (e.g., [{"type": "web_search"}])
        llm_response_timeout: Timeout in seconds for the LLM response
        extra_args: Additional arguments (reasoning, etc.)
        trajectory_id: Optional trajectory ID for tracking/tagging

    Returns:
        The OpenAI responses API response object
    """
    top_level_extra, extra_body = _split_extra_args(extra_args)
    kwargs: dict[str, Any] = {
        "model": model,
        "input": messages,
        "tools": tools,
        "timeout": llm_response_timeout,
        # Outer @with_retry owns retries — pin num_retries=0 so the LiteLLM
        # proxy doesn't retry on top, compounding caller × proxy attempts.
        # Caller's top_level_extra wins via the spread (later key overrides).
        "num_retries": 0,
        **top_level_extra,
    }

    if settings.LITELLM_PROXY_API_BASE and settings.LITELLM_PROXY_API_KEY:
        kwargs["api_base"] = settings.LITELLM_PROXY_API_BASE
        kwargs["api_key"] = settings.LITELLM_PROXY_API_KEY
        # Tags ship via headers only — see generate_response. metadata.tags
        # is stripped by litellm 1.83.10's default-deny.
        call_id = llm_call_id_ctx.get()
        attempt_num = llm_attempt_ctx.get() if call_id else None
        hdrs = dict(kwargs.get("extra_headers") or {})
        hdrs.setdefault("service", "trajectory")
        if trajectory_id:
            hdrs.setdefault("trajectory_id", trajectory_id)
        if call_id:
            hdrs.setdefault("call_id", call_id)
            hdrs.setdefault("attempt", str(attempt_num))
        kwargs["extra_headers"] = hdrs

    if extra_body:
        kwargs["extra_body"] = extra_body

    if stream:
        kwargs["stream"] = True
        stream_iter: Any = await aresponses(**kwargs)
        completed_response = None
        async for event in stream_iter:
            if getattr(event, "type", None) == "response.completed":
                completed_response = getattr(event, "response", None)
        if completed_response is None:
            raise RuntimeError(
                "No response.completed event received from Responses API stream"
            )
        return completed_response

    response = await aresponses(**kwargs)
    return response
