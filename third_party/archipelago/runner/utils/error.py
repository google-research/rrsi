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

"""Error classification logic for distinguishing system errors from model errors."""

import httpx
from litellm.exceptions import (
    APIConnectionError,
    BadRequestError,
    ContextWindowExceededError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from mcp import McpError

from runner.utils.image_fetch import ImageFetchError


def is_system_error(exception: Exception) -> bool:
    """Determine if an exception represents a system error (retryable) vs model error.

    System errors are transient infrastructure issues that can be retried.
    Model errors are non-retryable failures like context overflow.

    Returns:
        True if the exception is a system error (should use ERROR status),
        False if it's a model error (should use FAILED status).
    """
    if isinstance(
        exception,
        (
            RateLimitError,
            Timeout,
            ServiceUnavailableError,
            APIConnectionError,
            InternalServerError,
        ),
    ):
        return True

    # BadRequestError could be either, check the error message
    if isinstance(exception, BadRequestError):
        error_str = str(exception).lower()
        if "exceeded your current quota" in error_str:
            return True  # System error
        # If it's context/token/multimodal related, it's a model error
        if (
            "context" in error_str
            or "token" in error_str
            or "is not a multimodal model" in error_str
            or "does not support multimodal" in error_str
        ):
            return False  # Model error
        return True  # System error (configuration/infrastructure issue)

    # Model errors (non-retryable)
    if isinstance(exception, ContextWindowExceededError):
        return False

    # ValueError is typically a configuration/validation error (non-retryable)
    if isinstance(exception, ValueError):
        return False

    # RuntimeError (e.g. SSE error events) — non-retryable
    if isinstance(exception, RuntimeError):
        return False

    # Image fetch failures are deterministic input/contract problems
    # (oversize, unsupported scheme, bad URL, 4xx). Retrying won't fix them.
    if isinstance(exception, ImageFetchError):
        return False

    # httpx HTTP errors — 5xx are retryable, 4xx are not
    if isinstance(exception, httpx.HTTPStatusError):
        return exception.response.status_code >= 500

    # httpx transport/connection errors are retryable system errors
    if isinstance(exception, (httpx.ConnectError, httpx.ReadError, httpx.WriteError)):
        return True

    # Unknown exceptions default to system error (safer to retry than fail permanently)
    return True


def is_fatal_mcp_error(exception: Exception) -> bool:
    """Determine if an exception is fatal and should immediately end the agent run.

    Fatal errors indicate the MCP session/connection is dead and cannot recover.
    Non-fatal errors can be reported to the LLM and the agent can continue.

    Args:
        exception: The exception to check.

    Returns:
        True if the error is fatal (session terminated, connection dead),
        False if the error is recoverable.
    """
    # Check for MCP-specific errors
    if isinstance(exception, McpError):
        # Check error code - handle both positive 32600 (current MCP bug) and
        # negative -32600 (JSON-RPC 2.0 standard) for forward compatibility
        error_code = (
            getattr(exception.error, "code", None)
            if hasattr(exception, "error")
            else None
        )
        if error_code in (32600, -32600):
            return True

        # Fallback to string matching for robustness
        if "Session terminated" in str(exception):
            return True

    # Check for FastMCP client disconnection errors
    if isinstance(exception, RuntimeError):
        error_str = str(exception)
        # FastMCP raises this when the client session has been closed/corrupted
        if "Client is not connected" in error_str:
            return True

    return False
