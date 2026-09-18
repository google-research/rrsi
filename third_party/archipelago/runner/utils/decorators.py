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
Utility decorators for the agent runner.
"""

import asyncio
import functools
import random
import time
import uuid
from collections.abc import Callable
from contextvars import ContextVar

from loguru import logger

from runner.utils.metrics import distribution, increment

# Per-logical-call correlation id, stable across `with_retry` attempts of the
# same wrapped function call. Downstream code (e.g. `runner.utils.llm`) reads
# this to tag each LiteLLM request so Datadog can dedupe retries:
#   unique_count(@call_id where status:429) / unique_count(@call_id)
# gives the "% of logical calls that hit a 429" instead of the inflated
# per-attempt rate.
llm_call_id_ctx: ContextVar[str | None] = ContextVar("llm_call_id", default=None)
llm_attempt_ctx: ContextVar[int] = ContextVar("llm_attempt", default=1)

# Trajectory-scoped "time to first successful LLM" SLI. Set in `modal_labs.run_agent`
# at function entry; consumed by `with_retry` to emit `studio.trajectory.time_to_first_llm_seconds`
# (success) or `studio.trajectory.first_llm_giveup_seconds` (terminal failure) on the
# first wrapped LLM call only. The flag prevents repeat emits on subsequent agent turns.
# Outside `run_agent` these vars remain unset (None / False) and the emit is a no-op,
# so non-trajectory `with_retry` usages (grading, tests) don't pollute trajectory metrics.
trajectory_started_at_ctx: ContextVar[float | None] = ContextVar(
    "trajectory_started_at", default=None
)
trajectory_tags_ctx: ContextVar[list[str] | None] = ContextVar(
    "trajectory_tags", default=None
)
first_llm_seen_ctx: ContextVar[bool] = ContextVar("first_llm_seen", default=False)


def with_retry(
    max_retries=3,
    base_backoff=1.5,
    jitter: float = 1.0,
    retry_on: tuple[type[Exception], ...] | None = None,
    skip_on: tuple[type[Exception], ...] | None = None,
    skip_if: Callable[[Exception], bool] | None = None,
):
    """
    This decorator is used to retry a function if it fails.
    It will retry the function up to the specified number of times, with a backoff between attempts.

    Args:
        max_retries: Maximum number of retry attempts
        base_backoff: Base backoff time in seconds
        jitter: Random jitter to add to backoff time
        retry_on: Tuple of exception types to retry on. If None, retries on all exceptions.
        skip_on: Tuple of exception types to never retry on, even if they match retry_on.
        skip_if: Predicate function that returns True if the exception should NOT be retried.
                 Useful for checking error messages (e.g., context window errors).
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # One id for the whole logical call; reused across retry attempts
            # so each retry of the same call carries the same `call_id` tag.
            # If we're already inside an enclosing `with_retry` (nested
            # decorators), inherit its call_id to keep correlation intact.
            outer_call_id = llm_call_id_ctx.get()
            call_id = outer_call_id or uuid.uuid4().hex
            call_id_token = (
                llm_call_id_ctx.set(call_id) if outer_call_id is None else None
            )

            # Per-attempt + cumulative timing for the retry summary log. Used by
            # Datadog to sum "wall-time wasted on retries" across logical calls.
            # We track failed-attempt time and final-attempt time separately so
            # downstream consumers can distinguish productive vs wasted work.
            sequence_start = time.perf_counter()
            total_backoff_seconds = 0.0
            failed_attempt_seconds = 0.0
            final_attempt_seconds = 0.0
            final_status = "failure"
            # Distinguishes a caller-handled skip (skip_on / skip_if / not in
            # retry_on) from a genuine "ran out of retries" terminal failure.
            # Both raise and leave `final_status == "failure"`, but the
            # first-LLM SLI must NOT count a skip as a give-up — the agent
            # loop will handle the exception (e.g. context-window → compact)
            # and try again, and the next attempt's outcome should be what
            # populates time_to_first_llm.
            terminal_outcome: str = "max_retries"
            final_error_class: str | None = None
            attempts_used = 0
            try:
                for attempt in range(1, max_retries + 1):
                    attempts_used = attempt
                    attempt_token = llm_attempt_ctx.set(attempt)
                    attempt_start = time.perf_counter()
                    try:
                        result = await func(*args, **kwargs)
                        final_attempt_seconds = time.perf_counter() - attempt_start
                        final_status = "success"
                        distribution(
                            "studio.llm.attempt_seconds",
                            final_attempt_seconds,
                            tags=[
                                f"func:{func.__name__}",
                                "outcome:success",
                            ],
                        )
                        return result
                    except Exception as e:
                        attempt_duration = time.perf_counter() - attempt_start
                        # Tentatively classify this attempt as failed; if we
                        # decide below not to retry, promote it to "final".
                        failed_attempt_seconds += attempt_duration
                        final_error_class = type(e).__name__

                        # "skipped" = caller-handled exception (skip_on /
                        # skip_if / not in retry_on). Distinct from "failure"
                        # so dashboards don't inflate the failure rate with
                        # expected non-retriable cases (e.g. context-window).
                        is_skipped = (
                            (skip_on is not None and isinstance(e, skip_on))
                            or (skip_if is not None and skip_if(e))
                            or (retry_on is not None and not isinstance(e, retry_on))
                        )
                        distribution(
                            "studio.llm.attempt_seconds",
                            attempt_duration,
                            tags=[
                                f"func:{func.__name__}",
                                f"outcome:{'skipped' if is_skipped else 'failure'}",
                                f"exc:{type(e).__name__}",
                            ],
                        )

                        if is_skipped:
                            failed_attempt_seconds -= attempt_duration
                            final_attempt_seconds = attempt_duration
                            terminal_outcome = "skipped"
                            raise

                        is_last_attempt = attempt >= max_retries
                        if is_last_attempt:
                            failed_attempt_seconds -= attempt_duration
                            final_attempt_seconds = attempt_duration
                            logger.error(
                                f"Error in {func.__name__}: {repr(e)}, after {max_retries} attempts (call_id={call_id})"
                            )
                            raise

                        backoff = base_backoff * (2 ** (attempt - 1))
                        jitter_delay = random.uniform(0, jitter) if jitter > 0 else 0
                        delay = backoff + jitter_delay
                        logger.warning(
                            f"Error in {func.__name__}: {repr(e)} (call_id={call_id}, attempt={attempt}/{max_retries})"
                        )
                        total_backoff_seconds += delay
                        await asyncio.sleep(delay)
                    finally:
                        llm_attempt_ctx.reset(attempt_token)
            finally:
                # Only the outermost retry wrapper emits the per-logical-call
                # summary; nested wrappers would double-count the same call.
                total_wall_seconds = time.perf_counter() - sequence_start
                if outer_call_id is None:
                    summary_tags = [
                        f"func:{func.__name__}",
                        f"status:{final_status}",
                    ]
                    if final_status != "success" and final_error_class is not None:
                        summary_tags.append(f"exc:{final_error_class}")
                    distribution(
                        "studio.llm.total_seconds",
                        total_wall_seconds,
                        tags=summary_tags,
                    )
                    distribution(
                        "studio.llm.attempts",
                        float(attempts_used),
                        tags=summary_tags,
                    )
                    if total_backoff_seconds > 0:
                        distribution(
                            "studio.llm.backoff_seconds",
                            total_backoff_seconds,
                            tags=summary_tags,
                        )
                    if failed_attempt_seconds > 0:
                        distribution(
                            "studio.llm.failed_attempt_seconds",
                            failed_attempt_seconds,
                            tags=summary_tags,
                        )
                    if attempts_used > 1:
                        increment(
                            "studio.llm.retries",
                            value=attempts_used - 1,
                            tags=summary_tags,
                        )
                    if final_status != "success":
                        increment(
                            "studio.llm.errors",
                            tags=summary_tags,
                        )

                    # Trajectory-scoped "time to first LLM" SLI — fires once per
                    # trajectory on the first wrapped LLM call that reaches a
                    # *resolved* outcome: either a success, or a max_retries
                    # give-up. A caller-handled skip (skip_on / skip_if / not in
                    # retry_on) is treated as "not yet resolved" — the agent
                    # loop will catch the exception and likely call again (e.g.
                    # context-window → compact → retry), so we leave the flag
                    # unset so the next attempt populates the SLI correctly.
                    # Without this, a compact-then-success flow would emit
                    # `first_llm_never_succeeded` followed by never emitting
                    # `time_to_first_llm_seconds`, systematically over-counting
                    # give-ups and under-counting successes.
                    trajectory_started_at = trajectory_started_at_ctx.get()
                    if (
                        trajectory_started_at is not None
                        and not first_llm_seen_ctx.get()
                        and terminal_outcome != "skipped"
                    ):
                        elapsed_since_trajectory_start = (
                            time.perf_counter() - trajectory_started_at
                        )
                        traj_tags = list(trajectory_tags_ctx.get() or [])
                        if final_status == "success":
                            distribution(
                                "studio.trajectory.time_to_first_llm_seconds",
                                elapsed_since_trajectory_start,
                                tags=traj_tags,
                            )
                            distribution(
                                "studio.trajectory.first_llm_attempts",
                                float(attempts_used),
                                tags=traj_tags,
                            )
                        else:
                            exc_tags = (
                                [f"exc:{final_error_class}"]
                                if final_error_class
                                else []
                            )
                            increment(
                                "studio.trajectory.first_llm_never_succeeded",
                                tags=traj_tags + exc_tags,
                            )
                            distribution(
                                "studio.trajectory.first_llm_giveup_seconds",
                                elapsed_since_trajectory_start,
                                tags=traj_tags + exc_tags,
                            )
                        first_llm_seen_ctx.set(True)

                # The structured retry-summary log keeps the same gate it had
                # before metrics existed: only log when something interesting
                # happened, since logs are the expensive ingestion path while
                # distributions aggregate server-side.
                should_emit = outer_call_id is None and (
                    attempts_used > 1 or final_status == "failure"
                )
                if should_emit:
                    logger.bind(
                        message_type="llm_retry_summary",
                        call_id=call_id,
                        func_name=func.__name__,
                        attempts=attempts_used,
                        retries_used=max(attempts_used - 1, 0),
                        total_backoff_seconds=round(total_backoff_seconds, 4),
                        failed_attempt_seconds=round(failed_attempt_seconds, 4),
                        final_attempt_seconds=round(final_attempt_seconds, 4),
                        total_wall_seconds=round(total_wall_seconds, 4),
                        final_status=final_status,
                        final_error_class=final_error_class,
                    ).info(
                        f"llm_retry_summary call_id={call_id} func={func.__name__} "
                        f"attempts={attempts_used} status={final_status} "
                        f"backoff_s={total_backoff_seconds:.2f} "
                        f"failed_attempt_s={failed_attempt_seconds:.2f} "
                        f"final_attempt_s={final_attempt_seconds:.2f}"
                    )
                if call_id_token is not None:
                    llm_call_id_ctx.reset(call_id_token)

        return wrapper

    return decorator
