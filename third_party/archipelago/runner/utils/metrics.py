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

"""Datadog metric submission for the agents runner.

Design goals (in priority order):
  1. NEVER crash the caller. A misconfigured DD client, a serialization
     bug, a network outage, or anything else cannot break the wrapping
     trajectory. The flow runs identically with or without metrics.
  2. NEVER block the event loop. Every emit returns to the caller
     immediately; serialization AND HTTP both happen on worker threads.
  3. Fire-and-forget. We do not wait for the DD intake; failed
     submissions are dropped at debug level.

Architecture:
  caller (event loop)
    └─→ asyncio.create_task(_safely(...))
          └─→ asyncio.to_thread(serialize + submit)
                └─→ DD ThreadedApiClient pool ──HTTPS──→ intake

Two layers of thread offload (asyncio default executor + the DD client's
own pool) is intentional belt-and-suspenders: even if a future SDK bump
regresses to blocking on the typed wrapper, the asyncio.to_thread layer
guarantees the event loop never sees it.

If there is no running event loop (e.g. sync caller, import-time emit),
we fall back to a direct call. ThreadedApiClient still routes HTTP via
its own pool in that path, so the network is non-blocking either way.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from datadog_api_client import Configuration, ThreadedApiClient
from datadog_api_client.v1.api.metrics_api import MetricsApi as MetricsApiV1
from datadog_api_client.v1.model.distribution_point import DistributionPoint
from datadog_api_client.v1.model.distribution_points_payload import (
    DistributionPointsPayload,
)
from datadog_api_client.v1.model.distribution_points_series import (
    DistributionPointsSeries,
)
from datadog_api_client.v2.api.metrics_api import MetricsApi
from datadog_api_client.v2.model.metric_intake_type import MetricIntakeType
from datadog_api_client.v2.model.metric_payload import MetricPayload
from datadog_api_client.v2.model.metric_point import MetricPoint
from datadog_api_client.v2.model.metric_series import MetricSeries
from loguru import logger

from runner.utils.settings import get_settings

settings = get_settings()

_api_client: ThreadedApiClient | None = None
_counts_api: MetricsApi | None = None
_dists_api: MetricsApiV1 | None = None

# Client init is wrapped so a bad API key, broken Configuration, or any other
# init-time exception cannot break this module's import.
try:
    if settings.DATADOG_API_KEY:
        _config = Configuration()
        _config.api_key["apiKeyAuth"] = settings.DATADOG_API_KEY
        _api_client = ThreadedApiClient(_config)
        _counts_api = MetricsApi(api_client=_api_client)
        _dists_api = MetricsApiV1(api_client=_api_client)
except Exception as e:
    logger.debug(f"Datadog metrics disabled — client init failed: {e}")
    _api_client = None
    _counts_api = None
    _dists_api = None

BASE_TAGS = [f"env:{settings.ENV.value}", "service:rl-studio-agents"]

# Strong refs for in-flight emits so they aren't GC'd mid-flight.
# asyncio.create_task only weakly references its Task; we add to the set and
# discard via done_callback to avoid an unbounded leak.
_inflight: set[asyncio.Task[None]] = set()


def _do_count(api: MetricsApi, metric: str, tags: list[str], value: int) -> None:
    """Serialize + submit a COUNT metric. Sync; intended for off-loop use."""
    series = MetricSeries(
        metric=metric,
        type=MetricIntakeType.COUNT,
        points=[MetricPoint(timestamp=int(time.time()), value=float(value))],
        tags=tags,
    )
    api.submit_metrics(body=MetricPayload(series=[series]))


def _do_dist(api: MetricsApiV1, metric: str, tags: list[str], v: float) -> None:
    """Serialize + submit a DISTRIBUTION metric. Sync; intended for off-loop use."""
    point = DistributionPoint(value=[float(int(time.time())), [float(v)]])
    series = DistributionPointsSeries(metric=metric, points=[point], tags=tags)
    api.submit_distribution_points(body=DistributionPointsPayload(series=[series]))


async def _safely(fn: Callable[..., Any], *args: Any) -> None:
    """Run fn off the loop; swallow any Exception so nothing reaches the caller.

    Note: this catches `Exception`, NOT `BaseException`. CancelledError /
    KeyboardInterrupt / SystemExit deliberately propagate — we never want
    metrics to swallow cancellation signals or shutdown.
    """
    try:
        await asyncio.to_thread(fn, *args)
    except Exception as e:
        logger.debug(f"DD emit dropped: {e}")


def _fire_and_forget(fn: Callable[..., Any], *args: Any) -> None:
    """Schedule fn(*args) so the caller returns immediately. Never raises.

    When a running event loop is available (the normal case in the agent
    runner), serialization + submission both run off the loop via
    `create_task → to_thread`. Otherwise (sync caller, import-time emit)
    we fall through to a direct call — ThreadedApiClient still offloads
    HTTP to its internal pool, so the network is non-blocking either way.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        try:
            fn(*args)
        except Exception as e:
            logger.debug(f"DD emit dropped (sync path): {e}")
        return

    try:
        task = loop.create_task(_safely(fn, *args))
        _inflight.add(task)
        task.add_done_callback(_inflight.discard)
    except Exception as e:
        logger.debug(f"DD emit task-schedule failed: {e}")


def increment(metric: str, tags: list[str] | None = None, value: int = 1) -> None:
    """Fire-and-forget COUNT submit. Returns immediately. Never raises."""
    if _counts_api is None:
        return
    try:
        all_tags = BASE_TAGS + (tags or [])
        _fire_and_forget(_do_count, _counts_api, metric, all_tags, value)
    except Exception as e:
        logger.debug(f"DD increment dropped: {e}")


def distribution(metric: str, value: float, tags: list[str] | None = None) -> None:
    """Fire-and-forget DISTRIBUTION submit. Returns immediately. Never raises.

    Use for any quantity where you want percentiles (latencies, payload sizes,
    token counts). DD aggregates server-side into a sketch and exposes
    p50/p90/p95/p99/avg/min/max/count on the same metric name.
    """
    if _dists_api is None:
        return
    try:
        all_tags = BASE_TAGS + (tags or [])
        _fire_and_forget(_do_dist, _dists_api, metric, all_tags, value)
    except Exception as e:
        logger.debug(f"DD distribution dropped: {e}")


@dataclass
class PhaseHandle:
    """Mutable handle yielded by `phase()` so the body can attach tags or
    emit related metrics under the same prefix without re-typing them.

    - `tag()`   adds a tag known only mid-block (e.g. result count).
    - `value()` emits a sibling distribution (e.g. `populate_objects`).
    - `count()` emits a sibling counter.
    """

    prefix: str
    name: str
    tags: list[str] = field(default_factory=list)

    def tag(self, value: str) -> None:
        self.tags.append(value)

    def value(
        self,
        suffix: str,
        v: float,
        extra_tags: list[str] | None = None,
    ) -> None:
        distribution(
            f"{self.prefix}.{suffix}",
            float(v),
            tags=self.tags + (extra_tags or []),
        )

    def count(
        self,
        suffix: str,
        extra_tags: list[str] | None = None,
        value: int = 1,
    ) -> None:
        increment(
            f"{self.prefix}.{suffix}",
            tags=self.tags + (extra_tags or []),
            value=value,
        )


@asynccontextmanager
async def phase(
    name: str,
    *,
    prefix: str,
    tags: list[str] | None = None,
    emit_errors: bool = True,
) -> AsyncIterator[PhaseHandle]:
    """Time one flow step end-to-end, auto-tagging success vs failure.

    Emits `{prefix}.{name}_seconds` (distribution) with `status:ok` on
    success or `status:error` on exception. When `emit_errors=True`,
    also increments `{prefix}.{name}_errors` tagged with `exc:<class>`
    on exception. The exception is always re-raised — wrap the call site
    in try/except if it should be swallowed.

    Cancellation and other BaseException-derived signals propagate
    without emitting, matching the existing teardown convention.
    """
    handle = PhaseHandle(prefix=prefix, name=name, tags=list(tags or []))
    start = time.perf_counter()
    try:
        yield handle
    except Exception as e:
        elapsed = time.perf_counter() - start
        distribution(
            f"{prefix}.{name}_seconds",
            elapsed,
            tags=handle.tags + ["status:error"],
        )
        if emit_errors:
            increment(
                f"{prefix}.{name}_errors",
                tags=handle.tags + [f"exc:{type(e).__name__}"],
            )
        raise
    else:
        distribution(
            f"{prefix}.{name}_seconds",
            time.perf_counter() - start,
            tags=handle.tags + ["status:ok"],
        )
