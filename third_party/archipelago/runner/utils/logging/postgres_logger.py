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

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any
from uuid import uuid4 as uuid

import asyncpg
import loguru

from runner.utils.settings import get_settings

settings = get_settings()

_log_queue: asyncio.Queue[dict[str, Any] | None] | None = None
_worker_task: asyncio.Task[None] | None = None
_init_lock: asyncio.Lock | None = None
_conn: asyncpg.Connection | None = None
_stopping: bool = False  # Block new enqueues during shutdown


def _generate_trajectory_log_id() -> str:
    return f"log_{uuid().hex}"


async def _log_worker() -> None:
    """
    Background worker that processes logs from the queue using a single connection.
    Exits when it receives a `None` sentinel or is cancelled.
    """
    global _conn, _log_queue

    if not settings.POSTGRES_URL or _log_queue is None:
        print("[Postgres Logger] POSTGRES_URL is not set or queue not initialized")
        return

    conn: asyncpg.Connection | None = None
    try:
        conn = await asyncpg.connect(
            dsn=settings.POSTGRES_URL,
            timeout=10,  # connect timeout
            command_timeout=10,  # per-command timeout
        )
        _conn = conn
        print("[Postgres Logger] Connected with single persistent connection")

        while True:
            try:
                log_data = await _log_queue.get()
            except asyncio.CancelledError:
                break

            if log_data is None:
                break

            if conn is None:
                print("[Postgres Logger] Connection not established")
                continue

            try:
                await conn.execute(
                    """
                    INSERT INTO trajectory_logs (
                        trajectory_log_id, trajectory_id, log_timestamp,
                        log_message, log_level, log_extra
                    )
                    VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                    """,
                    log_data["trajectory_log_id"],
                    log_data["trajectory_id"],
                    log_data["log_timestamp"],
                    log_data["log_message"],
                    log_data["log_level"],
                    log_data["log_extra"],
                )
            except Exception as e:
                print(f"[Postgres Logger] Error inserting log: {repr(e)}")
            finally:
                _log_queue.task_done()

    except Exception as e:
        print(f"[Postgres Logger] Worker error: {repr(e)}")
    finally:
        try:
            if conn is not None and not conn.is_closed():
                await conn.close()
        except (asyncio.CancelledError, RuntimeError) as e:
            print(
                f"[Postgres Logger] Suppressed close error during shutdown: {repr(e)}"
            )
        except Exception as e:
            print(f"[Postgres Logger] Error during connection close: {repr(e)}")
        finally:
            _conn = None
            print("[Postgres Logger] Connection closed")


async def _ensure_worker_started() -> None:
    """
    Ensure the background worker is running (with locking to prevent races).
    """
    global _log_queue, _worker_task, _init_lock

    if _init_lock is None:
        _init_lock = asyncio.Lock()

    if _log_queue is not None and _worker_task is not None and not _worker_task.done():
        return

    async with _init_lock:
        if _log_queue is None:
            _log_queue = asyncio.Queue(maxsize=1000)

        if _worker_task is None or _worker_task.done():
            _worker_task = asyncio.create_task(
                _log_worker(), name="postgres-logger-worker"
            )
            print("[Postgres Logger] Started background worker")


async def postgres_sink(message: loguru.Message) -> None:
    """
    Queue a log message to be written to the database.
    This is non-blocking and safe to call even during spikes.
    Expected `message.record` interface with fields used below.
    """
    global _stopping

    record = getattr(message, "record", None)
    if not record:
        return

    trajectory_id = record.get("extra", {}).get("trajectory_id")
    if not trajectory_id:
        return

    if not settings.POSTGRES_URL:
        return

    if _stopping:
        return

    try:
        await _ensure_worker_started()

        if _log_queue is None:
            print("[Postgres Logger] Queue not initialized")
            return

        log_data = {
            "trajectory_log_id": _generate_trajectory_log_id(),
            "trajectory_id": trajectory_id,
            "log_timestamp": record["time"],
            "log_message": record["message"],
            "log_level": record["level"].name,
            "log_extra": json.dumps(record["extra"], default=str),
        }

        try:
            _log_queue.put_nowait(log_data)
        except asyncio.QueueFull:
            print("[Postgres Logger] Queue full, dropping log")

    except Exception as e:
        print(f"[Postgres Logger] Error queuing log: {repr(e)}")


async def teardown_postgres_logger(timeout: float = 180.0) -> None:
    """
    Flush all pending logs and shut down the worker cleanly.
    Idempotent. Call from your app's shutdown path BEFORE the loop closes.
    """
    global _stopping, _log_queue, _worker_task

    _stopping = True

    if _log_queue is None or _worker_task is None:
        return

    # Wait for queue to drain with timeout
    try:
        with contextlib.suppress(RuntimeError):
            await asyncio.wait_for(_log_queue.join(), timeout=timeout)
    except TimeoutError:
        print(
            f"[Postgres Logger] Queue drain timed out after {timeout}s, forcing shutdown"
        )

    # Send shutdown signal
    with contextlib.suppress(RuntimeError):
        await _log_queue.put(None)

    # Wait for worker to finish
    try:
        await asyncio.wait_for(_worker_task, timeout=timeout)
    except (TimeoutError, asyncio.CancelledError):
        print("[Postgres Logger] Worker shutdown timed out, cancelling task")
        _worker_task.cancel()
        with contextlib.suppress(Exception):
            await _worker_task
    finally:
        _worker_task = None
        _log_queue = None
        _stopping = False
