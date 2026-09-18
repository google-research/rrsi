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

import json
from pathlib import Path
from threading import Lock
from typing import TextIO

import loguru
from loguru import logger

from runner.utils.settings import get_settings

settings = get_settings()

if not settings.FILE_LOG_PATH:
    raise ValueError("FILE_LOG_PATH must be set to use the file logger")

_file_handle: TextIO | None = None
_file_lock: Lock = Lock()


def _ensure_log_file() -> TextIO:
    global _file_handle

    if _file_handle is not None:
        return _file_handle

    if settings.FILE_LOG_PATH is None:
        raise ValueError("FILE_LOG_PATH must be set to use the file logger")

    log_path = Path(settings.FILE_LOG_PATH)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _file_handle = log_path.open("a", encoding="utf-8")
    return _file_handle


def file_sink(message: loguru.Message) -> None:
    record = message.record

    log_entry = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "extra": record["extra"],
    }

    try:
        with _file_lock:
            log_file = _ensure_log_file()
            log_file.write(json.dumps(log_entry, default=str) + "\n")
            log_file.flush()
    except Exception as exc:
        logger.debug(f"Error writing log to file: {exc!r}")


async def teardown_file_logger() -> None:
    global _file_handle

    with _file_lock:
        if _file_handle is None:
            return

        try:
            _file_handle.flush()
        finally:
            _file_handle.close()
            _file_handle = None
