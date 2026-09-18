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

import loguru

from runner.utils.redis import redis_client
from runner.utils.settings import get_settings

settings = get_settings()


async def redis_sink(message: loguru.Message) -> None:
    record = message.record

    trajectory_id = record["extra"].get("trajectory_id")

    if not trajectory_id:
        return

    log_data = {
        "log_timestamp": record["time"].isoformat(),
        "log_level": record["level"].name,
        "log_message": record["message"],
        "log_extra": record["extra"],
    }

    stream_name = f"{settings.REDIS_STREAM_PREFIX}:{trajectory_id}"

    await redis_client.xadd(stream_name, {"log": json.dumps(log_data, default=str)})
    await redis_client.expire(stream_name, 43200)  # 12 hours
