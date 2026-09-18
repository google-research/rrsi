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

import sys

from loguru import logger

from runner.utils.settings import Environment, get_settings

settings = get_settings()


def setup_logger() -> None:
    logger.remove()

    if settings.DATADOG_LOGGING:
        # Datadog logger
        from .datadog_logger import datadog_sink  # import-check-ignore

        logger.debug("Adding Datadog logger")
        logger.add(datadog_sink, level="DEBUG", enqueue=True)

    if settings.REDIS_LOGGING:
        # Redis logger
        from .redis_logger import redis_sink  # import-check-ignore

        logger.debug("Adding Redis logger")
        logger.add(redis_sink, level="INFO")

    if settings.FILE_LOGGING:
        # File logger
        from .file_logger import file_sink  # import-check-ignore

        logger.debug("Adding File logger")
        logger.add(file_sink, level="DEBUG")

    if settings.POSTGRES_LOGGING:
        # Postgres logger
        from .postgres_logger import postgres_sink  # import-check-ignore

        logger.debug("Adding Postgres logger")
        logger.add(postgres_sink, level="INFO")

    if settings.ENV == Environment.LOCAL:
        # Local logger
        logger.add(
            sys.stdout,
            level="DEBUG",
            enqueue=True,
            backtrace=True,
            diagnose=True,
            colorize=True,
        )
    else:
        # Structured logger
        logger.add(
            sys.stdout,
            level="DEBUG",
            enqueue=True,
            backtrace=True,
            diagnose=True,
            serialize=True,
        )


async def teardown_logger() -> None:
    await logger.complete()

    if settings.POSTGRES_LOGGING:
        # Postgres logger
        from .postgres_logger import teardown_postgres_logger  # import-check-ignore

        logger.debug("Tearing down Postgres logger")
        await teardown_postgres_logger()

    if settings.FILE_LOGGING:
        from .file_logger import teardown_file_logger  # import-check-ignore

        logger.debug("Tearing down File logger")
        await teardown_file_logger()
