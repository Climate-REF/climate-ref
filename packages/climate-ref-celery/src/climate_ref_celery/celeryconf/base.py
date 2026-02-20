"""
Base configuration for Celery.

Other environments can use these settings as a base and override them as needed.

All settings can be overridden via environment variables prefixed with ``CELERY_``.
For example, ``CELERY_TASK_TIME_LIMIT=3600`` overrides ``task_time_limit``.
See the Helm chart README for per-provider override examples.
"""

import os

from climate_ref_core.env import get_available_cpu_count, get_env

env = get_env()

broker_url = env.str("CELERY_BROKER_URL", "redis://localhost:6379/1")
result_backend = env.str("CELERY_RESULT_BACKEND", broker_url)
broker_connection_retry_on_startup = True

accept_content = ["json", "pickle"]
task_serializer = "pickle"
result_serializer = "pickle"

# Number of concurrent worker processes to use
worker_concurrency = int(os.environ.get("CELERY_WORKER_CONCURRENCY", get_available_cpu_count()))

# Only prefetch one task at a time per worker process.
# Higher values cause multiple tasks to be lost/redelivered on a worker crash.
worker_prefetch_multiplier = int(os.environ.get("CELERY_WORKER_PREFETCH_MULTIPLIER", 1))  # noqa: PLW1508

# Acknowledge tasks AFTER execution completes, not when they are received.
# If a worker crashes mid-task the message returns to the broker for redelivery.
task_acks_late = True

# When a worker process is lost (OOM kill, segfault, SIGKILL) reject the task
# so the broker redelivers it instead of leaving it stuck in PENDING state.
task_reject_on_worker_lost = True

# Track STARTED state so we can distinguish "not started" from "worker died".
task_track_started = True

# Maximum number of retries for a task that fails or is killed.
# After this many retries the task is marked as permanently failed.
task_max_retries = int(os.environ.get("CELERY_TASK_MAX_RETRIES", 2))  # noqa: PLW1508

# Default hard kill after 6 hours
# Override per-provider via CELERY_TASK_TIME_LIMIT env var.
task_time_limit = int(os.environ.get("CELERY_TASK_TIME_LIMIT", 6 * 60 * 60))  # noqa: PLW1508

# Soft limit raises SoftTimeLimitExceeded, giving the task a chance to clean up.
task_soft_time_limit = int(os.environ.get("CELERY_TASK_SOFT_TIME_LIMIT", int(5.5 * 60 * 60)))

# Expire results after 48 hours to prevent unbounded Redis memory growth.
result_expires = int(os.environ.get("CELERY_RESULT_EXPIRES", 48 * 60 * 60))  # noqa: PLW1508

# visibility_timeout MUST be >= task_time_limit.
#
# Redis does not support real message acknowledgment.
# Instead, when a worker picks up a task, Celery moves it to an "unacked" sorted set.
# If the worker does not ACK within visibility_timeout seconds,
# Redis assumes the worker died and redelivers the task.
# If a task legitimately runs longer than this value,
# a second worker will start executing the same task concurrently.
#
# https://docs.celeryq.dev/en/v5.5.3/getting-started/backends-and-brokers/redis.html#id3
#
# With task_acks_late=True the ACK is sent after execution,
# so the task sits in the unacked set for its entire runtime.
# Setting visibility_timeout shorter than task_time_limit will cause duplicate executions.
_visibility_timeout = int(os.environ.get("CELERY_VISIBILITY_TIMEOUT", task_time_limit))
broker_transport_options = {
    "visibility_timeout": _visibility_timeout,
}


if _visibility_timeout < task_time_limit:
    raise ValueError(
        f"CELERY_VISIBILITY_TIMEOUT ({_visibility_timeout}s) must be >= CELERY_TASK_TIME_LIMIT "
        f"({task_time_limit}s) to prevent duplicate execution of long-running tasks."
    )
