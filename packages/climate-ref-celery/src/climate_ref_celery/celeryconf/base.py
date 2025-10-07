"""
Base configuration for Celery.

Other environments can use these settings as a base and override them as needed.
"""

import os

from climate_ref_core.env import get_available_cpu_count, get_env

env = get_env()

broker_url = env.str("CELERY_BROKER_URL", "redis://localhost:6379/1")
result_backend = env.str("CELERY_RESULT_BACKEND", broker_url)
broker_connection_retry_on_startup = True
worker_concurrency = os.environ.get("CELERY_WORKER_CONCURRENCY", get_available_cpu_count())

# Accept JSON and pickle as content
accept_content = ["json", "pickle"]
task_serializer = "pickle"
result_serializer = "pickle"
