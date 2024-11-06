"""
Base configuration for Celery.

Other environments can use these settings as a base and override them as needed.
"""

from environs import Env

env = Env()
env.read_env()

broker_url = env.str("CELERY_BROKER_URL", "redis://localhost:6379/1")
result_backend = env.str("CELERY_RESULT_BACKEND", broker_url)
