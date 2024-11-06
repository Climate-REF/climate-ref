"""Configuration for running celery locally"""

from loguru import logger

from .base import *  # noqa: F403

# Currently the dev environment is the same as the base environment
logger.info("Using dev configuration")
