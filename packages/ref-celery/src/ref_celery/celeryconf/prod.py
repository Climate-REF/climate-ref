"""Configuration for running celery in production"""

from loguru import logger

from .base import *  # noqa: F403

# Currently the production environment is the same as the base environment
logger.info("Using prod configuration")
