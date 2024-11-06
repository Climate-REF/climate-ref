"""
Celery application for running metrics asynchronously across multiple workers


"""

import importlib.metadata

__version__ = importlib.metadata.version("ref_celery")
