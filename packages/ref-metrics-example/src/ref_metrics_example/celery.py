from ref_celery.app import create_celery_app
from ref_celery.tasks import register_celery_tasks

from ref_metrics_example import provider

app = create_celery_app("ref_metrics_example")

register_celery_tasks(app, provider)
