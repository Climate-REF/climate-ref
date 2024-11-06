"""
Celery configuration.

The modules in this package are used to configure Celery for different environments.
The selected environment is determined by the `CELERY_CONFIG_MODULE` environment variable.
The default environment is `ref_celery.celeryconf.dev` which is used when running the app locally.
"""
