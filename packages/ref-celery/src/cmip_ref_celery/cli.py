"""
Managing remote execution workers
"""

import importlib

import typer

from cmip_ref_celery.app import create_celery_app
from cmip_ref_celery.tasks import register_celery_tasks
from cmip_ref_core.providers import MetricsProvider

app = typer.Typer(help=__doc__)


def import_provider(provider_package: str) -> MetricsProvider:
    """
    Import the provider from a given package.

    Parameters
    ----------
    provider_package:
        The package to import the provider from

    Raises
    ------
    typer.Abort
        If the provider_package does not define a 'provider' variable

        If the provider_package is not found

    Returns
    -------
    :
        The provider instance
    """
    try:
        imp = importlib.import_module(provider_package.replace("-", "_"))
    except ModuleNotFoundError:
        typer.echo(f"Package '{provider_package}' not found")
        raise typer.Abort()

    # Get the provider from the provider_package
    try:
        provider = imp.provider
    except AttributeError:
        typer.echo("The package must define a 'provider' attribute")
        raise typer.Abort()
    return provider


@app.command()
def start_worker(
    loglevel: str = typer.Option("info", help="Log level for the worker"),
    package: str | None = typer.Option(help="Package to import tasks from", default=None),
    extra_args: list[str] = typer.Argument(None, help="Additional arguments for the worker"),
) -> None:
    """
    Start a Celery worker for the given package.

    A celery worker enables the execution of tasks in the background on multiple different nodes.
    This worker will register a celery task for each metric in the provider.
    The worker tasks can be executed by sending a celery task with the name '{package_name}_{metric_name}'.

    The package must define a 'provider' variable that is an instance of 'ref_core.MetricsProvider'.
    """
    # Create a new celery app
    celery_app = create_celery_app("cmip_ref_celery")

    if package:
        # Attempt to import the provider
        provider = import_provider(package)
        queue = provider.slug

        # Wrap each metrics in the provider with a celery tasks
        register_celery_tasks(celery_app, provider)
    else:
        # Include the tasks that should be run on the core worker
        import cmip_ref_celery.worker_tasks  # noqa: F401

        queue = "celery"

    argv = ["worker", f"--loglevel={loglevel}", f"--queues={queue}", *(extra_args or [])]
    celery_app.worker_main(argv=argv)


@app.command()
def list_config() -> None:
    """
    List the celery configuration
    """
    celery_app = create_celery_app("cmip_ref_celery")

    print(celery_app.conf.humanize())


if __name__ == "__main__":  # pragma: no cover
    app()
