"""
Managing remote celery workers

This module is used to manage remote execution workers for the Climate REF project.
It is added to the `ref` command line interface if the `climate-ref-celery` package is installed.
"""

import importlib

import typer

from climate_ref_celery.app import create_celery_app
from climate_ref_celery.tasks import register_celery_tasks
from climate_ref_core.providers import DiagnosticProvider

app = typer.Typer(help=__doc__)


def import_provider(provider_package: str) -> DiagnosticProvider:
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
    # Note that this only works for provider packages that have a single provider.
    for entry_point in importlib.metadata.entry_points(group="climate-ref.providers"):
        if entry_point.dist is not None and entry_point.dist.name == provider_package:
            break
    else:
        typer.echo(
            f"Package '{provider_package}' is missing "
            '[project.entry-points."climate-ref.providers"] in pyproject.toml'
        )
        raise typer.Abort()

    # Get the provider from the provider_package
    try:
        provider = entry_point.load()
    except ModuleNotFoundError:
        typer.echo(f"Package '{provider_package}' not found")
        raise typer.Abort()
    except AttributeError:
        typer.echo("The package must define a 'provider' attribute")
        raise typer.Abort()
    if not isinstance(provider, DiagnosticProvider):
        typer.echo(f"Expected DiagnosticProvider, got {type(provider)}")
        raise typer.Abort()
    return provider


@app.command()
def start_worker(
    ctx: typer.Context,
    loglevel: str = typer.Option("info", help="Log level for the worker"),
    package: str | None = typer.Option(help="Package to import tasks from", default=None),
    extra_args: list[str] = typer.Argument(None, help="Additional arguments for the worker"),
) -> None:
    """
    Start a Celery worker for the given package.

    A celery worker enables the execution of tasks in the background on multiple different nodes.
    This worker will register a celery task for each diagnostic in the provider.
    The worker tasks can be executed by sending a celery task with the name
    '{package_slug}_{diagnostic_slug}'.

    The package must define a 'provider' variable that is an instance of 'ref_core.DiagnosticProvider'.
    """
    # Create a new celery app
    celery_app = create_celery_app("climate_ref_celery")

    if package:
        # Attempt to import the provider
        provider = import_provider(package)

        if hasattr(ctx.obj, "config"):
            # Configure the provider so that it knows where the conda environments are
            provider.configure(ctx.obj.config)

        # Wrap each diagnostics in the provider with a celery tasks
        register_celery_tasks(celery_app, provider)
        queue = provider.slug
    else:
        # This might need some tweaking in later PRs to pull in the appropriate tasks
        import climate_ref_celery.worker_tasks  # noqa: F401

        queue = "celery"

    argv = ["worker", "-E", f"--loglevel={loglevel}", f"--queues={queue}", *(extra_args or [])]
    celery_app.worker_main(argv=argv)


@app.command()
def list_config() -> None:
    """
    List the celery configuration
    """
    celery_app = create_celery_app("climate_ref_celery")

    print(celery_app.conf.humanize())


if __name__ == "__main__":  # pragma: no cover
    app()
