"""
View and ingest input datasets

The metadata from these datasets are stored in the database so that they can be used to determine
which executions are required for a given diagnostic without having to re-parse the datasets.

"""

import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from climate_ref.cli._utils import pretty_print_df
from climate_ref.database import ModelState
from climate_ref.datasets import get_dataset_adapter
from climate_ref.models import Dataset
from climate_ref.provider_registry import ProviderRegistry
from climate_ref.solver import solve_required_executions
from climate_ref.testing import fetch_sample_data
from climate_ref_core.dataset_registry import dataset_registry_manager, fetch_all_files
from climate_ref_core.datasets import SourceDatasetType

app = typer.Typer(help=__doc__)


@app.command(name="list")
def list_(
    ctx: typer.Context,
    source_type: Annotated[
        SourceDatasetType, typer.Option(help="Type of source dataset")
    ] = SourceDatasetType.CMIP6.value,  # type: ignore
    column: Annotated[list[str] | None, typer.Option()] = None,
    include_files: bool = typer.Option(False, help="Include files in the output"),
    limit: int = typer.Option(
        100,
        help=(
            "Limit the number of datasets (or files when using --include-files) to display to this number."
        ),
    ),
) -> None:
    """
    List the datasets that have been ingested

    The data catalog is sorted by the date that the dataset was ingested (first = newest).
    """
    database = ctx.obj.database

    adapter = get_dataset_adapter(source_type.value)
    data_catalog = adapter.load_catalog(database, include_files=include_files, limit=limit)

    if column:
        missing = set(column) - set(data_catalog.columns)
        if missing:

            def format_(columns: Iterable[str]) -> str:
                return ", ".join(f"'{c}'" for c in sorted(columns))

            logger.error(
                f"Column{'s' if len(missing) > 1 else ''} "
                f"{format_(missing)} not found in data catalog. "
                f"Choose from: {format_(data_catalog.columns)}"
            )
            raise typer.Exit(code=1)
        data_catalog = data_catalog[column].sort_values(by=column)

    pretty_print_df(data_catalog, console=ctx.obj.console)


@app.command()
def list_columns(
    ctx: typer.Context,
    source_type: Annotated[
        SourceDatasetType, typer.Option(help="Type of source dataset")
    ] = SourceDatasetType.CMIP6.value,  # type: ignore
    include_files: bool = typer.Option(False, help="Include files in the output"),
) -> None:
    """
    Print the current climate_ref configuration

    If a configuration directory is provided,
    the configuration will attempt to load from the specified directory.
    """
    database = ctx.obj.database

    adapter = get_dataset_adapter(source_type.value)
    data_catalog = adapter.load_catalog(database, include_files=include_files)

    for column in sorted(data_catalog.columns.to_list()):
        print(column)


@app.command()
def ingest(  # noqa
    ctx: typer.Context,
    file_or_directory: list[Path],
    source_type: Annotated[SourceDatasetType, typer.Option(help="Type of source dataset")],
    solve: Annotated[bool, typer.Option(help="Solve for new diagnostic executions after ingestion")] = False,
    dry_run: Annotated[bool, typer.Option(help="Do not ingest datasets into the database")] = False,
    n_jobs: Annotated[int | None, typer.Option(help="Number of jobs to run in parallel")] = None,
    skip_invalid: Annotated[
        bool, typer.Option(help="Ignore (but log) any datasets that don't pass validation")
    ] = True,
) -> None:
    """
    Ingest a directory of datasets into the database

    Each dataset will be loaded and validated using the specified dataset adapter.
    This will extract metadata from the datasets and store it in the database.

    A table of the datasets will be printed to the console at the end of the operation.
    """
    config = ctx.obj.config
    db = ctx.obj.database
    console = ctx.obj.console

    kwargs = {}

    if n_jobs is not None:
        kwargs["n_jobs"] = n_jobs

    # Create a data catalog from the specified file or directory
    adapter = get_dataset_adapter(source_type.value, **kwargs)

    for _dir in file_or_directory:
        _dir = Path(_dir).expanduser()
        logger.info(f"Ingesting {_dir}")

        if not _dir.exists():
            logger.error(f"File or directory {_dir} does not exist")
            continue

        # TODO: This assumes that all datasets are nc files.
        # THis is true for CMIP6 and obs4MIPs but may not be true for other dataset types in the future.
        if not _dir.rglob("*.nc"):
            logger.error(f"No .nc files found in {_dir}")
            continue

        try:
            data_catalog = adapter.find_local_datasets(_dir)
            data_catalog = adapter.validate_data_catalog(data_catalog, skip_invalid=skip_invalid)
        except Exception as e:
            logger.error(f"Error ingesting datasets from {_dir}: {e}")
            continue

        logger.info(
            f"Found {len(data_catalog)} files for {len(data_catalog[adapter.slug_column].unique())} datasets"
        )
        pretty_print_df(adapter.pretty_subset(data_catalog), console=console)

        # track stats for a given directory
        num_created_datasets = 0
        num_updated_datasets = 0
        num_unchanged_datasets = 0
        num_created_files = 0
        num_updated_files = 0
        num_removed_files = 0
        num_unchanged_files = 0

        for instance_id, data_catalog_dataset in data_catalog.groupby(adapter.slug_column):
            logger.debug(f"Processing dataset {instance_id}")
            with db.session.begin():
                if dry_run:
                    dataset = (
                        db.session.query(Dataset)
                        .filter_by(slug=instance_id, dataset_type=source_type)
                        .first()
                    )
                    if not dataset:
                        logger.info(f"Would save dataset {instance_id} to the database")
                else:
                    results = adapter.register_dataset(db, data_catalog_dataset)

                    if results.dataset_state == ModelState.CREATED:
                        num_created_datasets += 1
                    elif results.dataset_state == ModelState.UPDATED:
                        num_updated_datasets += 1
                    else:
                        num_unchanged_datasets += 1
                    num_created_files += len(results.files_added)
                    num_updated_files += len(results.files_updated)
                    num_removed_files += len(results.files_removed)
                    num_unchanged_files += len(results.files_unchanged)

        if not dry_run:
            ingestion_msg = (
                f"Datasets: {num_created_datasets}/{num_updated_datasets}/{num_unchanged_datasets}"
                " (created/updated/unchanged), "
                f"Files: "
                f"{num_created_files}/{num_updated_files}/{num_removed_files}/{num_unchanged_files}"
                " (created/updated/removed/unchanged)"
            )
            logger.info(ingestion_msg)

    if solve:
        solve_required_executions(
            config=config,
            db=db,
            dry_run=dry_run,
        )


@app.command(name="fetch-sample-data")
def _fetch_sample_data(
    force_cleanup: Annotated[bool, typer.Option(help="If True, remove any existing files")] = False,
    symlink: Annotated[
        bool, typer.Option(help="If True, symlink files into the output directory, otherwise perform a copy")
    ] = False,
) -> None:
    """
    Fetch the sample data for the given version.

    These data will be written into the test data directory.
    This operation may fail if the test data directory does not exist,
    as is the case for non-source-based installations.
    """
    # TODO: Remove
    fetch_sample_data(force_cleanup=force_cleanup, symlink=symlink)


@app.command(name="fetch-data")
def fetch_data(  # noqa: PLR0913
    ctx: typer.Context,
    registry: Annotated[
        str,
        typer.Option(help="Name of the data registry to use"),
    ],
    output_directory: Annotated[
        Path | None,
        typer.Option(help="Output directory where files will be saved"),
    ] = None,
    force_cleanup: Annotated[
        bool,
        typer.Option(help="If True, remove any existing files"),
    ] = False,
    symlink: Annotated[
        bool,
        typer.Option(help="If True, symlink files into the output directory, otherwise perform a copy"),
    ] = False,
    verify: Annotated[
        bool,
        typer.Option(help="Verify the checksums of the fetched files"),
    ] = True,
) -> None:
    """
    Fetch REF-specific datasets

    These datasets have been verified to have open licenses
    and are in the process of being added to Obs4MIPs.
    """
    config = ctx.obj.config
    db = ctx.obj.database

    # Setup the provider registry to register any dataset registries in the configured providers
    ProviderRegistry.build_from_config(config, db)

    if output_directory and force_cleanup and output_directory.exists():
        logger.warning(f"Removing existing directory {output_directory}")
        shutil.rmtree(output_directory)

    try:
        _registry = dataset_registry_manager[registry]
    except KeyError:
        logger.error(f"Registry {registry} not found")
        logger.error(f"Available registries: {', '.join(dataset_registry_manager.keys())}")
        raise typer.Exit(code=1)

    fetch_all_files(
        _registry,
        registry,
        output_directory,
        symlink=symlink,
        verify=verify,
    )
