import dataclasses
import importlib.resources
import shutil
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any
from urllib import parse as urlparse

import pandas as pd
import pytest
import xarray as xr
from climate_ref_example import provider as example_provider
from pytest_regressions.data_regression import RegressionYamlDumper
from yaml.representer import SafeRepresenter

from climate_ref.config import Config
from climate_ref.database import Database
from climate_ref.datasets.cmip6 import CMIP6DatasetAdapter
from climate_ref.datasets.cmip6_parsers import parse_cmip6_complete, parse_cmip6_drs
from climate_ref.datasets.cmip7 import CMIP7DatasetAdapter
from climate_ref.datasets.cmip7_parsers import parse_cmip7_complete, parse_cmip7_drs
from climate_ref.datasets.obs4mips import Obs4MIPsDatasetAdapter
from climate_ref.models.metric_value import MetricValue
from climate_ref.provider_registry import _register_provider
from climate_ref_core.cmip6_to_cmip7 import (
    convert_cmip6_dataset,
    create_cmip7_filename,
    create_cmip7_path,
    format_cmip7_time_range,
)
from climate_ref_core.pycmec.controlled_vocabulary import CV

# Ignore the alembic folder
collect_ignore = ["src/climate_ref/migrations"]

# Add a representer for pandas Timestamps/NaT in the regression tests
RegressionYamlDumper.add_representer(
    pd.Timestamp, lambda dumper, data: SafeRepresenter.represent_datetime(dumper, data.to_pydatetime())
)
RegressionYamlDumper.add_representer(
    type(pd.NaT), lambda dumper, data: SafeRepresenter.represent_none(dumper, data)
)


def _clone_db(target_db_url: str, template_db_path: Path) -> None:
    split_url = urlparse.urlsplit(target_db_url)
    target_db_path = Path(split_url.path[1:])
    target_db_path.parent.mkdir(parents=True)

    shutil.copy(template_db_path, target_db_path)


@pytest.fixture
def db(config) -> Generator[Database, None, None]:
    database = Database.from_config(config, run_migrations=True)
    yield database
    database.close()


@pytest.fixture(scope="session")
def cmip7_aft_cv() -> CV:
    cv_file = str(importlib.resources.files("climate_ref_core.pycmec") / "cv_cmip7_aft.yaml")

    return CV.load_from_file(cv_file)


@pytest.fixture(scope="session")
def prepare_db(cmip7_aft_cv):
    MetricValue.register_cv_dimensions(cmip7_aft_cv)


@pytest.fixture(scope="session")
def db_seeded_template(tmp_path_session, cmip6_data_catalog, obs4mips_data_catalog, prepare_db) -> Path:
    template_db_path = tmp_path_session / "climate_ref_template_seeded.db"

    config = Config.default()  # This is just dummy config
    database = Database(f"sqlite:///{template_db_path}")
    database.migrate(config)

    # Seed the CMIP6 sample datasets
    adapter = CMIP6DatasetAdapter()
    with database.session.begin():
        for instance_id, data_catalog_dataset in cmip6_data_catalog.groupby(adapter.slug_column):
            adapter.register_dataset(database, data_catalog_dataset)

    # Seed the obs4MIPs sample datasets
    adapter_obs = Obs4MIPsDatasetAdapter()
    with database.session.begin():
        for instance_id, data_catalog_dataset in obs4mips_data_catalog.groupby(adapter_obs.slug_column):
            adapter_obs.register_dataset(database, data_catalog_dataset)

    with database.session.begin():
        _register_provider(database, example_provider)

    database.close()

    return template_db_path


@pytest.fixture
def db_seeded(db_seeded_template, config) -> Database:
    # Copy the template database to the location in the config
    _clone_db(config.db.database_url, db_seeded_template)

    database = Database.from_config(config, run_migrations=True)
    yield database
    database.close()


@pytest.fixture
def cmip7_converted_file(sample_data_dir, tmp_path) -> Path:
    """
    Convert a CMIP6 file to CMIP7 format and save it.

    Uses the cmip6_to_cmip7 converter from climate_ref_core.
    Returns the path to the converted file.
    """
    # Find a CMIP6 file from sample data
    cmip6_dir = sample_data_dir / "CMIP6"
    if not cmip6_dir.exists():
        pytest.skip("CMIP6 sample data not available")

    # Find the first .nc file
    nc_files = list(cmip6_dir.rglob("**/tas_*.nc"))
    if not nc_files:
        pytest.skip("No CMIP6 netCDF files found in sample data")

    cmip6_file = nc_files[0]

    # Open and convert to CMIP7
    # TODO: Have a higher-level utility function that handles the full conversion and saving
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    with xr.open_dataset(cmip6_file, decode_times=time_coder) as ds:
        ds_cmip7 = convert_cmip6_dataset(ds)

        # Create filename
        time_range = format_cmip7_time_range(ds_cmip7, ds_cmip7.attrs["frequency"])
        cmip7_filename = create_cmip7_filename(ds_cmip7.attrs, time_range=time_range)

        cmip7_drs_path = create_cmip7_path(ds_cmip7.attrs)

        # Save to a simple output directory (not full DRS structure for easier testing)
        output_dir = tmp_path / cmip7_drs_path
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / cmip7_filename

        # Save the converted file
        ds_cmip7.to_netcdf(output_file)

    return output_file


@dataclasses.dataclass(frozen=True)
class AdapterTestConfig:
    """Adapter-specific values needed by the parameterised tests."""

    adapter_cls: type
    complete_parser_patch_path: str
    default_instance_id: str
    default_source_id: str
    default_experiment_id: str
    successful_parsed_result: dict[str, Any]
    metadata_checks: dict[str, Any]

    # Parser dispatch
    complete_parser: Callable
    drs_parser: Callable
    parser_config_attr: str

    # Instance ID format
    instance_id_prefix: str
    instance_id_part_count: int

    # Round-trip columns to drop (not persisted through the DB)
    non_roundtrip_columns: list[str]

    # Complete parser core fields (must be non-NA after parsing)
    complete_parser_core_fields: list[str]


ADAPTER_CONFIGS = {
    "cmip6": AdapterTestConfig(
        adapter_cls=CMIP6DatasetAdapter,
        complete_parser_patch_path="climate_ref.datasets.cmip6.parse_cmip6_complete",
        default_instance_id="CMIP6.test.inst.model.exp.r1i1p1f1.Amon.tas.gn.v1",
        default_source_id="model",
        default_experiment_id="exp",
        successful_parsed_result={
            "frequency": "mon",
            "grid": "native atmosphere grid",
            "realm": "atmos",
            "branch_method": "standard",
            "start_time": "2000-01-01",
            "end_time": "2000-12-30",
        },
        metadata_checks={
            "frequency": "mon",
            "grid": "native atmosphere grid",
            "realm": "atmos",
            "branch_method": "standard",
        },
        complete_parser=parse_cmip6_complete,
        drs_parser=parse_cmip6_drs,
        parser_config_attr="cmip6_parser",
        instance_id_prefix="CMIP6",
        instance_id_part_count=10,
        non_roundtrip_columns=["time_range"],
        complete_parser_core_fields=[
            "source_id",
            "experiment_id",
            "variable_id",
            "frequency",
            "grid_label",
        ],
    ),
    "cmip7": AdapterTestConfig(
        adapter_cls=CMIP7DatasetAdapter,
        complete_parser_patch_path="climate_ref.datasets.cmip7.parse_cmip7_complete",
        default_instance_id="CMIP7.CMIP.NCAR.CESM3.hist.r1.glb.mon.tas.tavg-h2m-hxy-u.gn.v1",
        default_source_id="CESM3",
        default_experiment_id="hist",
        successful_parsed_result={
            "frequency": "mon",
            "realm": "atmos",
            "nominal_resolution": "100 km",
            "standard_name": "air_temperature",
            "long_name": "Near-Surface Air Temperature",
            "units": "K",
            "start_time": "2000-01-01",
            "end_time": "2000-12-30",
        },
        metadata_checks={
            "realm": "atmos",
            "nominal_resolution": "100 km",
            "standard_name": "air_temperature",
        },
        complete_parser=parse_cmip7_complete,
        drs_parser=parse_cmip7_drs,
        parser_config_attr="cmip7_parser",
        instance_id_prefix="CMIP7",
        instance_id_part_count=12,
        non_roundtrip_columns=["time_range", "branded_variable", "tracking_id"],
        complete_parser_core_fields=[
            "source_id",
            "experiment_id",
            "variable_id",
            "frequency",
            "grid_label",
        ],
    ),
}


@pytest.fixture(params=list(ADAPTER_CONFIGS.keys()))
def adapter_config(request) -> AdapterTestConfig:
    """Parameterised fixture providing adapter-specific test configuration."""
    return ADAPTER_CONFIGS[request.param]
