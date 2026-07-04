"""
Generate a CMIP7 parquet catalog from the existing CMIP6 parquet catalog.

Translates all attributes using the DReq variable mappings (cmip6_to_cmip7 module)
without requiring actual netCDF files on disk.

Usage::

    uv run python scripts/generate_cmip7_catalog.py \
        --input tests/test-data/esgf-catalog/cmip6_catalog.parquet \
        --output tests/test-data/esgf-catalog/cmip7_catalog.parquet
"""

from pathlib import Path

import pandas as pd
import typer
from loguru import logger

from climate_ref.datasets import CMIP7DatasetAdapter
from climate_ref_core.cmip6_to_cmip7 import (
    convert_cmip6_to_cmip7_attrs,
    create_cmip7_filename,
    create_cmip7_path,
)

# CMIP7 columns expected by the CMIP7DatasetAdapter
# branded_variable is a derived column added by find_local_datasets, not in dataset_specific_metadata
CMIP7_COLUMNS = [
    *CMIP7DatasetAdapter.dataset_specific_metadata,
    "branded_variable",
    *CMIP7DatasetAdapter.file_specific_metadata,
]


def _convert_row(row: pd.Series) -> dict | None:
    """
    Convert a single CMIP6 catalog row to CMIP7 attributes.

    Uses :func:`convert_cmip6_to_cmip7_attrs` for attribute translation,
    then builds the CMIP7 path, filename, and instance_id.

    Returns None if the variable compound has no DReq mapping.
    """
    cmip6_attrs = row.to_dict()

    try:
        attrs = convert_cmip6_to_cmip7_attrs(cmip6_attrs)
    except KeyError:
        return None

    # Carry over file-level and catalog metadata not handled by convert_cmip6_to_cmip7_attrs
    attrs["start_time"] = row.get("start_time")
    attrs["end_time"] = row.get("end_time")
    attrs["finalised"] = row.get("finalised", True)
    attrs["time_range"] = row.get("time_range")

    # Build CMIP7 DRS path and filename
    cmip7_dir = create_cmip7_path(attrs, version=attrs.get("version", "v1"))
    cmip7_filename = create_cmip7_filename(attrs, time_range=attrs.get("time_range"))
    attrs["path"] = f"{{data_dir}}/{cmip7_dir}/{cmip7_filename}"

    # Build instance_id in CMIP7 DRS format
    instance_parts = [*CMIP7DatasetAdapter.dataset_id_metadata, "version"]
    attrs["instance_id"] = "CMIP7." + ".".join(str(attrs[p]) for p in instance_parts)

    return attrs


# The canonical last monthly timestep of a full CMIP6/CMIP7 ``historical`` run.
# Only rows that already reach this end are lifted, which is what keeps the
# extension from perturbing any other diagnostic (see ``_extend_historical_end``).
FULL_HISTORICAL_END = "2014-12-16 12:00:00"


def _extend_historical_end(
    cmip7_catalog: pd.DataFrame,
    end_time: str,
    only_from: str = FULL_HISTORICAL_END,
) -> int:
    """
    Relabel full-length CMIP7 ``historical`` runs so their coverage reaches ``end_time``.

    Only rows whose ``end_time`` is *exactly* ``only_from`` (the canonical full
    historical end, 2014-12) are lifted. This deliberately narrow rule is what keeps
    the extension from perturbing other diagnostics: those rows already satisfy every
    diagnostic that requires coverage up to 2014-12 or earlier, so pushing their end
    further out never removes them and never newly-satisfies such a constraint. Only a
    diagnostic that requires coverage *beyond* 2014-12 (the fire diagnostic's 2002-2021
    window) sees any change.

    Shorter historical runs (e.g. a 5-year GFDL slice ending 1854) are left alone so
    they cannot suddenly satisfy another diagnostic's timerange. Fixed-frequency rows
    (``fx``, e.g. ``sftlf``) carry a null ``end_time`` and are untouched. Only
    ``end_time`` is changed -- ``start_time``, ``instance_id``, ``path`` and every other
    column are preserved.

    Parameters
    ----------
    cmip7_catalog
        The CMIP7 catalog to modify in place.
    end_time
        Target ``end_time`` string (``YYYY-MM-DD HH:MM:SS``).
    only_from
        Only rows whose ``end_time`` equals this value are extended.

    Returns
    -------
    int
        Number of rows whose ``end_time`` was extended.
    """
    is_historical = cmip7_catalog["experiment_id"] == "historical"
    is_full_run = cmip7_catalog["end_time"] == only_from
    mask = is_historical & is_full_run

    cmip7_catalog.loc[mask, "end_time"] = end_time
    return int(mask.sum())


def generate_cmip7_catalog(
    cmip6_catalog: pd.DataFrame,
    extend_historical_end: str | None = None,
) -> pd.DataFrame:
    """
    Convert a CMIP6 catalog DataFrame to CMIP7 format.

    Parameters
    ----------
    cmip6_catalog
        DataFrame from cmip6_catalog.parquet
    extend_historical_end
        If set, extend the ``end_time`` of every *full-length* CMIP7 ``historical``
        run (those ending at 2014-12) up to this timestamp (``YYYY-MM-DD HH:MM:SS``).
        Opt-in; when ``None`` the catalog copies the CMIP6 time bounds verbatim. See
        :func:`_extend_historical_end` for why only full runs are lifted.

    Returns
    -------
    :
        DataFrame with CMIP7 columns and translated attributes
    """
    converted_rows: list[dict] = []
    skipped_compounds: set[str] = set()

    for _, row in cmip6_catalog.iterrows():
        result = _convert_row(row)
        if result is None:
            compound = f"{row['table_id']}.{row['variable_id']}"
            skipped_compounds.add(compound)
            continue
        converted_rows.append(result)

    if skipped_compounds:
        logger.warning(
            f"Skipped {len(skipped_compounds)} compounds with no DReq mapping: {sorted(skipped_compounds)}"
        )

    if not converted_rows:
        logger.error("No rows were converted")
        return pd.DataFrame(columns=CMIP7_COLUMNS)

    cmip7_df = pd.DataFrame(converted_rows)

    # Ensure all expected columns are present
    for col in CMIP7_COLUMNS:
        if col not in cmip7_df.columns:
            cmip7_df[col] = pd.NA

    # Reorder columns to match expected layout
    cmip7_df = cmip7_df[CMIP7_COLUMNS]

    if extend_historical_end is not None:
        extended = _extend_historical_end(cmip7_df, extend_historical_end)
        logger.info(f"Extended {extended} historical rows to end_time {extend_historical_end!r}")

    logger.info(f"Converted {len(cmip7_df)} rows ({len(cmip6_catalog) - len(cmip7_df)} skipped)")
    logger.info(f"Unique instance_ids: {cmip7_df['instance_id'].nunique()}")
    logger.info(f"Unique variables: {sorted(cmip7_df['variable_id'].unique())}")
    logger.info(f"Unique branded_variables: {sorted(cmip7_df['branded_variable'].unique())}")

    return cmip7_df


app = typer.Typer()


@app.command()
def main(
    input_path: Path = typer.Option(
        Path("tests/test-data/esgf-catalog/cmip6_catalog.parquet"),
        "--input",
        exists=True,
        readable=True,
        help="Path to the CMIP6 parquet catalog",
    ),
    output_path: Path = typer.Option(
        Path("tests/test-data/esgf-catalog/cmip7_catalog.parquet"),
        "--output",
        help="Path for the output CMIP7 parquet catalog",
    ),
    extend_historical_end_year: int | None = typer.Option(
        None,
        "--extend-historical-end-year",
        "-E",
        help=(
            "Opt-in: extend every full-length CMIP7 `historical` row (those already "
            "ending 2014-12) so its coverage reaches December of this year (e.g. 2021 "
            "for the fire diagnostic). Shorter historical runs are left untouched. "
            "Only `end_time` is changed; fixed-frequency (fx) rows are left untouched. "
            "Omit to copy the CMIP6 time bounds verbatim."
        ),
    ),
) -> None:
    """Generate a CMIP7 parquet catalog from an existing CMIP6 parquet catalog."""
    logger.info(f"Reading CMIP6 catalog from {input_path}")
    cmip6_df = pd.read_parquet(input_path)
    logger.info(f"Read {len(cmip6_df)} rows with {cmip6_df['instance_id'].nunique()} datasets")

    extend_historical_end = (
        f"{extend_historical_end_year}-12-16 12:00:00" if extend_historical_end_year is not None else None
    )
    cmip7_df = generate_cmip7_catalog(cmip6_df, extend_historical_end=extend_historical_end)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmip7_df.to_parquet(output_path, index=False)
    logger.info(f"Wrote {len(cmip7_df)} rows to {output_path}")


if __name__ == "__main__":
    app()
