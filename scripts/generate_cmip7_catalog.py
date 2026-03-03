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


def generate_cmip7_catalog(cmip6_catalog: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a CMIP6 catalog DataFrame to CMIP7 format.

    Parameters
    ----------
    cmip6_catalog
        DataFrame from cmip6_catalog.parquet

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
) -> None:
    """Generate a CMIP7 parquet catalog from an existing CMIP6 parquet catalog."""
    logger.info(f"Reading CMIP6 catalog from {input_path}")
    cmip6_df = pd.read_parquet(input_path)
    logger.info(f"Read {len(cmip6_df)} rows with {cmip6_df['instance_id'].nunique()} datasets")

    cmip7_df = generate_cmip7_catalog(cmip6_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmip7_df.to_parquet(output_path, index=False)
    logger.info(f"Wrote {len(cmip7_df)} rows to {output_path}")


if __name__ == "__main__":
    app()
