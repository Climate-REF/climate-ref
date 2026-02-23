"""
Script to convert CMIP6 datasets to CMIP7

This is a quick test script to check if diagnostic providers are ready for CMIP7 data.

"""

from pathlib import Path

import typer
import xarray as xr
from loguru import logger

from climate_ref_core.cmip6_to_cmip7 import (
    convert_cmip6_dataset,
    create_cmip7_filename,
    create_cmip7_path,
    format_cmip7_time_range,
)

app = typer.Typer()


def _convert_file(
    cmip6_path: Path,
    output_dir: Path,
) -> Path | None:
    """
    Convert a single CMIP6 file to CMIP7 format.

    Parameters
    ----------
    cmip6_path
        Path to the CMIP6 file
    output_dir
        Root directory for converted CMIP7 files

    Returns
    -------
    Path | None
        Path that the file was written to or None if something went wrong
    """
    try:
        ds: xr.Dataset = xr.open_dataset(cmip6_path)
    except Exception as e:
        logger.warning(f"Failed to open {cmip6_path}: {e}")
        return None

    try:
        ds_cmip7 = convert_cmip6_dataset(ds)

        # Build output path using CMIP7 DRS
        cmip7_subpath = create_cmip7_path(ds_cmip7.attrs)
        cmip7_dir = output_dir / cmip7_subpath
        cmip7_dir.mkdir(parents=True, exist_ok=True)

        # Build CMIP7 filename with time range
        time_range = format_cmip7_time_range(ds_cmip7, ds_cmip7.attrs["frequency"])
        cmip7_filename = create_cmip7_filename(ds_cmip7.attrs, time_range=time_range)
        cmip7_path = cmip7_dir / cmip7_filename

        # Only write if file doesn't already exist
        if not cmip7_path.exists():
            ds_cmip7.to_netcdf(cmip7_path)
            logger.debug(f"Wrote CMIP7 file: {cmip7_path}")
        else:
            logger.debug(f"CMIP7 file already exists: {cmip7_path}")

        return cmip7_path

    except Exception as e:
        logger.warning(f"Failed to convert {cmip6_path}: {e}")
        return None
    finally:
        ds.close()


@app.command()
def main(
    file_or_directory: list[Path],
    output_dir: Path = typer.Option(
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Output directory to store results",
    ),
):
    """
    Translate CMIP6 datasets into CMIP7-like

    This copies the contents of the file so be careful about which directories this is applied to.

    Examples
    --------
    ```
    # Translate all ACCESS-ESM1-5/historical/r1i1p1f1
    uv run scripts/create-cmip7-datasets.py \
        --output-dir /tmp \
        ~/.esgf/CMIP6/CMIP/CSIRO/ACCESS-ESM1-5/historical/r1i1p1f1/*mon \
        ~/.esgf/CMIP6/CMIP/CSIRO/ACCESS-ESM1-5/historical/r1i1p1f1/*fx
    ```
    """
    input_datasets: list[Path] = []

    for input_dir in file_or_directory:
        input_datasets.extend(input_dir.rglob("*.nc"))

    if not input_datasets:
        print("No .nc files found")
        raise typer.Exit()

    logger.info(f"Found {len(input_datasets)} datasets")

    success_count = 0
    failed_files = []

    for fname in sorted(input_datasets):
        output_fname = _convert_file(fname, output_dir=output_dir)
        if output_fname:
            success_count += 1
        else:
            failed_files.append(input_datasets)

    if len(failed_files) == 0:
        logger.success("All files processed successfully")
    else:
        bad_files = "\n -".join(failed_files)
        logger.warning(f"{len(failed_files)} files failed: \n{bad_files}")


if __name__ == "__main__":
    app()
