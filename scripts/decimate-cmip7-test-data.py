"""
Decimate published CMIP7 files into the small fixtures in ``tests/test-data/cmip7``.

The fixtures keep every global and variable attribute of the published file,
so the dataset adapters are tested against real CMIP7 headers.
Only the data are cut down: a few grid cells and the first ``--months`` time steps.
The DRS directories are kept and the filename's time range is updated to match.

Usage::

    D=~/.esgf/MIP-DRS7/CMIP7/C4MIP/CCCma/CanESM5-1/esm-flat10/r1i1p2f1/glb
    uv run python scripts/decimate-cmip7-test-data.py \
        $D/mon/tas/tavg-h2m-hxy-u/g120/v20190429/tas_*_185001-190012.nc \
        $D/fx/areacella/ti-u-hxy-u/g120/v20190429/areacella_*.nc
"""

from __future__ import annotations

from pathlib import Path

import typer
import xarray as xr

app = typer.Typer()

KEPT_ENCODING = ("dtype", "_FillValue", "missing_value", "units", "calendar")
"""Encoding carried over from the source; layout settings refer to the original shape."""


def _drs_relative_path(path: Path) -> Path:
    """Return the path from its ``MIP-DRS7`` directory onwards."""
    parts = path.parts
    return Path(*parts[parts.index("MIP-DRS7") :])


def _time_range(ds: xr.Dataset) -> str:
    start, end = ds["time"].values[0], ds["time"].values[-1]
    return f"{start.strftime('%Y%m')}-{end.strftime('%Y%m')}"


def decimate(source: Path, output_root: Path, months: int, stride: int) -> Path:
    """Write a decimated copy of ``source`` below ``output_root`` and return its path."""
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    with xr.open_dataset(source, decode_times=time_coder) as ds:
        subset = ds.isel(lat=slice(None, None, stride), lon=slice(None, None, stride))
        if "time" in subset.dims:
            subset = subset.isel(time=slice(0, months))
        subset = subset.load()

    subset.attrs["history"] = (
        f"{subset.attrs.get('history', '')}\n"
        f"Decimated for the Climate-REF test suite (every {stride}th grid cell, first {months} months)."
    ).strip()

    relative = _drs_relative_path(source)
    name = relative.name
    if "time" in subset.dims:
        stem, _ = name.removesuffix(".nc").rsplit("_", 1)
        name = f"{stem}_{_time_range(subset)}.nc"
    output = output_root / relative.parent / name
    output.parent.mkdir(parents=True, exist_ok=True)

    for variable_name, variable in subset.variables.items():
        encoding = {key: value for key, value in variable.encoding.items() if key in KEPT_ENCODING}
        if variable.ndim:
            encoding.update(zlib=True, complevel=4)
        if str(variable_name).endswith("_bnds"):
            # Stop xarray from adding a ``coordinates`` attribute the source does not have
            encoding["coordinates"] = None
        elif "coordinates" in variable.encoding:
            encoding["coordinates"] = variable.encoding["coordinates"]
        variable.encoding = encoding
    subset.to_netcdf(output)
    return output


@app.command()
def main(
    sources: list[Path],
    output_root: Path = typer.Option(Path("tests/test-data/cmip7"), help="Root to write the DRS tree to"),
    months: int = typer.Option(24, help="Number of time steps to keep"),
    stride: int = typer.Option(16, help="Keep every n-th latitude and longitude"),
) -> None:
    """Decimate published CMIP7 files into test fixtures."""
    for source in sources:
        output = decimate(source.expanduser(), output_root, months=months, stride=stride)
        print(f"{output} ({output.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    app()
