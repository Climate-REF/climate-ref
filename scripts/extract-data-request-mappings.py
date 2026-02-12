"""
Extract CMIP6-to-CMIP7 variable mappings from the CMIP7 Data Request (DReq).

Downloads the DReq release export JSON from GitHub and extracts a subset
containing only the variables used by Climate REF diagnostic providers.
The output is written to the climate-ref-core package data directory.

Usage
-----
    uv run python scripts/extract-data-request-mappings.py

We are suggested not to interact with the data request JSON directly,
but in this case its simpler to filter and extract without to depend on the DReq software.

References
----------
- CMIP7 Data Request Content: https://github.com/CMIP-Data-Request/CMIP7_DReq_Content
"""

from __future__ import annotations

import json
import pathlib
import urllib.request
from collections import Counter, defaultdict

import typer

from climate_ref_core.cmip6_to_cmip7 import DReqVariableMapping

app = typer.Typer(help="Extract CMIP6-to-CMIP7 variable mappings from the CMIP7 Data Request.")

DATA_REQUEST_VERSION = "v1.2.2.3"
DATA_REQUEST_URL = f"https://github.com/CMIP-Data-Request/CMIP7_DReq_Content/raw/refs/tags/{DATA_REQUEST_VERSION}/airtable_export/dreq_release_export.json"

DEFAULT_OUTPUT_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "packages"
    / "climate-ref-core"
    / "src"
    / "climate_ref_core"
    / "data"
    / "cmip6_cmip7_variable_map.json"
)

# Variable IDs used by Climate REF diagnostic providers.
# Sourced from DataRequirement definitions across ESMValTool, PMP, ILAMB, and example providers.
# A few extras are fine -- this is a broad filter to avoid shipping hundreds of unused mappings.
INCLUDED_VARIABLES = {
    # Supplementary / grid variables
    "areacella",
    "areacello",
    "sftlf",
    "sftof",
    "volcello",
    # Atmosphere: temperature, humidity, pressure
    "hurs",
    "hus",
    "pr",
    "psl",
    "ta",
    "tas",
    "tasmax",
    "tasmin",
    "ts",
    # Atmosphere: winds
    "tauu",
    "ua",
    "uas",
    "va",
    "vas",
    # Radiation
    "hfls",
    "hfss",
    "rlds",
    "rlus",
    "rlut",
    "rlutcs",
    "rsds",
    "rsdt",
    "rsus",
    "rsut",
    "rsutcs",
    # Clouds
    "cli",
    "clivi",
    "clt",
    "clwvi",
    # Ocean / sea ice
    "siconc",
    "tos",
    # Land / carbon
    "cVeg",
    "fco2antt",
    "treeFrac",
    "vegFrac",
    # Geopotential
    "zg",
}


def download_dreq() -> dict:
    """Download the DReq release export JSON."""
    typer.echo(f"Downloading DReq from {typer.style(DATA_REQUEST_URL, dim=True)}")
    with urllib.request.urlopen(DATA_REQUEST_URL) as resp:  # noqa: S310
        data = json.loads(resp.read())
    # The top-level key is the version string
    version_key = next(iter(data.keys()))
    typer.echo(f"Version: {typer.style(version_key, fg=typer.colors.CYAN, bold=True)}")
    return data[version_key]


def extract_mappings(dreq: dict) -> dict[str, DReqVariableMapping]:
    """
    Extract per-variable mappings from the DReq Variables table.

    Returns a dict keyed by CMIP6 compound name (e.g. ``"Amon.tas"``)
    with the global (``region=glb``) entry preferred when duplicates exist.
    """
    variables = dreq["Variables"]["records"]

    mappings: dict[str, DReqVariableMapping] = {}

    for _rec_id, rec in variables.items():
        cmip6_cn = rec.get("CMIP6 Compound Name", "")
        if not cmip6_cn or "." not in cmip6_cn:
            continue

        branded = rec.get("Branded Variable Name", "")
        cmip7_cn = rec.get("CMIP7 Compound Name", "")
        region = rec.get("Region", "glb")

        # Parse branded variable name -> out_name + branding_suffix
        parts = branded.split("_", 1)
        out_name = parts[0] if parts else ""
        branding_suffix = parts[1] if len(parts) > 1 else ""

        # Parse branding suffix components (format: temporal-vertical-horizontal-area)
        temporal_label, vertical_label, horizontal_label, area_label = branding_suffix.split("-")

        # Resolve realm from CMIP7 compound name (first component)
        cmip7_parts = cmip7_cn.split(".")
        realm = cmip7_parts[0] if cmip7_parts else ""

        # Parse CMIP6 compound name
        table_id, variable_id = cmip6_cn.split(".", 1)

        entry = DReqVariableMapping(
            table_id=table_id,
            variable_id=variable_id,
            cmip6_compound_name=cmip6_cn,
            cmip7_compound_name=cmip7_cn,
            branded_variable_name=branded,
            out_name=out_name,
            branding_suffix=branding_suffix,
            temporal_label=temporal_label,
            vertical_label=vertical_label,
            horizontal_label=horizontal_label,
            area_label=area_label,
            realm=realm,
            region=region,
        )

        # Check if a duplicate has been seen
        if cmip6_cn in mappings:
            raise ValueError(f"Duplicate CMIP6 compound name found: {cmip6_cn}")
        mappings[cmip6_cn] = entry

    return mappings


def print_summary(mappings: dict[str, DReqVariableMapping]) -> None:
    """Print a summary of what was extracted."""
    typer.echo(typer.style(f"\nTotal variables extracted: {len(mappings)}", bold=True))

    # Counts by table
    table_counts = Counter(v.table_id for v in mappings.values())
    typer.echo(typer.style("\nVariables by table:", bold=True))
    for table, count in sorted(table_counts.items(), key=lambda x: -x[1]):
        typer.echo(f"  {typer.style(table, fg=typer.colors.CYAN)}: {count}")

    # Counts by realm
    realm_counts = Counter(v.realm for v in mappings.values())
    typer.echo(typer.style("\nVariables by realm:", bold=True))
    for realm, count in sorted(realm_counts.items(), key=lambda x: -x[1]):
        typer.echo(f"  {typer.style(realm or '(empty)', fg=typer.colors.MAGENTA)}: {count}")

    # Check for overlapping variable_ids across tables
    by_variable_id: dict[str, list[str]] = defaultdict(list)
    for cn, v in mappings.items():
        by_variable_id[v.variable_id].append(cn)

    all_overlaps = {vid: cns for vid, cns in by_variable_id.items() if len(cns) > 1}
    detail_fields = ("out_name", "branding_suffix", "realm", "region")

    # Only show overlaps where at least one field varies across tables
    varying_overlaps = {}
    identical_count = 0
    for vid, cns in all_overlaps.items():
        has_variation = any(len({getattr(mappings[cn], field) for cn in cns}) > 1 for field in detail_fields)
        if has_variation:
            varying_overlaps[vid] = cns
        else:
            identical_count += 1

    typer.echo(
        typer.style("\nVariable IDs in multiple tables: ", bold=True)
        + f"{len(all_overlaps)} total, "
        + f"{typer.style(str(identical_count), fg=typer.colors.GREEN)} identical, "
        + f"{typer.style(str(len(varying_overlaps)), fg=typer.colors.RED)} with differences"
    )

    if varying_overlaps:
        typer.echo(typer.style("\nVariables with differing metadata across tables:", bold=True))
        for vid, cns in sorted(varying_overlaps.items()):
            typer.echo(f"\n  {typer.style(vid, fg=typer.colors.YELLOW, bold=True)} ({len(cns)} entries):")
            for field in detail_fields:
                unique_vals = sorted({getattr(mappings[cn], field) for cn in cns})
                if len(unique_vals) == 1:
                    typer.echo(f"    {typer.style(field, dim=True)}: {unique_vals[0]}")
                else:
                    typer.echo(
                        f"    {typer.style(field, dim=True)}: {', '.join(unique_vals)}  "
                        f"{typer.style('[VARIES]', fg=typer.colors.RED, bold=True)}"
                    )
            # Per-table breakdown
            for cn in sorted(cns):
                m = mappings[cn]
                typer.echo(
                    f"    {typer.style(cn, fg=typer.colors.GREEN)}: "
                    f"{m.branded_variable_name}  "
                    f"(realm={typer.style(m.realm, fg=typer.colors.MAGENTA)})"
                )


@app.command()
def main(
    output: pathlib.Path = typer.Option(
        DEFAULT_OUTPUT_PATH,
        "--output",
        "-o",
        help="Path to write the output JSON file.",
    ),
) -> None:
    """Download and extract CMIP6-to-CMIP7 variable mappings from the Data Request."""
    data_request = download_dreq()
    all_mappings = extract_mappings(data_request)

    # Filter to included tables and variables used by REF providers
    def _is_mon_or_fx(table_id: str) -> bool:
        return "mon" in table_id or "fx" in table_id

    filtered = {
        k: v
        for k, v in all_mappings.items()
        if _is_mon_or_fx(v.table_id) and v.variable_id in INCLUDED_VARIABLES
    }

    print_summary(filtered)

    output_data = {
        "_metadata": {
            "source": DATA_REQUEST_URL,
            "data_request_version": DATA_REQUEST_VERSION,
            "description": (
                "CMIP6-to-CMIP7 variable mappings extracted from the CMIP7 Data Request. "
                "Filtered to mon/fx tables and variables used by REF providers. "
                "Keyed by CMIP6 compound name (table_id.variable_id). "
                f"Regenerate with: uv run python scripts/{__file__} "
            ),
        },
        "variables": {k: v.to_dict() for k, v in sorted(filtered.items())},
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(output_data, f, indent=1, separators=(",", ": "))
        f.write("\n")

    typer.echo(
        f"\nWrote {typer.style(str(len(filtered)), fg=typer.colors.GREEN, bold=True)}"
        f" variable mappings to {typer.style(str(output), dim=True)}"
    )


if __name__ == "__main__":
    app()
