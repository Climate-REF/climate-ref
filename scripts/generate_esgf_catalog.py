r"""
Generate parquet catalogs from a local ESGF archive.

Usage::

    python scripts/generate_esgf_catalog.py \
        --cmip6-dir /path/to/CMIP6 \
        --obs4mips-dir /path/to/obs4MIPs \
        --output-dir tests/test-data/esgf-catalog/ \
        --strip-prefix /path/to/data/root
"""

from __future__ import annotations

import argparse
from pathlib import Path

from climate_ref.solve_helpers import generate_catalog, write_catalog_parquet


def main() -> None:
    """Generate parquet catalogs from a local ESGF archive."""
    parser = argparse.ArgumentParser(description="Generate parquet catalogs from a local ESGF archive")
    parser.add_argument(
        "--cmip6-dir",
        type=Path,
        action="append",
        default=[],
        help="Directory containing CMIP6 data (can be specified multiple times)",
    )
    parser.add_argument(
        "--obs4mips-dir",
        type=Path,
        action="append",
        default=[],
        help="Directory containing obs4MIPs data (can be specified multiple times)",
    )
    parser.add_argument(
        "--pmp-climatology-dir",
        type=Path,
        action="append",
        default=[],
        help="Directory containing PMP climatology data (can be specified multiple times)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for parquet catalog files",
    )
    parser.add_argument(
        "--strip-prefix",
        type=str,
        default=None,
        help="Path prefix to replace with {data_dir} for portability",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.cmip6_dir:
        print(f"Scanning CMIP6 directories: {args.cmip6_dir}")
        catalog = generate_catalog("cmip6", args.cmip6_dir, strip_path_prefix=args.strip_prefix)
        out_path = output_dir / "cmip6_catalog.parquet"
        write_catalog_parquet(catalog, out_path)
        print(f"  Wrote {len(catalog)} rows to {out_path}")

    if args.obs4mips_dir:
        print(f"Scanning obs4MIPs directories: {args.obs4mips_dir}")
        catalog = generate_catalog("obs4mips", args.obs4mips_dir, strip_path_prefix=args.strip_prefix)
        out_path = output_dir / "obs4mips_catalog.parquet"
        write_catalog_parquet(catalog, out_path)
        print(f"  Wrote {len(catalog)} rows to {out_path}")

    if args.pmp_climatology_dir:
        print(f"Scanning PMP climatology directories: {args.pmp_climatology_dir}")
        catalog = generate_catalog(
            "pmp-climatology", args.pmp_climatology_dir, strip_path_prefix=args.strip_prefix
        )
        out_path = output_dir / "pmp_climatology_catalog.parquet"
        write_catalog_parquet(catalog, out_path)
        print(f"  Wrote {len(catalog)} rows to {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
