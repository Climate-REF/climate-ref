"""
Script to calculate MD5 checksums for all .nc files in a specified directory.

This is used to update the registry with checksums of netCDF files,
ensuring data integrity.

"""

import hashlib
from pathlib import Path

import typer

app = typer.Typer()


@app.command()
def main(
    input_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Directory containing .nc files to process.",
    ),
):
    """
    Calculate and print the MD5 checksums for all .nc files in the INPUT_DIR.
    """
    contents = {}

    for file_path in input_dir.rglob("*.nc"):
        with open(file_path, "rb", buffering=0) as f:
            contents[file_path.relative_to(input_dir)] = hashlib.file_digest(f, "md5").hexdigest()

    if not contents:
        print(f"No .nc files found in {input_dir}")
        raise typer.Exit()

    sorted_keys = sorted(contents.keys(), key=str)

    for key in sorted_keys:
        print(f"{key} md5:{contents[key]}")


if __name__ == "__main__":
    app()
