"""
Copy the canonical grey list into the ``climate_ref`` package.

``default_ignore_datasets.yaml`` at the repository root is the canonical copy.
It is served over HTTPS from the default branch, so already-released clients fetch it
from that path and it cannot move.
The same file also ships inside the ``climate_ref`` wheel, so an installation with no
network access and no writable cache still has a grey list to work from.

Two copies means they can drift, so this script keeps them identical.
It runs as a pre-commit hook and rewrites the packaged copy rather than merely
reporting a difference.
"""

import filecmp
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FILENAME = "default_ignore_datasets.yaml"
CANONICAL = REPO_ROOT / FILENAME
PACKAGED = REPO_ROOT / "packages" / "climate-ref" / "src" / "climate_ref" / FILENAME


def main() -> int:
    """
    Copy the canonical grey list over the packaged copy if they differ.

    Returns
    -------
    :
        0 if the copies already matched, 1 if the packaged copy was rewritten.
    """
    if not CANONICAL.is_file():
        print(f"Canonical grey list not found at {CANONICAL}", file=sys.stderr)
        return 1

    if PACKAGED.is_file() and filecmp.cmp(CANONICAL, PACKAGED, shallow=False):
        return 0

    PACKAGED.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CANONICAL, PACKAGED)
    print(f"Updated {PACKAGED.relative_to(REPO_ROOT)} from {FILENAME}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
