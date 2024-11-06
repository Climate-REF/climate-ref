"""
Generate virtual doc files for the mkdocs site.

This script can also be run directly to actually write out those files,
as a preview.

All credit to the creators of:
https://oprypin.github.io/mkdocs-gen-files/
and the docs at:
https://mkdocstrings.github.io/crystal/quickstart/migrate.html
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Iterable
from pathlib import Path

import mkdocs_gen_files
from attrs import define

ROOT_DIR = Path("api")
nav = mkdocs_gen_files.Nav()


@define
class PackageInfo:
    """
    Package information used to help us auto-generate the docs
    """

    full_name: str
    stem: str
    summary: str


def write_subpackage_pages(package: object) -> tuple[PackageInfo, ...]:
    """
    Write pages for the sub-packages of a package
    """
    sub_packages = []
    for _, name, is_pkg in pkgutil.walk_packages(package.__path__):
        subpackage_full_name = package.__name__ + "." + name
        sub_package_info = write_module_page(subpackage_full_name)
        sub_packages.append(sub_package_info)

    return tuple(sub_packages)


def get_write_file(package_full_name: str) -> Path:
    """Get directory in which to write the doc file"""
    write_dir = ROOT_DIR
    for sub_dir in package_full_name.split(".")[:-1]:
        write_dir = write_dir / sub_dir

    write_file = write_dir / package_full_name.split(".")[-1] / "index.md"

    return write_file


def create_sub_packages_table(sub_packages: Iterable[PackageInfo]) -> str:
    """Create the table summarising the sub-packages"""
    links = [f"[{sp.stem}][{sp.full_name}]" for sp in sub_packages]
    sub_package_header = "Sub-package"
    sub_package_width = max([len(v) for v in [sub_package_header, *links]])

    descriptions = [sp.summary for sp in sub_packages]
    description_header = "Description"
    description_width = max([len(v) for v in [description_header, *descriptions]])

    sp_column = [sub_package_header, *links]
    description_column = [description_header, *descriptions]

    sub_packages_table_l = []
    for i, (sub_package_value, description) in enumerate(zip(sp_column, description_column)):
        sp_padded = sub_package_value.ljust(sub_package_width)
        desc_padded = description.ljust(description_width)

        line = f"| {sp_padded} | {desc_padded} |"
        sub_packages_table_l.append(line)

        if i == 0:
            underline = f"| {'-'*sub_package_width} | {'-'*description_width} |"
            sub_packages_table_l.append(underline)

    sub_packages_table = "\n".join(sub_packages_table_l)
    return sub_packages_table


def write_module_page(
    package_full_name: str,
) -> PackageInfo:
    """
    Write the docs pages for a module/package
    """
    package = importlib.import_module(package_full_name)

    if hasattr(package, "__path__"):
        sub_packages = write_subpackage_pages(package)

    else:
        sub_packages = None

    package_name = package_full_name.split(".")[-1]

    write_file = get_write_file(package_full_name)

    nav[package_full_name.split(".")] = write_file.relative_to(ROOT_DIR).as_posix()

    with mkdocs_gen_files.open(write_file, "w") as fh:
        fh.write(f"# {package_full_name}\n")

        fh.write("\n")
        fh.write(f"::: {package_full_name}")

        if sub_packages:
            fh.write("\n")
            fh.write("## sub-packages \n")
            fh.write(f"{create_sub_packages_table(sub_packages)}\n")

    if package.__doc__ is None:
        summary = ""
    else:
        package_doc_split = package.__doc__.splitlines()
        if not package_doc_split[0]:
            summary = package_doc_split[1]
        else:
            summary = package_doc_split[0]

    return PackageInfo(package_full_name, package_name, summary)


write_module_page("ref_core")
write_module_page("ref_celery")
write_module_page("ref_metrics_example")

with mkdocs_gen_files.open(ROOT_DIR / "NAVIGATION.md", "w") as fh:
    print(list(nav.build_literate_nav()))
    fh.writelines(nav.build_literate_nav())
