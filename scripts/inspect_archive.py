#!/usr/bin/env python3
"""Summarize a tar archive without extracting its contents."""

from __future__ import annotations

import argparse
import re
import tarfile
from collections import Counter
from pathlib import Path


def classify_member(name: str) -> str:
    known_suffixes = (
        ("matrix.mtx.gz", "matrix.mtx.gz"),
        ("barcodes.tsv.gz", "barcodes.tsv.gz"),
        ("features.tsv.gz", "features.tsv.gz"),
        ("CMO_assignment_confidence_table.csv.gz", "CMO confidence table"),
        ("_CMO.csv.gz", "CMO metadata"),
        ("_geno.csv.gz", "genotype metadata"),
    )
    for suffix, label in known_suffixes:
        if name.endswith(suffix):
            return label
    return Path(name).suffix or "no extension"


def build_report(archive_path: Path) -> str:
    with tarfile.open(archive_path, mode="r:*") as archive:
        members = archive.getmembers()

    regular_files = [member for member in members if member.isfile()]
    type_counts = Counter(classify_member(member.name) for member in regular_files)
    sample_ids = sorted(
        {
            match.group(0)
            for member in regular_files
            if (match := re.match(r"GSM\d+", Path(member.name).name))
        }
    )
    payload_bytes = sum(member.size for member in regular_files)

    lines = [
        f"Archive: {archive_path}",
        f"Archive size: {archive_path.stat().st_size:,} bytes",
        f"Tar members: {len(members)}",
        f"Regular files: {len(regular_files)}",
        f"Sum of tar-reported file sizes: {payload_bytes:,} bytes",
        "Note: entries are listed only; no files were extracted.",
        "",
        f"Unique GSM accessions ({len(sample_ids)}):",
        *[f"  {sample_id}" for sample_id in sample_ids],
        "",
        "File types:",
        *[
            f"  {kind}: {count}"
            for kind, count in sorted(type_counts.items())
        ],
        "",
        "Archive member names:",
        *[f"  {member.name}" for member in members],
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List archive contents and summarize file names without extraction."
    )
    parser.add_argument("archive", type=Path, help="Path to a tar archive")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for saving the text inventory",
    )
    args = parser.parse_args()

    if not args.archive.is_file():
        parser.error(f"archive does not exist or is not a file: {args.archive}")

    try:
        report = build_report(args.archive)
    except (OSError, tarfile.TarError) as error:
        parser.error(f"could not read archive: {error}")

    print(report, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
