#!/usr/bin/env python3
"""Summarize CMO donor labels and barcode assignments inside a tar archive."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import tarfile
from collections import Counter
from pathlib import Path


def read_csv_rows(archive: tarfile.TarFile, member: tarfile.TarInfo):
    source = archive.extractfile(member)
    if source is None:
        raise OSError(f"could not read archive member: {member.name}")
    with gzip.GzipFile(fileobj=source, mode="rb") as compressed:
        with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
            yield from csv.DictReader(text)


def summarize(archive_path: Path) -> str:
    lines = [
        f"Archive: {archive_path}",
        "Metadata is streamed from the archive; expression matrices are not read or extracted.",
        "",
        "CMO donor metadata:",
    ]

    with tarfile.open(archive_path, mode="r:*") as archive:
        members = archive.getmembers()
        cmo_members = [
            member for member in members if member.name.endswith("_CMO.csv.gz")
        ]
        assignment_members = [
            member
            for member in members
            if member.name.endswith("_CMO_assignment_confidence_table.csv.gz")
        ]
        metadata_by_prefix: dict[str, list[dict[str, str]]] = {}

        for member in cmo_members:
            prefix = member.name.removesuffix("_CMO.csv.gz")
            rows = list(read_csv_rows(archive, member))
            metadata_by_prefix[prefix] = rows
            lines.append(f"  {member.name}")
            if not rows:
                lines.append("    (no data rows)")
                continue
            for row in rows:
                sample_id = row.get("sample_id", "")
                cmo_id = row.get("CMO_id", "")
                disease = row.get("disease", "not stated in this file")
                lines.append(f"    {cmo_id}: {sample_id}; label={disease}")

        lines.extend(["", "Barcode assignment counts:"])
        for member in assignment_members:
            prefix = member.name.removesuffix(
                "_CMO_assignment_confidence_table.csv.gz"
            )
            metadata = metadata_by_prefix.get(prefix, [])
            metadata_by_cmo = {
                row.get("CMO_id", ""): row for row in metadata if row.get("CMO_id")
            }
            assignment_counts: Counter[str] = Counter()
            total_rows = 0
            for row in read_csv_rows(archive, member):
                assignment_counts[row.get("Assignment", "")] += 1
                total_rows += 1

            lines.append(f"  {member.name}")
            lines.append(f"    barcode rows: {total_rows:,}")
            for assignment, count in sorted(assignment_counts.items()):
                donor = metadata_by_cmo.get(assignment)
                if donor:
                    label = donor.get("disease", "not stated in metadata")
                    sample_id = donor.get("sample_id", "")
                    lines.append(
                        f"    {assignment} ({sample_id}; {label}): {count:,}"
                    )
                else:
                    lines.append(f"    {assignment or '(blank assignment)'}: {count:,}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect CMO labels and assignments without extracting archive files."
    )
    parser.add_argument("archive", type=Path, help="Path to a tar archive")
    parser.add_argument("--output", type=Path, help="Optional text report path")
    args = parser.parse_args()

    if not args.archive.is_file():
        parser.error(f"archive does not exist or is not a file: {args.archive}")

    try:
        report = summarize(args.archive)
    except (OSError, EOFError, tarfile.TarError, gzip.BadGzipFile) as error:
        parser.error(f"could not inspect metadata: {error}")

    print(report, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
