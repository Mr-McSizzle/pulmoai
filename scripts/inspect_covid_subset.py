#!/usr/bin/env python3
"""Validate the COVID/healthy subset's matrix and label-file dimensions."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import tarfile
from collections import Counter
from pathlib import Path


def open_member_text(archive: tarfile.TarFile, member: tarfile.TarInfo):
    source = archive.extractfile(member)
    if source is None:
        raise OSError(f"could not read archive member: {member.name}")
    compressed = gzip.GzipFile(fileobj=source, mode="rb")
    return io.TextIOWrapper(compressed, encoding="utf-8", newline="")


def matrix_dimensions(archive: tarfile.TarFile, member: tarfile.TarInfo):
    with open_member_text(archive, member) as stream:
        for line in stream:
            if line.startswith("%") or not line.strip():
                continue
            dimensions = tuple(int(value) for value in line.split())
            if len(dimensions) != 3:
                raise ValueError(f"invalid Matrix Market dimensions in {member.name}")
            return dimensions
    raise ValueError(f"no Matrix Market dimensions found in {member.name}")


def read_lines(archive: tarfile.TarFile, member: tarfile.TarInfo) -> list[str]:
    with open_member_text(archive, member) as stream:
        return [line.rstrip("\r\n") for line in stream]


def read_assignments(
    archive: tarfile.TarFile, member: tarfile.TarInfo
) -> tuple[int, dict[str, dict[str, str]]]:
    with open_member_text(archive, member) as stream:
        rows = [
            row for row in csv.DictReader(stream) if row.get("Barcodes")
        ]
    return len(rows), {row["Barcodes"]: row for row in rows}


def validate(archive_path: Path) -> str:
    lines = [
        f"Archive: {archive_path}",
        "Reads Matrix Market headers and compressed text metadata; extracts no files.",
        "",
    ]

    with tarfile.open(archive_path, mode="r:*") as archive:
        members = {member.name: member for member in archive.getmembers()}
        matrix_members = sorted(
            (
                member
                for member in members.values()
                if member.name.endswith("_matrix.mtx.gz")
                and Path(member.name).name.startswith(("GSM8335813_", "GSM8335815_"))
            ),
            key=lambda member: member.name,
        )
        if len(matrix_members) != 2:
            raise ValueError(f"expected two selected matrices, found {len(matrix_members)}")

        for matrix in matrix_members:
            base = matrix.name.removesuffix("_matrix.mtx.gz")
            stem = Path(base).name
            shared_token = stem.split("_", 1)[1]
            barcode_member = members[f"{base}_barcodes.tsv.gz"]
            feature_member = members[f"{base}_features.tsv.gz"]
            metadata_member = next(
                member
                for member in members.values()
                if member.name.endswith("_CMO.csv.gz")
                and member.name.split("_", 1)[1].removesuffix("_CMO.csv.gz")
                == shared_token
            )
            assignment_member = next(
                member
                for member in members.values()
                if member.name.endswith("_CMO_assignment_confidence_table.csv.gz")
                and member.name.split("_", 1)[1].removesuffix(
                    "_CMO_assignment_confidence_table.csv.gz"
                )
                == shared_token
            )

            feature_count, barcode_count, nonzero_count = matrix_dimensions(archive, matrix)
            barcodes = read_lines(archive, barcode_member)
            features = read_lines(archive, feature_member)
            assignment_rows, assignments = read_assignments(archive, assignment_member)
            assigned_barcodes = set(assignments)
            matrix_barcodes = set(barcodes)
            assignment_only = sorted(assigned_barcodes - matrix_barcodes)
            matrix_only = sorted(matrix_barcodes - assigned_barcodes)

            with open_member_text(archive, metadata_member) as stream:
                metadata = list(csv.DictReader(stream))
            labels = sorted({row.get("disease", "") for row in metadata if row.get("disease")})
            metadata_by_cmo = {
                row.get("CMO_id", ""): row for row in metadata if row.get("CMO_id")
            }
            feature_types = Counter(
                fields[2] if len(fields) > 2 else "type not provided"
                for fields in (feature.split("\t") for feature in features)
            )
            donor_counts: Counter[tuple[str, str]] = Counter()
            other_assignments: Counter[str] = Counter()
            for barcode in matrix_barcodes & assigned_barcodes:
                assignment = assignments[barcode].get("Assignment", "")
                donor = metadata_by_cmo.get(assignment)
                if donor:
                    donor_counts[(donor.get("sample_id", ""), donor.get("disease", "label missing"))] += 1
                else:
                    other_assignments[assignment or "(blank assignment)"] += 1

            lines.extend(
                [
                    f"Library: {Path(matrix.name).name}",
                    f"  compressed matrix member: {matrix.size:,} bytes",
                    f"  matrix shape (features x barcodes): {feature_count:,} x {barcode_count:,}",
                    f"  declared nonzero entries: {nonzero_count:,}",
                    f"  feature rows: {len(features):,}",
                    f"  barcode rows: {len(barcodes):,}",
                    f"  assignment rows: {assignment_rows:,}",
                    f"  unique assignment barcodes: {len(assigned_barcodes):,}",
                    f"  duplicate matrix barcodes: {len(barcodes) - len(matrix_barcodes):,}",
                    f"  duplicate assignment barcodes: {assignment_rows - len(assigned_barcodes):,}",
                    f"  matching matrix/assignment barcodes: {len(matrix_barcodes & assigned_barcodes):,}",
                    f"  feature count matches matrix: {len(features) == feature_count}",
                    f"  barcode count matches matrix: {len(barcodes) == barcode_count}",
                    f"  matrix barcodes missing assignments: {len(matrix_only):,}",
                    f"  assignments without matrix barcodes: {len(assignment_only):,}",
                    *([f"    assignment-only examples: {', '.join(assignment_only[:5])}"] if assignment_only else []),
                    *([f"    matrix-only examples: {', '.join(matrix_only[:5])}"] if matrix_only else []),
                    f"  donor labels in paired CMO file: {', '.join(labels)}",
                    "  feature types:",
                    *[f"    {kind}: {count:,}" for kind, count in sorted(feature_types.items())],
                    "  matched barcodes assigned to labeled donors:",
                    *[
                        f"    {sample_id} ({label}): {count:,}"
                        for (sample_id, label), count in sorted(donor_counts.items())
                    ],
                    *(
                        ["  matched barcodes without a donor CMO label:"]
                        + [f"    {assignment}: {count:,}" for assignment, count in sorted(other_assignments.items())]
                        if other_assignments
                        else []
                    ),
                    "",
                ]
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check selected expression matrix and donor-label dimensions."
    )
    parser.add_argument("archive", type=Path, help="Path to the GSE248688 tar archive")
    parser.add_argument("--output", type=Path, help="Optional text report path")
    args = parser.parse_args()

    if not args.archive.is_file():
        parser.error(f"archive does not exist or is not a file: {args.archive}")

    try:
        report = validate(args.archive)
    except (OSError, EOFError, KeyError, StopIteration, ValueError, tarfile.TarError, gzip.BadGzipFile) as error:
        parser.error(f"could not validate selected files: {error}")

    print(report, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
