#!/usr/bin/env python3
"""Download and verify the published ImmuneAI atlas splits from Zenodo."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import quote


FILES = (
    ("INFLAMMATION_ATLAS_main_afterQC.h5ad", 17_229_436_935, "14fa25b5ff8d5703e13838b2f17274c5"),
    ("INFLAMMATION_ATLAS_validation_afterQC.h5ad", 2_519_082_825, "32fe8866b1de6a2b08ce9740de31c9c2"),
    ("INFLAMMATION_ATLAS_external_afterQC.h5ad", 2_085_105_651, "187862d15398187c874f62bd5ebf02ce"),
    ("INFLAMMATION_ATLAS_afterQC_sampleMetadata.csv", 149_243, "395e6f817fce9d5091f274d07c91359b"),
    ("Supplementary_Table_1.xlsx", 114_526, "6160aad1280c8177431afd14df524822"),
)
RECORD_ID = 14851902
GIB = 1024**3


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def existing_size(path: Path) -> int:
    if path.is_symlink():
        raise ValueError(f"refusing to write through a symlink: {path}")
    if not path.exists():
        return 0
    if not path.is_file():
        raise ValueError(f"expected a file path but found another item: {path}")
    return path.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "atlas",
        help="destination for the published files",
    )
    parser.add_argument(
        "--safety-buffer-gb",
        type=float,
        default=2.0,
        help="minimum free space to preserve after downloads",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show required space and planned downloads without writing files",
    )
    args = parser.parse_args()

    if args.safety_buffer_gb < 0:
        parser.error("safety-buffer-gb must not be negative")
    if shutil.which("curl") is None:
        parser.error("curl is required for resumable downloads")

    if not args.dry_run:
        args.data_dir.mkdir(parents=True, exist_ok=True)

    required_bytes = 0
    for name, expected_size, _ in FILES:
        destination = args.data_dir / name
        try:
            current_size = existing_size(destination)
        except ValueError as error:
            parser.error(str(error))
        if current_size > expected_size:
            parser.error(f"existing file is larger than expected; preserving it: {destination}")
        required_bytes += expected_size - current_size

    disk_path = args.data_dir
    while not disk_path.exists():
        disk_path = disk_path.parent
    available_bytes = shutil.disk_usage(disk_path).free
    buffer_bytes = int(args.safety_buffer_gb * GIB)
    print(f"Zenodo record: {RECORD_ID}")
    print(f"Destination: {args.data_dir}")
    print(f"Free space: {available_bytes / GIB:.2f} GiB")
    print(f"Download/resume required: {required_bytes / GIB:.2f} GiB")
    print(f"Safety buffer: {buffer_bytes / GIB:.2f} GiB")
    if available_bytes < required_bytes + buffer_bytes:
        parser.error("not enough free space for these files and the requested safety buffer")
    if args.dry_run:
        print("Dry run only; no atlas files downloaded.")
        return 0

    verification = []
    for name, expected_size, expected_md5 in FILES:
        destination = args.data_dir / name
        current_size = destination.stat().st_size if destination.exists() else 0
        if current_size == expected_size:
            actual_md5 = md5(destination)
            if actual_md5 != expected_md5:
                parser.error(f"checksum mismatch; preserving the existing file: {destination}")
            print(f"Already verified: {name}")
        else:
            url = (
                f"https://zenodo.org/api/records/{RECORD_ID}/files/"
                f"{quote(name)}/content"
            )
            print(f"Downloading/resuming: {name} ({current_size:,}/{expected_size:,} bytes)")
            subprocess.run(
                [
                    "curl",
                    "--fail",
                    "--location",
                    "--retry",
                    "5",
                    "--retry-delay",
                    "3",
                    "--retry-connrefused",
                    "--continue-at",
                    "-",
                    "--progress-bar",
                    "--output",
                    str(destination),
                    url,
                ],
                check=True,
            )
            actual_size = destination.stat().st_size
            if actual_size != expected_size:
                parser.error(
                    f"size mismatch for {name}: expected {expected_size}, received {actual_size}; "
                    "partial file is preserved for resume"
                )
            actual_md5 = md5(destination)
            if actual_md5 != expected_md5:
                parser.error(
                    f"checksum mismatch for {name}; file is preserved for inspection"
                )
            print(f"Verified: {name}")

        verification.append(
            {
                "file": name,
                "size_bytes": expected_size,
                "md5": expected_md5,
                "verified": True,
                "source": f"https://zenodo.org/records/{RECORD_ID}",
            }
        )

    project_root = Path(__file__).resolve().parents[1]
    report_path = project_root / "reports" / "atlas_download_verification.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "record_id": RECORD_ID,
                "license": "CC BY 4.0",
                "verified_at_utc": datetime.now(timezone.utc).isoformat(),
                "files": verification,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Verification report: {report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"Download failed; any partial file is retained so it can be resumed: {error}", file=sys.stderr)
        raise SystemExit(1)
