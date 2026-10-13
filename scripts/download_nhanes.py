#!/usr/bin/env python3
"""Download NHANES XPT files for cycles 2013-2014 (H), 2015-2016 (I), and 2017-2018 (J).

Files downloaded per cycle into data/raw/nhanes/{cycle}/:
  DEMO, BMX, BPX, GHB, GLU, TCHOL, HDL, TRIGLY, BIOPRO, CBC, SMQ, ALQ, PAQ, DIQ, BPQ, MCQ.
Skips existing valid files, checks disk space (>3 GB), verifies SAS Transport magic bytes,
and reports any failures.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import certifi
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NHANES_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "nhanes"

CYCLES = {
    "2013-2014": ("2013", "H"),
    "2015-2016": ("2015", "I"),
    "2017-2018": ("2017", "J"),
}

FILE_NAMES = [
    "DEMO", "BMX", "BPX", "GHB", "GLU", "TCHOL", "HDL", "TRIGLY",
    "BIOPRO", "CBC", "SMQ", "ALQ", "PAQ", "DIQ", "BPQ", "MCQ"
]


def check_disk(min_free_gb: float = 3.0) -> float:
    total, used, free = shutil.disk_usage(PROJECT_ROOT)
    free_gb = free / (1024**3)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"ABORT: Free disk space ({free_gb:.2f} GiB) is below minimum {min_free_gb} GiB threshold!"
        )
    return free_gb


def download_file(cycle: str, start_yr: str, name: str, suffix: str, dest_path: Path) -> tuple[bool, str]:
    """Download a single XPT file, verifying it is a genuine SAS Transport file."""
    # Check if existing file is already a valid SAS XPT file
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        with open(dest_path, "rb") as f:
            header = f.read(30)
            if header.startswith(b"HEADER RECORD"):
                size_kb = dest_path.stat().st_size / 1024
                return True, f"already exists ({size_kb:.1f} KB, valid SAS XPT)"

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")

    # URL candidates to try
    candidate_urls = [
        f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{start_yr}/DataFiles/{name}_{suffix}.xpt",
        f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{start_yr}/DataFiles/{name}_{suffix}.XPT",
        f"https://wwwn.cdc.gov/Nchs/Nhanes/{cycle}/{name}_{suffix}.XPT",
    ]

    last_err = ""
    for url in candidate_urls:
        try:
            r = requests.get(url, stream=True, verify=certifi.where(), timeout=30)
            if r.status_code == 200:
                with open(temp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=128 * 1024):
                        if chunk:
                            f.write(chunk)

                # Verify file header is SAS XPT, not HTML error page
                with open(temp_path, "rb") as f:
                    magic = f.read(30)
                if magic.startswith(b"HEADER RECORD"):
                    temp_path.rename(dest_path)
                    size_kb = dest_path.stat().st_size / 1024
                    return True, f"downloaded ({size_kb:.1f} KB)"
                else:
                    if temp_path.exists():
                        temp_path.unlink()
                    last_err = f"Returned HTML instead of SAS XPT (header: {magic[:15]})"
            else:
                last_err = f"HTTP {r.status_code}"
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            last_err = str(e)

    return False, last_err


def main() -> None:
    t_start = time.time()
    free_gb = check_disk(3.0)
    print("=" * 80)
    print("PHASE 2: DOWNLOADING CDC NHANES DATA (2013-2018)")
    print("=" * 80)
    print(f"Initial free disk space: {free_gb:.2f} GiB")

    total_files = len(CYCLES) * len(FILE_NAMES)
    downloaded_count = 0
    skipped_count = 0
    failures = []

    for cycle, (start_yr, suffix) in CYCLES.items():
        cycle_dir = NHANES_RAW_DIR / cycle
        cycle_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- Cycle {cycle} (Start Year {start_yr}, Suffix _{suffix}) ---")

        for name in FILE_NAMES:
            filename = f"{name}_{suffix}.XPT"
            dest = cycle_dir / filename

            success, msg = download_file(cycle, start_yr, name, suffix, dest)
            if success:
                if "already exists" in msg:
                    skipped_count += 1
                else:
                    downloaded_count += 1
                print(f"  [OK] {filename:<16} : {msg}")
            else:
                failures.append((cycle, filename, msg))
                print(f"  [FAIL] {filename:<14} : ERROR: {msg}")

    elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    print(f"Total target files: {total_files}")
    print(f"Newly downloaded:   {downloaded_count}")
    print(f"Skipped (cached):   {skipped_count}")
    print(f"Failures:           {len(failures)}")
    print(f"Elapsed time:       {elapsed:.1f}s")
    print(f"Remaining disk:     {check_disk(3.0):.2f} GiB")

    if failures:
        print("\nFAILURE DETAILS:")
        for cycle, fn, err in failures:
            print(f"  - {cycle}/{fn}: {err}")
        raise RuntimeError(f"Download completed with {len(failures)} failures.")
    else:
        print("\nAll 48 NHANES XPT files successfully verified on disk!")


if __name__ == "__main__":
    main()
