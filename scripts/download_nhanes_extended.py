#!/usr/bin/env python3
"""Download NHANES XPT files for cycles 1999-2000 through 2011-2012.

Handles cycle-specific file naming conventions:
- 1999-2000 (no suffix, legacy lab names LAB13, LAB10, LAB18, LAB25, LAB10AM)
- 2001-2002 (_B suffix, L13_B, L10_B, L40_B, L25_B, L10AM_B)
- 2003-2004 (_C suffix, L13_C, L10_C, L40_C, L25_C, L10AM_C)
- 2005-2006 (_D suffix, TCHOL_D, HDL_D, GHB_D, BIOPRO_D, CBC_D, GLU_D)
- 2007-2008 (_E suffix, TCHOL_E, HDL_E, GHB_E, BIOPRO_E, CBC_E, GLU_E)
- 2009-2010 (_F suffix, TCHOL_F, HDL_F, GHB_F, BIOPRO_F, CBC_F, GLU_F)
- 2011-2012 (_G suffix, TCHOL_G, HDL_G, GHB_G, BIOPRO_G, CBC_G, GLU_G)

Enforces free disk space >= 3.0 GiB and checks SAS transport headers.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import certifi
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NHANES_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "nhanes"

CYCLE_SPECS = {
    "1999-2000": {
        "year": "1999",
        "files": [
            "DEMO.XPT", "BMX.XPT", "BPX.XPT", "BPQ.XPT", "DIQ.XPT", "MCQ.XPT",
            "SMQ.XPT", "ALQ.XPT", "PAQ.XPT", "LAB13.XPT", "LAB10.XPT", "LAB18.XPT",
            "LAB25.XPT", "LAB10AM.XPT",
        ],
    },
    "2001-2002": {
        "year": "2001",
        "files": [
            "DEMO_B.XPT", "BMX_B.XPT", "BPX_B.XPT", "BPQ_B.XPT", "DIQ_B.XPT", "MCQ_B.XPT",
            "SMQ_B.XPT", "ALQ_B.XPT", "PAQ_B.XPT", "L13_B.XPT", "L10_B.XPT", "L40_B.XPT",
            "L25_B.XPT", "L10AM_B.XPT",
        ],
    },
    "2003-2004": {
        "year": "2003",
        "files": [
            "DEMO_C.XPT", "BMX_C.XPT", "BPX_C.XPT", "BPQ_C.XPT", "DIQ_C.XPT", "MCQ_C.XPT",
            "SMQ_C.XPT", "ALQ_C.XPT", "PAQ_C.XPT", "L13_C.XPT", "L10_C.XPT", "L40_C.XPT",
            "L25_C.XPT", "L10AM_C.XPT",
        ],
    },
    "2005-2006": {
        "year": "2005",
        "files": [
            "DEMO_D.XPT", "BMX_D.XPT", "BPX_D.XPT", "BPQ_D.XPT", "DIQ_D.XPT", "MCQ_D.XPT",
            "SMQ_D.XPT", "ALQ_D.XPT", "PAQ_D.XPT", "TCHOL_D.XPT", "HDL_D.XPT", "GHB_D.XPT",
            "BIOPRO_D.XPT", "CBC_D.XPT", "GLU_D.XPT",
        ],
    },
    "2007-2008": {
        "year": "2007",
        "files": [
            "DEMO_E.XPT", "BMX_E.XPT", "BPX_E.XPT", "BPQ_E.XPT", "DIQ_E.XPT", "MCQ_E.XPT",
            "SMQ_E.XPT", "ALQ_E.XPT", "PAQ_E.XPT", "TCHOL_E.XPT", "HDL_E.XPT", "GHB_E.XPT",
            "BIOPRO_E.XPT", "CBC_E.XPT", "GLU_E.XPT",
        ],
    },
    "2009-2010": {
        "year": "2009",
        "files": [
            "DEMO_F.XPT", "BMX_F.XPT", "BPX_F.XPT", "BPQ_F.XPT", "DIQ_F.XPT", "MCQ_F.XPT",
            "SMQ_F.XPT", "ALQ_F.XPT", "PAQ_F.XPT", "TCHOL_F.XPT", "HDL_F.XPT", "GHB_F.XPT",
            "BIOPRO_F.XPT", "CBC_F.XPT", "GLU_F.XPT",
        ],
    },
    "2011-2012": {
        "year": "2011",
        "files": [
            "DEMO_G.XPT", "BMX_G.XPT", "BPX_G.XPT", "BPQ_G.XPT", "DIQ_G.XPT", "MCQ_G.XPT",
            "SMQ_G.XPT", "ALQ_G.XPT", "PAQ_G.XPT", "TCHOL_G.XPT", "HDL_G.XPT", "GHB_G.XPT",
            "BIOPRO_G.XPT", "CBC_G.XPT", "GLU_G.XPT",
        ],
    },
}


def check_disk(min_free_gb: float = 3.0) -> float:
    total, used, free = shutil.disk_usage(PROJECT_ROOT)
    free_gb = free / (1024**3)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"ABORT: Free disk space ({free_gb:.2f} GiB) is below minimum {min_free_gb} GiB threshold!"
        )
    return free_gb


def download_xpt(year: str, filename: str, dest_path: Path) -> tuple[bool, str]:
    if dest_path.exists() and dest_path.stat().st_size > 500:
        with open(dest_path, "rb") as f:
            header = f.read(30)
            if header.startswith(b"HEADER RECORD"):
                size_kb = dest_path.stat().st_size / 1024
                return True, f"already exists ({size_kb:.1f} KB, valid SAS XPT)"

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")

    urls = [
        f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{filename}",
        f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{filename.lower()}",
        f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{filename.upper()}",
    ]

    last_err = ""
    for url in urls:
        try:
            r = requests.get(url, stream=True, verify=certifi.where(), timeout=30)
            if r.status_code == 200:
                with open(temp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=128 * 1024):
                        if chunk:
                            f.write(chunk)
                with open(temp_path, "rb") as f:
                    magic = f.read(30)
                if magic.startswith(b"HEADER RECORD"):
                    temp_path.rename(dest_path)
                    size_kb = dest_path.stat().st_size / 1024
                    return True, f"downloaded ({size_kb:.1f} KB)"
                else:
                    if temp_path.exists():
                        temp_path.unlink()
                    last_err = f"HTML instead of SAS XPT (header: {magic[:15]})"
            else:
                last_err = f"HTTP {r.status_code}"
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            last_err = str(e)

    return False, last_err


def main() -> None:
    t0 = time.time()
    free_gb = check_disk(3.0)
    print("=" * 80)
    print("DOWNLOADING NHANES 1999-2012 CORE MODULES FOR MORTALITY LINKAGE")
    print("=" * 80)
    print(f"Initial free disk space: {free_gb:.2f} GiB")

    downloaded = 0
    skipped = 0
    failed = []

    for cycle, spec in CYCLE_SPECS.items():
        year = spec["year"]
        files = spec["files"]
        cycle_dir = NHANES_RAW_DIR / cycle
        cycle_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- Cycle {cycle} (Year {year}) ---")

        for fn in files:
            dest = cycle_dir / fn
            ok, msg = download_xpt(year, fn, dest)
            if ok:
                if "already exists" in msg:
                    skipped += 1
                else:
                    downloaded += 1
                    print(f"  [OK] {fn:<15} {msg}")
            else:
                failed.append((cycle, fn, msg))
                print(f"  [FAIL] {fn:<15} {msg}")

    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    print(f"Files downloaded: {downloaded}")
    print(f"Files skipped (already cached): {skipped}")
    print(f"Failures: {len(failed)}")
    for cycle, fn, err in failed:
        print(f"  FAILED: {cycle}/{fn} -> {err}")

    check_disk(3.0)
    print(f"Elapsed time: {time.time() - t0:.1f}s")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
