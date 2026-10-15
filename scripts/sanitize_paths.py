#!/usr/bin/env python3
"""Sanitize absolute paths in tracked documentation, reports, and text files (Pass 7 H4).

Replaces:
- '/Users/nisarg/Desktop/ml project/' -> '' (relative path)
- '/Users/nisarg/Downloads/' -> '<DOWNLOADS>/'

Verifies:
- Every modified JSON file still parses cleanly with json.load.
- No binary files or source code logic are corrupted.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sanitize_paths")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TARGET_PREFIX_PROJECT = "/Users/nisarg/Desktop/ml project/"
TARGET_PREFIX_DOWNLOADS = "/Users/nisarg/Downloads/"

# File extensions allowed for path sanitization
TEXT_EXTENSIONS = {".json", ".txt", ".md", ".csv", ".rst"}


def get_tracked_files() -> list[Path]:
    res = subprocess.check_output(["git", "ls-files"], cwd=PROJECT_ROOT, text=True)
    files = []
    for line in res.splitlines():
        p = PROJECT_ROOT / line.strip()
        if p.is_file() and p.suffix.lower() in TEXT_EXTENSIONS:
            files.append(p)
    return files


def sanitize_file(file_path: Path) -> bool:
    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("Skipping non-UTF8 file: %s", file_path)
        return False

    if TARGET_PREFIX_PROJECT not in content and TARGET_PREFIX_DOWNLOADS not in content:
        return False

    modified = content.replace(TARGET_PREFIX_PROJECT, "").replace(TARGET_PREFIX_DOWNLOADS, "<DOWNLOADS>/")

    # If JSON, verify it is still valid JSON
    if file_path.suffix.lower() == ".json":
        try:
            json.loads(modified)
        except Exception as e:
            logger.error("JSON validation failed for %s after sanitization: %s", file_path, e)
            return False

    file_path.write_text(modified, encoding="utf-8")
    rel_path = file_path.relative_to(PROJECT_ROOT)
    logger.info("Sanitized absolute paths in: %s", rel_path)
    return True


def main() -> None:
    files = get_tracked_files()
    sanitized_count = 0
    for f in files:
        if sanitize_file(f):
            sanitized_count += 1

    logger.info("Path sanitization complete. Sanitized %d files.", sanitized_count)


if __name__ == "__main__":
    main()
