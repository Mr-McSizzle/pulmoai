#!/usr/bin/env python3
"""Resilient byte-range chunked downloader for large dataset artifacts.

Bypasses cellular hotspot connection timeouts and carrier stream cuts by fetching
files in small, discrete byte ranges with automatic resume and per-chunk retries.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import certifi
import requests

logger = logging.getLogger("utils_download_chunks")


def download_file_range_chunks(
    url: str,
    dest_path: str | Path,
    chunk_size_mb: int = 4,
    max_retries: int = 5,
    timeout: float = 20.0,
) -> bool:
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        head = requests.head(url, verify=certifi.where(), allow_redirects=True, timeout=15)
        total_size = int(head.headers.get("content-length", 0))
    except Exception as e:
        logger.warning("HEAD request failed for %s: %s", url, e)
        return False

    if total_size <= 0:
        logger.warning("Invalid content-length for %s: %d", url, total_size)
        return False

    if dest.exists() and dest.stat().st_size == total_size:
        logger.info("File already fully downloaded: %s (%d bytes)", dest.name, total_size)
        return True

    chunk_bytes = chunk_size_mb * 1024 * 1024
    current_size = dest.stat().st_size if dest.exists() else 0
    if current_size > total_size:
        # File is corrupt or oversized, reset
        dest.unlink(missing_ok=True)
        current_size = 0

    mode = "ab" if current_size > 0 else "wb"
    logger.info("Downloading %s (resuming from %.1f / %.1f MB)...", dest.name, current_size / (1024*1024), total_size / (1024*1024))

    with open(dest, mode) as f:
        for start in range(current_size, total_size, chunk_bytes):
            end = min(start + chunk_bytes - 1, total_size - 1)
            headers = {"Range": f"bytes={start}-{end}"}

            chunk_success = False
            for attempt in range(max_retries):
                try:
                    r = requests.get(url, headers=headers, verify=certifi.where(), timeout=timeout)
                    if r.status_code in (200, 206):
                        f.write(r.content)
                        f.flush()
                        chunk_success = True
                        break
                    else:
                        time.sleep(1.0)
                except Exception as e:
                    time.sleep(1.5)

            if not chunk_success:
                logger.warning("Failed to fetch chunk %d-%d for %s after %d retries.", start, end, dest.name, max_retries)
                return False

    final_size = dest.stat().st_size
    if final_size == total_size:
        logger.info("Successfully downloaded %s (%.1f MB).", dest.name, final_size / (1024*1024))
        return True
    else:
        logger.warning("Download size mismatch for %s: expected %d, got %d", dest.name, total_size, final_size)
        return False
