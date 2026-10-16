#!/usr/bin/env python3
"""Cryptographic artifact verification for deployed PulmoAI models.

Reads models/checksums.sha256, verifies every deployed artifact,
and returns a structured status dictionary.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("artifact_verifier")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKSUMS_PATH = PROJECT_ROOT / "models" / "checksums.sha256"

MODULE_MAPPING = {
    "models/ensemble_benchmark_best.joblib": "pulmo_service",
    "models/lung_uncalibrated_dual_rule.joblib": "pulmo_service",
    "models/nhanes_risk_models.joblib": "risk_service",
    "models/future_risk_models_v3.joblib": "future_risk_service",
    "models/modality_gate_resnet18_v3.pt": "image_service",
    "models/medmnist/pneumoniamnist_resnet18.pt": "image_service",
    "models/medmnist/dermamnist_resnet18.pt": "image_service",
    "models/medmnist/retinamnist_resnet18.pt": "image_service",
    "models/medmnist/bloodmnist_resnet18.pt": "image_service",
    "models/medmnist/chestmnist_resnet18.pt": "image_service",
}


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file in streaming 64 KB chunks."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def verify_artifacts(
    checksums_path: Path | None = None,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """Verify all artifacts listed in checksums.sha256.

    Args:
        checksums_path: Path to the checksums.sha256 file (defaults to models/checksums.sha256).
        base_dir: Base directory for relative file paths (defaults to PROJECT_ROOT).

    Returns:
        Structured dictionary with per-artifact status and module-level health flags.
    """
    if checksums_path is None:
        checksums_path = DEFAULT_CHECKSUMS_PATH
    if base_dir is None:
        base_dir = PROJECT_ROOT

    if not checksums_path.exists():
        logger.error("Checksums file not found: %s", checksums_path)
        return {
            "all_passed": False,
            "error": f"Checksums file not found at {checksums_path}",
            "artifacts": {},
            "module_status": {},
        }

    artifacts: dict[str, Any] = {}
    module_status: dict[str, bool] = {
        "pulmo_service": True,
        "risk_service": True,
        "future_risk_service": True,
        "image_service": True,
    }

    with open(checksums_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue

            expected_sha, rel_path = parts[0].strip(), parts[1].strip()
            file_path = base_dir / rel_path
            module = MODULE_MAPPING.get(rel_path, "unknown")

            if not file_path.exists():
                status = "missing"
                actual_sha = None
                passed = False
            else:
                actual_sha = compute_file_sha256(file_path)
                passed = (actual_sha.lower() == expected_sha.lower())
                status = "verified" if passed else "hash_mismatch"

            if not passed:
                if module in module_status:
                    module_status[module] = False

            artifacts[rel_path] = {
                "module": module,
                "expected_sha256": expected_sha,
                "actual_sha256": actual_sha,
                "status": status,
                "passed": passed,
            }

    all_passed = len(artifacts) > 0 and all(a["passed"] for a in artifacts.values())
    verified_count = sum(1 for a in artifacts.values() if a["passed"])
    failed_count = len(artifacts) - verified_count

    return {
        "all_passed": all_passed,
        "total_artifacts": len(artifacts),
        "verified_count": verified_count,
        "failed_count": failed_count,
        "artifacts": artifacts,
        "module_status": module_status,
    }


def main() -> None:
    results = verify_artifacts()
    print("\n=== PulmoAI Cryptographic Artifact Verification ===")
    print(f"Total Artifacts Evaluated: {results.get('total_artifacts', 0)}")
    print(f"Verified Checkpoints:     {results.get('verified_count', 0)}")
    print(f"Failed Checkpoints:       {results.get('failed_count', 0)}\n")

    for path, info in results.get("artifacts", {}).items():
        status_tag = "PASS" if info["passed"] else ("MISSING" if info["status"] == "missing" else "CORRUPT")
        print(f"[{status_tag:7}] {path:<50} (module: {info['module']})")

    print("\nModule Verification Status:")
    for mod, passed in results.get("module_status", {}).items():
        status_str = "HEALTHY" if passed else "FAILED"
        print(f"  - {mod:<22}: {status_str}")

    if results.get("all_passed", False):
        print("\nSUCCESS: All deployed artifacts cryptographically verified.")
        raise SystemExit(0)
    else:
        print("\nFAILURE: One or more deployed artifacts failed verification.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
