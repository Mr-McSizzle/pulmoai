"""Checkpoint integrity verification and manifest generator.

Manages:
  - models/checksums.sha256
  - models/manifest.json
Provides automated verification callable at server startup or via CLI.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("checkpoint_verifier")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MANIFEST_PATH = MODELS_DIR / "manifest.json"
CHECKSUMS_PATH = MODELS_DIR / "checksums.sha256"

EXPECTED_MANIFEST: dict[str, dict[str, Any]] = {
    "single_cell_lung_atlas": {
        "filename": "ensemble_benchmark_best.joblib",
        "relative_path": "models/ensemble_benchmark_best.joblib",
        "sha256": "3c79abb652ac0099a34145c38cbdaf774d673941636a208c8c820be4ac37f03c",
        "size_bytes": 3004664,
        "module": "Single-Cell Lung Atlas Classifier",
        "critical_for_startup": True,
    },
    "nhanes_chronic_risk": {
        "filename": "nhanes_risk_models.joblib",
        "relative_path": "models/nhanes_risk_models.joblib",
        "sha256": "ac86d702fd069919354ca8c20d26168c5a08ce91589ede2eb53f74c8c9a71b7e",
        "size_bytes": 4272101,
        "module": "NHANES Chronic Disease Risk Engine",
        "critical_for_startup": True,
    },
    "medmnist_pneumoniamnist": {
        "filename": "pneumoniamnist_resnet18.pt",
        "relative_path": "models/medmnist/pneumoniamnist_resnet18.pt",
        "sha256": "3d3afa7bb4d4df4cc05aefe8b4ac21bcc697ff88b4f7570c060f5fed0b183c1e",
        "size_bytes": 44778131,
        "module": "Chest X-Ray Pneumonia Screening (MedMNIST)",
        "critical_for_startup": False,
    },
    "medmnist_dermamnist": {
        "filename": "dermamnist_resnet18.pt",
        "relative_path": "models/medmnist/dermamnist_resnet18.pt",
        "sha256": "2747ef9348a9f8a0cc9cff102de6ad63757a4211d4cf18dcc63d22c57b9c9c0e",
        "size_bytes": 44784403,
        "module": "Dermatoscopy Skin Lesions (MedMNIST)",
        "critical_for_startup": False,
    },
    "medmnist_retinamnist": {
        "filename": "retinamnist_resnet18.pt",
        "relative_path": "models/medmnist/retinamnist_resnet18.pt",
        "sha256": "9c7b2ab4bd6b376a94cd722840eab74f839095d64e429e441d8ff33cc359c381",
        "size_bytes": 44782355,
        "module": "Retinal Fundus Photography (MedMNIST)",
        "critical_for_startup": False,
    },
    "medmnist_bloodmnist": {
        "filename": "bloodmnist_resnet18.pt",
        "relative_path": "models/medmnist/bloodmnist_resnet18.pt",
        "sha256": "d684ab03dfd05b96f3115afaecf11920f7e098371d4da87ab8af72179aa3e295",
        "size_bytes": 44785427,
        "module": "Peripheral Blood Smear Microscopy (MedMNIST)",
        "critical_for_startup": False,
    },
    "medmnist_chestmnist": {
        "filename": "chestmnist_resnet18.pt",
        "relative_path": "models/medmnist/chestmnist_resnet18.pt",
        "sha256": "1bef4e29d1984cd2321658a9e297fb110ce7e924fff195fffe3533652b725fce",
        "size_bytes": 44791571,
        "module": "14-Condition Chest Multi-Label Screening (MedMNIST)",
        "critical_for_startup": False,
    },
}


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file in streaming chunks."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def write_manifest_and_checksums() -> None:
    """Write canonical manifest.json and checksums.sha256 to models/."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. manifest.json
    manifest = {
        "schema_version": "1.0.0",
        "generated_timestamp": "2026-10-13T23:35:00Z",
        "description": "Cryptographic SHA-256 manifest for reproducible PulmoAI model checkpoints.",
        "checkpoints": EXPECTED_MANIFEST,
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # 2. checksums.sha256
    lines = []
    for key, item in EXPECTED_MANIFEST.items():
        lines.append(f"{item['sha256']}  {item['relative_path']}")
    with open(CHECKSUMS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("Generated %s and %s", MANIFEST_PATH, CHECKSUMS_PATH)


def verify_checkpoints() -> dict[str, Any]:
    """Verify all checkpoints against expected manifest. Returns detailed verification status."""
    results: dict[str, Any] = {
        "all_critical_passed": True,
        "all_checkpoints_passed": True,
        "checkpoints": {},
    }

    for key, spec in EXPECTED_MANIFEST.items():
        file_path = PROJECT_ROOT / spec["relative_path"]
        if not file_path.exists():
            status = "missing"
            actual_sha = None
            passed = False
        else:
            actual_sha = compute_sha256(file_path)
            passed = (actual_sha.lower() == spec["sha256"].lower())
            status = "verified" if passed else "hash_mismatch"

        if not passed:
            results["all_checkpoints_passed"] = False
            if spec.get("critical_for_startup", False):
                results["all_critical_passed"] = False

        results["checkpoints"][key] = {
            "module": spec["module"],
            "relative_path": spec["relative_path"],
            "expected_sha256": spec["sha256"],
            "actual_sha256": actual_sha,
            "status": status,
            "passed": passed,
            "critical": spec.get("critical_for_startup", False),
        }

    return results


def main() -> None:
    write_manifest_and_checksums()
    verification = verify_checkpoints()
    print("\n=== Model Checkpoint Verification Report ===")
    for key, res in verification["checkpoints"].items():
        status_label = "PASS" if res["passed"] else ("MISSING" if res["status"] == "missing" else "CORRUPT")
        print(f"[{status_label:7}] {res['module']:45} ({res['relative_path']})")
    
    if verification["all_checkpoints_passed"]:
        print("\nAll model checkpoints cryptographically verified against SHA-256 manifest.")
    elif verification["all_critical_passed"]:
        print("\nAll critical core models verified. Optional vision models may be trained via scripts/train_medmnist_models.py.")
    else:
        print("\nCRITICAL: One or more required startup model checkpoints failed verification!")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
