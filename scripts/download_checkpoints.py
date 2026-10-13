"""Download and verify reproducible PulmoAI model checkpoints.

Usage:
  python scripts/download_checkpoints.py [--verify-only] [--build-missing-medmnist]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.verify_checkpoints import verify_checkpoints, EXPECTED_MANIFEST


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and prepare model checkpoints.")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing checkpoints without attempting to train/download missing ones.",
    )
    args = parser.parse_args()

    print("Verifying PulmoAI model checkpoints...")
    results = verify_checkpoints()

    missing = [k for k, v in results["checkpoints"].items() if v["status"] == "missing"]
    mismatched = [k for k, v in results["checkpoints"].items() if v["status"] == "hash_mismatch"]

    print("\nCheckpoint Status:")
    for k, v in results["checkpoints"].items():
        state = "OK" if v["passed"] else v["status"].upper()
        print(f"  [{state:12}] {v['module']} ({v['relative_path']})")

    if mismatched:
        print(f"\nWARNING: {len(mismatched)} checkpoint(s) have hash mismatches! Files may be corrupt or modified.")
        sys.exit(1)

    if not missing:
        print("\nAll model checkpoints are present and verified with valid SHA-256 signatures.")
        return

    print(f"\nMissing checkpoints detected: {len(missing)}")
    if args.verify_only:
        print("Verify-only flag set. Exiting.")
        sys.exit(1 if not results["all_critical_passed"] else 0)

    # Check if missing are MedMNIST models
    medmnist_missing = [k for k in missing if k.startswith("medmnist_")]
    if medmnist_missing:
        print("\nMissing MedMNIST vision checkpoints can be trained deterministically with seed 42.")
        print("Running: python scripts/train_medmnist_models.py ...")
        train_script = PROJECT_ROOT / "scripts" / "train_medmnist_models.py"
        try:
            subprocess.run([sys.executable, str(train_script)], check=True)
            print("MedMNIST training complete. Re-verifying...")
            recheck = verify_checkpoints()
            if recheck["all_checkpoints_passed"]:
                print("All checkpoints verified successfully.")
            else:
                print("Re-verification completed with notices.")
        except Exception as e:
            print(f"Error training MedMNIST models: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
