# PulmoAI Model Weights & Distribution Specification

This document details the storage, distribution, provenance, training reproduction scripts, and cryptographic SHA-256 integrity checksums for all trained model checkpoints in PulmoAI.

---

## 1. Distribution and GitHub Releases

Model weights are distributed through GitHub Releases attached to official release tags (e.g., `v0.3.1-audited`). Due to Git file size conventions and repository bloat avoidance, heavy neural network checkpoints (> 10 MB) are managed via release assets rather than raw git tracking.

### Downloading Pre-Trained Weights
To download and verify all pre-trained model checkpoints:
```bash
python scripts/download_checkpoints.py
```
This script downloads assets from the repository's GitHub Release tag, places them into `models/` and `models/medmnist/`, and automatically verifies their SHA-256 integrity against `models/checksums.sha256`.

### Missing Weights Error Check
All PulmoAI services (`pulmoai.service`, `pulmoai.risk_service`, `pulmoai.image_service`, `pulmoai.future_risk_service`) perform upfront existence and integrity checks. If a model file is missing or corrupted, a `FileNotFoundError` or `RuntimeError` is raised with explicit remediation instructions:
```text
FileNotFoundError: Model checkpoint for 'chest_xray' not found at models/medmnist/pneumoniamnist_resnet18.pt.
Run 'python scripts/download_checkpoints.py' or re-train via 'python scripts/train_medmnist_models.py'.
```

---

## 2. In-Repo Training Reproduction

Every model checkpoint can be deterministically reproduced from raw data using in-repo training scripts:

| Model Bundle | Training Script | Input Data Sources | Compute Requirements |
| :--- | :--- | :--- | :--- |
| **scRNA-seq Lung Classifier** (`ensemble_benchmark_best.joblib`) | `scripts/benchmark_models.py` | CELLxGENE Census lung single-cell pseudobulk counts | ~8 mins (8 GB RAM) |
| **Uncalibrated Dual-Rule Lung Model** (`lung_uncalibrated_dual_rule.joblib`) | `scripts/train_dual_rule.py` | CELLxGENE Census lung single-cell pseudobulk counts | ~3 mins |
| **NHANES Chronic Disease Risk Engine** (`nhanes_risk_models.joblib`) | `scripts/train_nhanes_risk_models.py` | CDC NHANES 1999–2018 survey & laboratory data | ~2 mins |
| **Future Mortality Risk Engine** (`future_risk_models_v3.joblib`) | `scripts/audit_future_risk_v3.py` | CDC NHANES Linked Mortality Files (1999–2014) | ~75 secs |
| **Modality Gate ResNet-18** (`modality_gate_resnet18_v3.pt`) | `scripts/audit_modality_gate_v3.py` | MedMNIST v2 (Chest, Derma, Retina, Blood, OrganA, Path, Breast) + CIFAR-100 + Synthetic | ~10 mins (MPS / CUDA) |
| **MedMNIST Screening Classifiers** (`models/medmnist/*.pt`) | `scripts/train_medmnist_models.py` | MedMNIST v2 (PneumoniaMNIST, DermaMNIST, RetinaMNIST, BloodMNIST, ChestMNIST) | ~15 mins (MPS / CUDA) |

---

## 3. Cryptographic SHA-256 Checksums

The following hashes represent the exact binary integrity of the deployed and audited checkpoints:

```text
3c79abb652ac0099a34145c38cbdaf774d673941636a208c8c820be4ac37f03c  models/ensemble_benchmark_best.joblib
f56ef0883e3d1e00f7c496e4b9f8edd9492b3c71210afcf08b7ba860d0fca043  models/lung_uncalibrated_dual_rule.joblib
ac86d702fd069919354ca8c20d26168c5a08ce91589ede2eb53f74c8c9a71b7e  models/nhanes_risk_models.joblib
76bc925d2fd9f955547528be5fda0940a755b647005e5bd36ca181c9f89adcc6  models/future_risk_models_v3.joblib
18617379bc35c25a3268b9405ad658c3ee7910370a87dcf71f39bd9d78e89d69  models/modality_gate_resnet18_v3.pt
3d3afa7bb4d4df4cc05aefe8b4ac21bcc697ff88b4f7570c060f5fed0b183c1e  models/medmnist/pneumoniamnist_resnet18.pt
2747ef9348a9f8a0cc9cff102de6ad63757a4211d4cf18dcc63d22c57b9c9c0e  models/medmnist/dermamnist_resnet18.pt
9c7b2ab4bd6b376a94cd722840eab74f839095d64e429e441d8ff33cc359c381  models/medmnist/retinamnist_resnet18.pt
d684ab03dfd05b96f3115afaecf11920f7e098371d4da87ab8af72179aa3e295  models/medmnist/bloodmnist_resnet18.pt
1bef4e29d1984cd2321658a9e297fb110ce7e924fff195fffe3533652b725fce  models/medmnist/chestmnist_resnet18.pt
```

To verify the integrity of local files against these checksums at any time:
```bash
shasum -a 256 -c models/checksums.sha256
```
