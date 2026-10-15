# Model Card: Modality Classifier Gate & OOD Rejection

## Model Details
- **Architecture:** 5-Class Convolutional Neural Network (ResNet-18, 64x64 input, 3 channels) with dual-stage heuristic pre-check (pixel standard deviation $\text{std} < 0.05$ and Laplacian edge variance $\text{var} < 5.0$).
- **Classes:** `chest_xray`, `dermoscopy`, `fundus`, `blood_smear`, `other`
- **Artifact Names & Paths:**
  - Deployed Checkpoint: `models/modality_gate_resnet18_v3.pt`
  - Previous Version: `models/modality_gate_resnet18_v2.pt`
- **SHA-256 Checkpoint Hashes:**
  - `modality_gate_resnet18_v3.pt`: `18617379bc35c25a3268b9405ad658c3ee7910370a87dcf71f39bd9d78e89d69`
  - `modality_gate_resnet18_v2.pt`: `bc6f689a14888ec8e85ea7fb9de7fe8456b18bdea1d68e65901abe4c54b91a32`
- **Developers:** PulmoAI Machine Learning Research Group.

## Intended Use & Safety Function
- **Primary Function:** Front-door modality verification and out-of-distribution rejection. Verifies that an image uploaded for medical screening matches the user-selected clinical modality before executing the task-specific screening network.
- **Enforcement Action:** If the gate classifies the image as `other` or detects a modality mismatch, execution of the screening model is refused, and a warning is displayed.

## Quantitative Metrics (Honest Leave-One-Family-Out Cross-Validation)
- **LOFO Macro-Average False-Accept Rate (FAR):** **19.03%**
- **Worst Left-Out Negative Family:** `synthetic_lab_reports` (**100.00%** FAR, 95% CI: [98.91%, 100.0%])
- **Per-Family Left-Out False-Accept Rates (Wilson 95% CIs):**
  - CIFAR-10 Photographic Images: 10.40% (95% CI: [8.02%, 13.38%])
  - BreastMNIST Ultrasound Non-Target: 0.00% (95% CI: [0.00%, 2.40%])
  - OrganCMNIST Abdominal CT (Coronal): 1.00% (95% CI: [0.43%, 2.32%])
  - OrganSMNIST Abdominal CT (Sagittal): 2.80% (95% CI: [1.68%, 4.64%])
  - Synthetic Lab Report Renderings: 100.00% (95% CI: [98.91%, 100.0%])
  - Synthetic Noise & Blanks (Pre-Checked): 0.00% (95% CI: [0.00%, 2.87%])
- **Adversarial / Synthetic Noise Stress-Test ($n=800$, Wilson 95% CIs):**
  - Evaluated families: Gaussian noise ($\sigma \in [0.10, 0.30]$), uniform noise, checkerboards, smooth linear/radial gradients, rendered text pages, screenshots.
  - Overall Rejection Rate: **100.00%** (800 / 800, 95% CI: [99.52%, 100.00%])
  - False-Accept Rate: **0.00%** (0 / 800, 95% CI: [0.00%, 0.48%])
- **Target True-Accept Rate (TAR / Recall) on Held-Out Test Sets:**
  - Chest X-ray: 99.75% (399 / 400)
  - Dermoscopy: 99.00% (396 / 400)
  - Fundus: 99.25% (397 / 400)
  - Blood Smear: 100.00% (400 / 400)
  - Overall Target TAR: **99.50%** (1,592 / 1,600, 95% CI: [99.02%, 99.75%])

## Known Limitations and Clinical Caveats
1. **Unseen Document Modalities Fail Gating:** When document thumbnails / lab report scans are omitted from negative training, the network experiences 100% false-accept rates (predominantly accepted as dermoscopy due to high-contrast edges and beige backgrounds).
2. **Pediatric vs. Adult X-rays:** The gate cannot differentiate between pediatric and adult chest radiographs (training data from PneumoniaMNIST consists of pediatric patients aged 1-5 years from Guangzhou Women and Children's Medical Center).
3. **Unseen Modalities Not Guaranteed Rejection:** Modalities completely absent from negative training (e.g., MRI, PET scans, dental panoramic radiographs) cannot be reliably detected or rejected.
4. **Safety Policy:** Because the LOFO macro FAR is 19.03% (exceeding the strict 10% clinical threshold), a permanent warning banner is displayed across all image screening outputs.
