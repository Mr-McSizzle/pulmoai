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
- **Adversarial / Synthetic Noise Stress-Test ($n=800$, Pass 6 Decomposition):**
  - Evaluated families (n=100 each): Gaussian noise ($\sigma \in [0.10, 0.30]$), uniform noise, checkerboards, smooth linear/radial gradients, rendered document text, desktop screenshots.
  - Component Decomposition: Dual-criterion pre-check (`lap_var < 5.0`) rejected 100/800 (smooth gradients); 5-class ResNet-18 classifier rejected 699/800; 1 false accept (checkerboard pattern).
  - Overall Rejection Rate: **99.88%** (799 / 800, 95% CI: [99.27%, 99.98%])
  - False-Accept Rate: **0.12%** (1 / 800, 95% CI: [0.02%, 0.73%])
  - Holdout Disclosure: These 8 categories were designed post-failure to probe known vulnerabilities and do not constitute an unseen holdout benchmark. Headline generalization metric remains LOFO macro FAR 19.03%.
- **Target True-Accept Rate (TAR / Recall) on Held-Out Test Sets:**
  - Chest X-ray: 99.75% (399 / 400)
  - Dermoscopy: 99.00% (396 / 400)
  - Fundus: 99.25% (397 / 400)
  - Blood Smear: 100.00% (400 / 400)
  - Overall Target TAR: **99.50%** (1,592 / 1,600, 95% CI: [99.02%, 99.75%])

## External Out-of-Distribution (OOD) Benchmark (Pass 9b)
[Tested - reports/ood_benchmark.json:datasets]
To verify open-world generalization beyond the training negative sets, the modality gate ResNet-18 was evaluated against four external unseen image datasets at 64x64 resolution:
| Dataset | Modality / Domain | N Evaluated | False Accepts | False Accept Rate (FAR) | 95% Wilson Score CI | Predictions Breakdown |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SVHN** | Natural scene digits (Street View House Numbers) | 1,000 | 35 | **3.50%** | [2.53%, 4.83%] | other: 965, blood_smear: 16, chest_xray: 9, dermoscopy: 9, fundus: 1 |
| **Flowers102** | Natural photography (Oxford Flowers) | 500 | 63 | **12.60%** | [9.97%, 15.80%] | other: 437, blood_smear: 29, fundus: 18, dermoscopy: 14, chest_xray: 2 |
| **OCTMNIST** | Retinal Optical Coherence Tomography (non-fundus) | 1,000 | 0 | **0.00%** | [0.00%, 0.38%] | other: 1,000, chest_xray: 0, dermoscopy: 0, fundus: 0, blood_smear: 0 |
| **TissueMNIST** | Human kidney cortex histopathology (non-blood) | 1,000 | 0 | **0.00%** | [0.00%, 0.38%] | other: 1,000, chest_xray: 0, dermoscopy: 0, fundus: 0, blood_smear: 0 |

## Adult-to-Pediatric Radiograph Domain Shift Audit (Pass 9b & 9c)
[Tested - reports/cxr_image_stats_experiment.json:perturbation_experiment]
To quantify the hazard of domain shift and investigate image statistics vs anatomical hypotheses:
- **Module Classification:** **Pediatric Benchmark Demonstration Only** (Not for clinical use or adult evaluation).
- **Source Cohort:** ChestMNIST test set (Adult NIH ChestX-ray14 cohort), filtered to adult "no finding" normal chest radiographs ($n = 11,928$).
- **Target Classifier:** PneumoniaMNIST ResNet-18 (`models/medmnist/pneumoniamnist_resnet18.pt`), trained strictly on pediatric patients aged 1-5 years from Guangzhou Women and Children's Medical Center.
- **Evaluation Findings:**
  - Adult normal radiographs classified as Normal: 85 (0.71%)
  - Adult normal radiographs predicted as Pediatric Pneumonia: **11,843** (**99.29%**, 95% Wilson Score CI: [99.12%, 99.42%])
- **Image Statistics Experiment (`reports/cxr_image_stats_experiment.json`):**
  - Pediatric Normal Mean Brightness: 141.50 (std: 50.40)
  - Pediatric Pneumonia Mean Brightness: 145.11 (std: 40.88)
  - Adult Normal Mean Brightness: 127.46 (std: 60.67)
  - Brightness Inversion ($255 - x$): 99.36% Pneumonia (0.64% Normal)
  - Brightness/Contrast Normalized to Pediatric Mean & Std: 99.51% Pneumonia (0.49% Normal)
- **Anatomical Attribution:**
  - [Hypothesis] Prior claims that adult vs pediatric domain shifts stem specifically from rib ossification, thoracic skeletal maturity, or thymic shadow geometry remain an unverified hypothesis. They have not been corroborated by clinical radiologist annotation or Grad-CAM saliency maps, whereas global pixel intensity inversion and normalization similarly preserve >99% pneumonia predictions.
- **Safety Directive & Mandatory Confirmation Gate:**
  - The module is strictly designated **pediatric benchmark demonstration only**.
  - In both the API (`/api/image/predict`) and UI, an explicit confirmation that the patient is a child aged 1-5 (`patient_is_pediatric=True`) is mandatory. If not confirmed, the system immediately refuses execution with HTTP 422 and a clear clinical explanation.

## Known Limitations and Clinical Caveats
1. **No Generic OOD Detection Guarantee:** The modality gate does NOT provide universal out-of-distribution detection. While medical cross-domain images (OCT, kidney histology) achieved 0.00% FAR, natural photography (Flowers102) produced 12.60% FAR and synthetic lab reports in LOFO testing produced 100.00% FAR. Modalities not represented in the training set cannot be guaranteed rejection.
2. **Pediatric Restriction & Anatomical Gate Blindness:** The modality gate identifies images only as `chest_xray` and cannot differentiate pediatric from adult anatomy. Adult radiographs will pass the gate and trigger an catastrophic 99.29% false alarm rate on the pediatric model.
3. **Mandatory UI Warning Banner:** Because LOFO macro FAR is 19.03% and natural image FAR reaches 12.60%, an unambiguous clinical safety disclaimer is permanently enforced across the user interface.
4. **Clinical Decision Support Only:** All image screening outputs are exploratory research decision support tools and do not substitute for certified radiological or pathological evaluation.
