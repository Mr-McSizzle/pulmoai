# Synthetic Demo Samples (Research Use Only)

This directory contains synthetic demonstration files designed for quick-start testing, presentations, and interactive walkthroughs:

1. `synthetic_lab_report.pdf`:
   - Digital PDF containing a standard metabolic and lipid panel (Fasting Glucose, HbA1c, Total Cholesterol, HDL, Triglycerides, Creatinine, Hemoglobin).
   - Use on `/reports` page: Drag and drop to demonstrate instant PyMuPDF digital text layer extraction with 100% numerical accuracy.

2. `synthetic_rotated_scan.png`:
   - High-resolution scan of the identical lab report with +3.5 degree rotation and optical scan noise.
   - Use on `/reports` page: Demonstrates automatic deskew preprocessing and local Tesseract OCR fallback with human verification confirmation.

3. `synthetic_lung_pseudobulk.csv`:
   - Long-format CSV containing all 30,566 gene identifiers in compliance with the model input contract.
   - Use on `/analyze` page: Demonstrates pseudobulk count normalization, CPM log1p calculation, and dual-rule classification.

**Privacy & Provenance Notice:**
All files in this directory are 100% synthetically generated. They contain ZERO real patient data, protected health information (PHI), or clinical records.
