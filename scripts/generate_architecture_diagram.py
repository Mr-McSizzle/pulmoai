"""Generate a high-resolution, professional architecture diagram for PulmoAI.

Outputs a clean 300 DPI architecture diagram with zero emojis directly to:
/Users/nisarg/Desktop/ml project/architecture_diagram.png
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set non-interactive backend
plt.switch_backend("Agg")

fig = plt.figure(figsize=(19, 12), dpi=300, facecolor="#F8FAFC")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def draw_box(
    x, y, w, h,
    bg_color="#FFFFFF",
    border_color="#CBD5E1",
    corner_radius=1.2,
    linewidth=1.5,
    zorder=2,
):
    """Draw a smooth rounded rectangular card."""
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={corner_radius}",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(rect)


def draw_arrow(x1, y1, x2, y2, color="#64748B", width=1.5, zorder=3):
    """Draw a clean directional connection arrow."""
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=width,
            mutation_scale=12,
            shrinkA=2,
            shrinkB=2,
        ),
        zorder=zorder,
    )


# ---------------------------------------------------------------------------
# Header Section
# ---------------------------------------------------------------------------
ax.text(
    50, 97.0,
    "PULMOAI MULTIMODAL HEALTH PLATFORM",
    ha="center", va="center",
    fontsize=20, fontweight="bold",
    color="#0F172A",
)
ax.text(
    50, 94.6,
    "Full-Stack Architecture: Unified Gateway (FastAPI + React 19) & 4 Audited Clinical ML Engines with Mortality Survival & Modality Gate",
    ha="center", va="center",
    fontsize=10.5, color="#475569",
)

# ---------------------------------------------------------------------------
# Tier 1: Client & User Interface Tier
# ---------------------------------------------------------------------------
draw_box(4, 76.5, 92, 15.5, bg_color="#F1F5F9", border_color="#CBD5E1", linewidth=1.5)
ax.text(
    6, 90.0,
    "TIER 1: CLIENT PRESENTATION LAYER (React 19 + TypeScript + Vite + TailwindCSS)",
    fontsize=10, fontweight="bold", color="#334155",
)
ax.text(
    94, 90.0,
    "Single-Port Deployment: http://127.0.0.1:8000",
    fontsize=9.5, fontweight="bold", color="#0284C7", ha="right",
)

# UI Pages (6 cards)
pages = [
    ("Overview Dashboard", "Route: /", "System health, dataset metrics, subsystem navigation & integrity flags", "#E0F2FE", "#0284C7"),
    ("Single-Cell Atlas", "Route: /analyze", "Pseudobulk upload, dual-rule classifier, whole-cell input contract", "#CCFBF1", "#0D9488"),
    ("Lab Report Reader", "Route: /reports", "Adaptive deskew / blur gate, PyMuPDF text layer, SI-to-US normalization", "#DBEAFE", "#2563EB"),
    ("Chronic & Survival Risk", "Route: /risk", "Prospective 5-yr/10-yr mortality, PCE & IDRS benchmarks, 6 current diagnoses", "#FFE4E6", "#E11D48"),
    ("Image Screening & Gate", "Route: /images", "ResNet-18 modality gate (FAR 7.63%), pediatric X-ray, retina experimental safeguard", "#F3E8FF", "#7C3AED"),
    ("Performance Audit", "Route: /performance", "Leave-one-study-out tables, holdout calibration slopes, Brier skill scores", "#FEF3C7", "#D97706"),
]

for idx, (title, route, desc, card_bg, accent) in enumerate(pages):
    cx = 5.5 + idx * 15.0
    cy = 78.0
    draw_box(cx, cy, 14.0, 10.5, bg_color="#FFFFFF", border_color="#E2E8F0", linewidth=1.2)
    # Accent top bar
    bar = patches.FancyBboxPatch(
        (cx, cy + 9.3), 14.0, 1.2,
        boxstyle="round,pad=0,rounding_size=0.6",
        facecolor=accent, edgecolor=accent,
    )
    ax.add_patch(bar)
    ax.text(cx + 7.0, cy + 7.6, title, ha="center", va="center", fontsize=8.2, fontweight="bold", color="#0F172A")
    ax.text(cx + 7.0, cy + 5.7, route, ha="center", va="center", fontsize=7.2, fontweight="bold", color=accent)
    ax.text(cx + 7.0, cy + 2.8, desc, ha="center", va="center", fontsize=6.6, color="#64748B", wrap=True, multialignment="center")

# Arrow from Tier 1 to Tier 2
draw_arrow(50, 76.5, 50, 70.8, color="#0284C7", width=2.0)
ax.text(50.8, 73.6, "HTTP REST / JSON / Multipart File Transfers (Port 8000)", fontsize=8, color="#0284C7", fontweight="bold")

# ---------------------------------------------------------------------------
# Tier 2: Unified API Gateway Tier (FastAPI)
# ---------------------------------------------------------------------------
draw_box(4, 56.0, 92, 14.5, bg_color="#EFF6FF", border_color="#BFDBFE", linewidth=1.5)
ax.text(
    6, 68.7,
    "TIER 2: UNIFIED API GATEWAY & STATIC SERVER (FastAPI in backend/main.py)",
    fontsize=10, fontweight="bold", color="#1E40AF",
)
ax.text(
    94, 68.7,
    "Interactive OpenAPI Docs: /docs  |  ReDoc: /redoc",
    fontsize=9.5, fontweight="bold", color="#2563EB", ha="right",
)

# API Modules (5 cards)
api_cards = [
    ("Static SPA Delivery", "GET /\nfrontend/dist", "Compiled React bundle served with static asset caching", "#FFFFFF"),
    ("Integrity & Telemetry", "GET /api/health\nGET /api/integrity", "SHA-256 checkpoint verification & subsystem health", "#FFFFFF"),
    ("Single-Cell Predictor", "POST /api/predict\nPOST /api/validate", "Expression normalization, gene count verification, dual rule", "#FFFFFF"),
    ("Lab Report Parser", "POST /api/report/extract\nPOST /api/risk/predict", "PyMuPDF layer extraction, adaptive preproc, cross-sectional risk", "#FFFFFF"),
    ("Future Risk & Vision", "POST /api/risk/future\nPOST /api/image/predict", "Prospective 5/10-yr survival, PCE/IDRS, ResNet-18 modality gate", "#FFFFFF"),
]

for idx, (title, endpoint, desc, card_bg) in enumerate(api_cards):
    cx = 5.5 + idx * 18.2
    cy = 57.5
    draw_box(cx, cy, 16.8, 9.8, bg_color=card_bg, border_color="#DBEAFE", linewidth=1.2)
    ax.text(cx + 8.4, cy + 8.0, title, ha="center", va="center", fontsize=8.3, fontweight="bold", color="#1E3A8A")
    ax.text(cx + 8.4, cy + 5.4, endpoint, ha="center", va="center", fontsize=7.2, fontweight="bold", color="#2563EB", multialignment="center")
    ax.text(cx + 8.4, cy + 2.2, desc, ha="center", va="center", fontsize=6.6, color="#64748B", multialignment="center")

# Arrows from Tier 2 to Tier 3 Engines
engine_x_targets = [15.5, 38.5, 61.5, 84.5]
for tx in engine_x_targets:
    draw_arrow(tx, 56.0, tx, 50.0, color="#475569", width=1.5)

# ---------------------------------------------------------------------------
# Tier 3: Core Machine Learning Engines Tier
# ---------------------------------------------------------------------------
draw_box(4, 27.5, 92, 22.2, bg_color="#F8FAFC", border_color="#CBD5E1", linewidth=1.5)
ax.text(
    6, 48.0,
    "TIER 3: CORE ANALYTICAL & MACHINE LEARNING ENGINES (Audited Research Modules)",
    fontsize=10, fontweight="bold", color="#334155",
)

engines = [
    (
        "Engine 1: Single-Cell Lung Atlas",
        "Dual-Rule Architecture (models/lung_uncalibrated_dual_rule.joblib)",
        "Discrete labels: uncalibrated LinearSVC (COVID-19 recall 4/19)",
        "Probability estimates: Platt sigmoid 5-fold CV (macro-F1 0.456)",
        "Whole-cell contract notice: Melms single-nucleus failure (1/26)",
        "LOGO-CV: 64.8% uncalibrated / 62.9% calibrated accuracy",
        "#0D9488", "#F0FDFA", "#CCFBF1",
    ),
    (
        "Engine 2: Clinical Lab Parser",
        "Adaptive Preprocessing Gate (reports/audit_ocr_part_c.json)",
        "Digital PDF layer parsed first (99.71% clean_pdf accuracy)",
        "Per-image adaptive gate: deskews tilt (+47.15%), preserves blur",
        "Sex-specific intervals (Hb, Cr, HDL) & direct total-chol priority",
        "Synthetic benchmark: n=350/cond across 5 clinical layouts",
        "#2563EB", "#EFF6FF", "#DBEAFE",
    ),
    (
        "Engine 3: Chronic & Future Risk",
        "Dual Estimand Pipeline (Cross-Sectional + Future Mortality Risk)",
        "Current Diagnosis: 6 targets (Cycle J holdout n=5,856, OOF tiers)",
        "Future Mortality Risk: US survey data (47,281 adults, 8,828 deaths)",
        "5-Yr All-Cause: C-index 0.8621, AUROC 0.8713 (temporal holdout 2011-2014)",
        "5-Yr CVD: C-index 0.8948, AUROC 0.9074 (PCE 0.7739, IDRS 0.7798)",
        "#E11D48", "#FFF1F2", "#FFE4E6",
    ),
    (
        "Engine 4: Image Screening & Gate",
        "Modality Gate Classifier (models/modality_gate_resnet18_v2.pt)",
        "5-class ResNet-18 (64x64): chest_xray, derma, fundus, smear, other",
        "Multi-family negative evaluation (audit_modality_gate_v2.json)",
        "Hard rejection of cross-modality mismatches & photographic inputs",
        "Pediatric pneumonia notice (ages 1-5); retina staging disabled",
        "#7C3AED", "#FAF5FF", "#F3E8FF",
    ),
]

for idx, (title, line1, line2, line3, line4, line5, accent, bg_color, header_bg) in enumerate(engines):
    cx = 5.5 + idx * 23.0
    cy = 28.2
    draw_box(cx, cy, 21.5, 18.2, bg_color=bg_color, border_color=accent, linewidth=1.5)
    # Card Header
    hdr = patches.FancyBboxPatch(
        (cx, cy + 15.6), 21.5, 2.6,
        boxstyle="round,pad=0,rounding_size=0.8",
        facecolor=header_bg, edgecolor=accent, linewidth=1.0,
    )
    ax.add_patch(hdr)
    ax.text(cx + 10.75, cy + 16.9, title, ha="center", va="center", fontsize=8.0, fontweight="bold", color=accent)

    # Bullet texts
    bullets = [line1, line2, line3, line4, line5]
    for b_idx, bullet in enumerate(bullets):
        by = cy + 13.7 - b_idx * 2.5
        ax.text(cx + 1.0, by, "-", fontsize=8.0, fontweight="bold", color=accent)
        ax.text(cx + 2.2, by, bullet, fontsize=6.5, color="#1E293B", va="center")

# Arrows from Tier 3 to Tier 4
for tx in engine_x_targets:
    draw_arrow(tx, 28.2, tx, 22.2, color="#64748B", width=1.5)

# ---------------------------------------------------------------------------
# Tier 4: Data Foundation & Verification Layer
# ---------------------------------------------------------------------------
draw_box(4, 5.0, 92, 17.0, bg_color="#F1F5F9", border_color="#CBD5E1", linewidth=1.5)
ax.text(
    6, 20.2,
    "TIER 4: TRAINING COHORTS, VALIDATION PROTOCOLS & QUALITY VERIFICATION",
    fontsize=10, fontweight="bold", color="#334155",
)

data_blocks = [
    (
        "16 Atlas scRNA-seq Studies",
        "1,470,730 cells | 372 donor profiles | 16 studies\nMelms GSE171524 (20 COVID, 7 ctrl) snRNA-seq holdout\nLOGO-CV: uncalibrated 64.8% vs calibrated 62.9%",
        "#0D9488",
    ),
    (
        "Standard Medical Dictionaries",
        "LOINC and UCUM standardized clinical units\nMolar mass factors for SI-to-US conversions\nSex-specific & printed lab reference intervals",
        "#2563EB",
    ),
    (
        "CDC NHANES + Linked Mortality Files",
        "16-Year Prospective Cohort (1999-2014, NCHS LMF)\n47,281 eligible adults (8,828 all-cause, 2,825 CVD deaths)\nTemporal holdout: 2011-2014 (n=11,949) + Cycle J (n=5,856)",
        "#E11D48",
    ),
    (
        "MedMNIST & OOD Benchmarks",
        "ResNet-18 modality gate: 97.15% test accuracy\nNegative classes: CIFAR-100, BreastMNIST, noise, blank\nEmpirical FAR: 7.63% overall, 6.00% on CIFAR photos",
        "#7C3AED",
    ),
]

for idx, (title, content, accent) in enumerate(data_blocks):
    cx = 5.5 + idx * 23.0
    cy = 6.8
    draw_box(cx, cy, 21.5, 11.5, bg_color="#FFFFFF", border_color="#E2E8F0", linewidth=1.2)
    ax.text(cx + 10.75, cy + 9.6, title, ha="center", va="center", fontsize=8.3, fontweight="bold", color=accent)
    ax.text(cx + 10.75, cy + 4.8, content, ha="center", va="center", fontsize=6.8, color="#475569", multialignment="center")

# ---------------------------------------------------------------------------
# Footer Section
# ---------------------------------------------------------------------------
ax.text(
    50, 2.0,
    "Verified for PulmoAI Multimodal Platform | Zero Emojis Compliance | 46 Automated Tests Passing",
    ha="center", va="center", fontsize=8, color="#94A3B8",
)

output_path = Path("/Users/nisarg/Desktop/ml project/architecture_diagram.png")
plt.savefig(output_path, format="png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close(fig)

print(f"Architecture diagram successfully generated at: {output_path}")
