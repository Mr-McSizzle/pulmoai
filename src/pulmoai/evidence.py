"""Source-backed benchmark summaries. Historical audit results are not current clinical validation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPORTS = {
    "lung": "audit_lung_part_a.json",
    "chronic": "audit_nhanes.json",
    "lab": "audit_ocr_pass9b.json",
    "vision": "audit_modality_gate_v3.json",
}


def _metric(label: str, value: Any, *, lower_is_better: bool = False) -> dict:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1:
        raise ValueError("Audit metric must be a finite fraction between zero and one.")
    return {"label": label, "value": value, "lower_is_better": lower_is_better}


def benchmark_evidence(module_status: dict[str, bool], report_dir: Path | None = None) -> dict:
    directory = report_dir or ROOT / "reports"
    config = {
        "lung": ("Single-cell lung classifier", "pulmo_service"),
        "chronic": ("Chronic disease risk", "risk_service"),
        "lab": ("Laboratory report extraction", None),
        "vision": ("Medical image modality gate", "image_service"),
    }
    modules = []
    for key, filename in REPORTS.items():
        name, service = config[key]
        item = {"id": key, "name": name, "source_filename": filename,
                "source_url": f"/api/evidence/reports/{key}", "metrics": [],
                "artifact_integrity": bool(module_status.get(service)) if service else None,
                "evidence_status": "unavailable", "evaluation_scope": "Historical repository audit; not rerun for this release."}
        try:
            raw = (directory / filename).read_bytes()
            audit = json.loads(raw)
            if key == "lung":
                model = audit["configuration_comparison"]["model_c_dual_rule"]
                item.update(cohort="372 sample/library profiles across 16 studies; patient independence is unverified.",
                    protocol="Leave-one-study-out evaluation of the dual-rule procedure.",
                    metrics=[_metric("Overall accuracy", model["accuracy"]),
                             _metric("Balanced accuracy", model["balanced_accuracy"]),
                             _metric("COVID-19 recall", model["covid_recall"])])
            elif key == "chronic":
                metadata = audit["audit_metadata"]
                model = audit["targets"]["diabetes"]["unweighted_metrics"]
                item.update(cohort=f"{metadata['holdout_sample_adults']:,} US adults in the later-cycle holdout.",
                    protocol="Existing diagnosed diabetes status; this is not future incidence.",
                    metrics=[_metric("Diabetes AUROC", model["auroc"]),
                             _metric("Always-negative baseline accuracy", model["accuracy_always_negative"])])
            elif key == "lab":
                clean = audit["condition_summaries"]["clean_pdf"]
                scan = audit["condition_summaries"]["clean_raster_150dpi"]
                item.update(cohort=f"Generated documents: {audit['metadata']['total_synthetic_documents']} documents, {clean['total_entities']} entities per condition.",
                    protocol="Generated holdout layouts; performance on real hospital reports is unknown.",
                    metrics=[_metric("Clean PDF accepted-entity accuracy", clean["accuracy_on_accepted"]),
                             _metric("150-dpi scan accepted-entity accuracy", scan["accuracy_on_accepted"]),
                             _metric("150-dpi scan silent-error rate", scan["silent_error_rate_on_accepted"], lower_is_better=True)])
            else:
                target = audit["target_modalities_true_accept_rates"]["overall_target_summary"]
                unseen = audit["headline_unseen_family_metrics"]
                item.update(cohort=f"{target['total_target_samples_evaluated']:,} target images; separate unseen-family rejection evaluation.",
                    protocol="Modality acceptance and rejection; these are not disease-detection accuracy.",
                    metrics=[_metric("Target true-accept rate", target["overall_true_accept_rate"]),
                             _metric("Unseen-family false-accept rate", unseen["macro_average_left_out_far"], lower_is_better=True),
                             _metric("Worst-family false-accept rate", unseen["worst_left_out_family_far"], lower_is_better=True)])
            item.update(evidence_status="reported", source_sha256=hashlib.sha256(raw).hexdigest())
        except (OSError, KeyError, ValueError, TypeError):
            item.update(metrics=[], cohort="Audit evidence unavailable.", protocol="No replacement metric is generated.")
        modules.append(item)
    return {"modules": modules, "notice": "Benchmark metrics describe a specific evaluated dataset and procedure. A model score is not accuracy or certainty for an individual result. Historical reports do not establish current clinical validation."}
