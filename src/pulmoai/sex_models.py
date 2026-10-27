"""Explicit reported-sex routing to locally pinned research models; honest pooled fallback."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REGISTRY_DIR = Path(__file__).resolve().parents[2] / "models" / "lung_sex_specific"
NOTICE = ("The released atlas has no verified reported-sex metadata. The pooled model cannot provide "
          "sex-specific accuracy or sex-adjusted predictions. Sex is never inferred from gene expression.")


def read_registry(pooled: Any, directory: Path = REGISTRY_DIR) -> dict[str, Any]:
    manifest_path = directory / "registry.json"
    if not manifest_path.is_file():
        return {}
    from pulmoai.service import PulmoAIService
    registry = json.loads(manifest_path.read_text(encoding="utf-8"))
    models = {}
    for sex in ("female", "male"):
        entry = registry.get("models", {}).get(sex)
        if not entry:
            continue
        evidence = entry.get("evaluation", {})
        if evidence.get("protocol") != "leave_one_study_out" or evidence.get("source_studies", 0) < 3:
            raise ValueError("Sex-specific model lacks held-out-study evaluation evidence.")
        if any(evidence.get("class_support", {}).get(label, 0) < 10 for label in pooled.label_order):
            raise ValueError("Sex-specific research model requires at least 10 evaluated profiles per class.")
        target = (directory / entry["filename"]).resolve()
        if directory.resolve() not in target.parents or target.suffix != ".joblib":
            raise ValueError("Sex-specific model path must stay inside its registry directory.")
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != entry.get("sha256"):
            raise ValueError("Sex-specific model failed SHA-256 integrity verification.")
        model = PulmoAIService.load(target)
        if model.feature_names != pooled.feature_names or model.label_order != pooled.label_order:
            raise ValueError("Sex-specific model has an incompatible input contract.")
        models[sex] = (model, evidence)
    return models


def select_model(pooled: Any, reported_sex: str) -> tuple[Any, dict[str, Any]]:
    if reported_sex not in {"female", "male", "intersex", "unknown"}:
        raise ValueError("Reported sex must be female, male, intersex or unknown.")
    models = {}
    if reported_sex in {"female", "male"}:
        if not hasattr(pooled, "_sex_registry"):
            pooled._sex_registry = read_registry(pooled)
        models = pooled._sex_registry
    context = {"reported_sex": reported_sex, "model_scope": "pooled", "sex_specific_available": False,
               "clinically_validated": False, "notice": NOTICE}
    if reported_sex in models:
        model, evidence = models[reported_sex]
        context.update(model_scope="sex_specific_research", sex_specific_available=True,
                       notice="Selected a sex-specific research model using reported metadata. Not clinically validated.",
                       evaluation=evidence)
        return model, context
    return pooled, context
