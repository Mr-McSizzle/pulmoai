"""Conservative longitudinal comparisons. Model scores are not disease severity."""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

COMPARISON_VERSION = "longitudinal-v1"
NOTICE = ("Lab direction describes movement relative to a reference interval, not overall health. "
          "Image scores describe model output, not disease severity or treatment response. "
          "Only reviewed, dated, compatible observations are compared.")


def _bounds(text: str) -> tuple[float | None, float | None] | None:
    text = text.strip().replace("–", "-").replace("−", "-").replace("≤", "<=").replace("≥", ">=")
    number = r"(\d+(?:\.\d+)?)"
    both = re.fullmatch(number + r"\s*-\s*" + number + r"(?:\s+[a-zA-Z%/\^\d]+)?", text)
    if both:
        lo, hi = map(float, both.groups())
        return (lo, hi) if hi > lo else None
    one = re.fullmatch(r"([<>]=?)\s*" + number + r"(?:\s+[a-zA-Z%/\^\d]+)?", text)
    if one:
        op, value = one.groups()
        return (None, float(value)) if op.startswith("<") else (float(value), None)
    return None


def _distance(value: float, bounds: tuple[float | None, float | None]) -> float:
    lo, hi = bounds
    return max((lo - value) if lo is not None else 0, (value - hi) if hi is not None else 0, 0)


def build_timeline(patient: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    labs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    images: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    excluded = 0
    for record in records:
        if record.get("review_status") != "reviewed" or not record.get("observed_on"):
            excluded += 1
            continue
        result = record["result"]
        base = {"date": record["observed_on"], "record_id": record["id"], "filename": record["filename"]}
        if record["kind"] == "lab_report":
            for item in result.get("extracted_tests", []):
                value = result.get("reviewed_values", {}).get(item["canonical_name"], item["value"])
                if not isinstance(value, (float, int)) or not math.isfinite(value) or not item.get("unit"):
                    continue
                labs[(item["canonical_name"], item["unit"])].append({**base, "value": value,
                    "display_name": item["display_name"], "reference_range": item["reference_range"],
                    "corrected": value != item["value"]})
        elif record["kind"] == "medical_image":
            if result.get("status") != "success" or not result.get("recognized") or not record.get("series_label"):
                continue
            version = result.get("artifact_id")
            if not version or result.get("staging_notice"):
                continue
            for label, score in (result.get("class_probabilities") or result.get("probabilities") or {}).items():
                if not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1:
                    continue
                images[(result["image_type"], record["series_label"], version, label)].append({**base, "value": score * 100})

    lab_series = []
    for (name, unit), points in sorted(labs.items()):
        points.sort(key=lambda p: (p["date"], p["record_id"]))
        first, last = points[0], points[-1]
        direction, reason = "insufficient_data", "Two reviewed observations on different dates are required."
        change = None
        if len(points) >= 2 and first["date"] != last["date"]:
            endpoints_unique = sum(p["date"] == first["date"] for p in points) == 1 and sum(p["date"] == last["date"] for p in points) == 1
            change = round(last["value"] - first["value"], 6) if endpoints_unique else None
            a, b = _bounds(first["reference_range"]), _bounds(last["reference_range"])
            if not endpoints_unique:
                direction, reason = "not_comparable", "Multiple observations share a baseline or latest date. A directional comparison is ambiguous."
            elif not a or a != b:
                direction, reason = "numeric_change_only", "Reference intervals differ or cannot be parsed; only the numeric change is shown."
            else:
                d1, d2 = _distance(first["value"], a), _distance(last["value"], b)
                if math.isclose(d1, d2, abs_tol=1e-8):
                    direction = "within_range" if d2 == 0 else "unchanged_distance"
                else:
                    direction = "toward_range" if d2 < d1 else "away_from_range"
                reason = "Baseline to latest, using the same reference interval and normalized unit."
        lab_series.append({"id": f"{name}:{unit}", "name": first["display_name"], "unit": unit,
                           "points": points, "direction": direction, "reason": reason, "change": change})
    image_series = []
    for (modality, series, version, label), points in sorted(images.items()):
        points.sort(key=lambda p: (p["date"], p["record_id"]))
        unique_dates = len({p["date"] for p in points}) == len(points)
        change = round(points[-1]["value"] - points[0]["value"], 2) if len(points) > 1 and unique_dates else None
        image_series.append({"id": f"{modality}:{series}:{version}:{label}", "name": label,
            "modality": modality, "series_label": series, "artifact_id": version,
            "unit": "% model score", "points": points, "change": change,
            "direction": "score_change_only", "reason": "Same modality, acquisition series and model artifact. A score change does not establish improvement or decline."})
    return {"patient": patient, "records": records, "lab_series": lab_series, "image_series": image_series,
            "excluded_unreviewed": excluded, "comparison_version": COMPARISON_VERSION, "notice": NOTICE}
